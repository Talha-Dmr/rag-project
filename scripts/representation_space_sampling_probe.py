"""Representation-space Langevin-style sampling probe for NLI model.

This probe perturbs the penultimate classifier representation (the input to the
classification head) instead of logits. It reports uncertainty stats
(entropy/variance/MI proxy) without updating model weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.core.config_loader import load_config
from src.training.base_trainer import TrainerFactory
from src.training.trainers import hallucination_trainer  # noqa: F401
from src.training.utils.model_utils import load_model_and_tokenizer
from src.training.data.nli_dataset import NLIDataset


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1)


def resolve_classifier(model: torch.nn.Module) -> torch.nn.Module:
    """Find classifier head module across plain HF and PEFT-wrapped models."""
    candidates = []
    if hasattr(model, "classifier"):
        candidates.append(getattr(model, "classifier"))
    if hasattr(model, "base_model"):
        base_model = getattr(model, "base_model")
        if hasattr(base_model, "classifier"):
            candidates.append(getattr(base_model, "classifier"))
        if hasattr(base_model, "model") and hasattr(base_model.model, "classifier"):
            candidates.append(getattr(base_model.model, "classifier"))

    for mod in candidates:
        if isinstance(mod, torch.nn.Module):
            return mod

    for name, mod in model.named_modules():
        if name.endswith("classifier"):
            return mod

    raise RuntimeError("Could not find classifier module for representation-space probe.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--step_size", type=float, default=0.02)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument(
        "--energy",
        choices=["entropy", "prior_l2", "prior_kl", "hybrid"],
        default="hybrid"
    )
    parser.add_argument("--prior_strength", type=float, default=1.0)
    parser.add_argument("--entropy_weight", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results/representation_sampling_probe.json"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.config:
        cfg = load_config(args.config).get("training", {})
        trainer = TrainerFactory.create("hallucination", config=cfg)
        trainer.build_model()
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        model = trainer.model
    else:
        model, _ = load_model_and_tokenizer(args.model, num_labels=3, device=device)
        if args.checkpoint:
            ckpt = Path(args.checkpoint)
            model_path = ckpt / "model.pt" if ckpt.is_dir() else ckpt
            state = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(state)
    model.eval()

    classifier = resolve_classifier(model)

    dataset = NLIDataset(
        data_path=args.data,
        tokenizer_name=args.model,
        max_length=args.max_length,
        cache_dir="./models/training"
    )

    indices = list(range(min(args.limit, len(dataset))))

    entropies = []
    vars_ = []
    mi_list = []
    base_entropy = []
    base_conf = []
    rep_norms = []

    rep_cache: dict[str, torch.Tensor] = {}

    def capture_representation(
        _module: torch.nn.Module,
        module_input: tuple[torch.Tensor, ...],
        _module_output: torch.Tensor
    ) -> None:
        if not module_input:
            return
        rep_cache["z"] = module_input[0].detach()

    hook_handle = classifier.register_forward_hook(capture_representation)

    try:
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start:start + args.batch_size]
            batch = [dataset[i] for i in batch_indices]

            input_ids = torch.stack([b["input_ids"] for b in batch]).to(device)
            attention_mask = torch.stack([b["attention_mask"] for b in batch]).to(device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                base_logits = out.logits
                base_probs = F.softmax(base_logits, dim=-1)

            base_z = rep_cache.pop("z", None)
            if base_z is None:
                raise RuntimeError(
                    "Failed to capture classifier input representation. "
                    "Classifier hook did not receive inputs."
                )
            base_z = base_z.to(device)
            rep_norms.extend(base_z.norm(dim=-1).detach().cpu().numpy().tolist())

            # Langevin-style sampling in representation space (no model update)
            z = base_z.detach().clone()
            samples = []

            for _ in range(args.steps):
                z = z.detach().requires_grad_(True)
                logits = classifier(z)
                probs = F.softmax(logits, dim=-1)
                entropy = entropy_from_probs(probs).mean()

                if args.energy == "entropy":
                    energy = -entropy
                elif args.energy == "prior_l2":
                    energy = 0.5 * (z - base_z).pow(2).mean()
                elif args.energy == "prior_kl":
                    energy = F.kl_div(
                        F.log_softmax(logits, dim=-1),
                        base_probs,
                        reduction="batchmean"
                    )
                else:
                    prior = 0.5 * (z - base_z).pow(2).mean()
                    energy = args.prior_strength * prior - args.entropy_weight * entropy

                grad = torch.autograd.grad(energy, z)[0]
                noise = torch.randn_like(z) * args.noise_std
                z = z - args.step_size * grad + noise
                samples.append(F.softmax(classifier(z), dim=-1).detach())

            stacked = torch.stack(samples, dim=0)  # [steps, batch, num_labels]
            mean_probs = stacked.mean(dim=0)
            ent = entropy_from_probs(mean_probs)
            var = stacked.var(dim=0).mean(dim=-1)
            mean_ent = entropy_from_probs(stacked).mean(dim=0)
            mi = ent - mean_ent

            entropies.extend(ent.cpu().numpy().tolist())
            vars_.extend(var.cpu().numpy().tolist())
            mi_list.extend(mi.cpu().numpy().tolist())

            base_ent = entropy_from_probs(base_probs)
            base_entropy.extend(base_ent.cpu().numpy().tolist())
            base_conf.extend(base_probs.max(dim=-1).values.cpu().numpy().tolist())
    finally:
        hook_handle.remove()

    out = {
        "space": "representation",
        "energy": args.energy,
        "prior_strength": args.prior_strength,
        "entropy_weight": args.entropy_weight,
        "base_entropy_mean": float(np.mean(base_entropy)),
        "base_entropy_std": float(np.std(base_entropy)),
        "base_conf_mean": float(np.mean(base_conf)),
        "base_conf_std": float(np.std(base_conf)),
        "rep_norm_mean": float(np.mean(rep_norms)),
        "rep_norm_std": float(np.std(rep_norms)),
        "entropy_mean": float(np.mean(entropies)),
        "entropy_std": float(np.std(entropies)),
        "variance_mean": float(np.mean(vars_)),
        "variance_std": float(np.std(vars_)),
        "mi_mean": float(np.mean(mi_list)),
        "mi_std": float(np.std(mi_list)),
        "steps": args.steps,
        "step_size": args.step_size,
        "noise_std": args.noise_std,
        "limit": args.limit,
        "seed": args.seed,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
