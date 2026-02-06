"""Logit-space Langevin-style sampling probe for NLI model.

Runs multiple noisy gradient steps on logits for a fixed batch and reports
uncertainty stats (entropy/variance/MI proxy) without updating model weights.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--step_size", type=float, default=0.05)
    parser.add_argument("--noise_std", type=float, default=0.1)
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
    parser.add_argument("--output", type=str, default="evaluation_results/logit_sampling_probe.json")
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

    for start in range(0, len(indices), args.batch_size):
        batch_indices = indices[start:start + args.batch_size]
        batch = [dataset[i] for i in batch_indices]

        input_ids = torch.stack([b["input_ids"] for b in batch]).to(device)
        attention_mask = torch.stack([b["attention_mask"] for b in batch]).to(device)

        with torch.no_grad():
            base_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            base_probs = F.softmax(base_logits, dim=-1)

        # Langevin-style sampling on logits (no model update)
        logits = base_logits.detach().clone()
        samples = []

        for _ in range(args.steps):
            logits = logits.detach().requires_grad_(True)
            probs = F.softmax(logits, dim=-1)
            entropy = entropy_from_probs(probs).mean()
            if args.energy == "entropy":
                energy = -entropy
            elif args.energy == "prior_l2":
                energy = 0.5 * (logits - base_logits).pow(2).mean()
            elif args.energy == "prior_kl":
                energy = F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    base_probs,
                    reduction="batchmean"
                )
            else:
                prior = 0.5 * (logits - base_logits).pow(2).mean()
                energy = args.prior_strength * prior - args.entropy_weight * entropy
            grad = torch.autograd.grad(energy, logits)[0]
            noise = torch.randn_like(logits) * args.noise_std
            logits = logits - args.step_size * grad + noise
            samples.append(F.softmax(logits, dim=-1).detach())

        stacked = torch.stack(samples, dim=0)  # [steps, batch, num_labels]
        mean_probs = stacked.mean(dim=0)
        ent = entropy_from_probs(mean_probs)
        var = stacked.var(dim=0).mean(dim=-1)
        # Mutual information proxy: H(mean) - mean(H)
        mean_ent = entropy_from_probs(stacked).mean(dim=0)
        mi = ent - mean_ent

        entropies.extend(ent.cpu().numpy().tolist())
        vars_.extend(var.cpu().numpy().tolist())
        mi_list.extend(mi.cpu().numpy().tolist())

        base_ent = entropy_from_probs(base_probs)
        base_entropy.extend(base_ent.cpu().numpy().tolist())
        base_conf.extend(base_probs.max(dim=-1).values.cpu().numpy().tolist())

    out = {
        "energy": args.energy,
        "prior_strength": args.prior_strength,
        "entropy_weight": args.entropy_weight,
        "base_entropy_mean": float(np.mean(base_entropy)),
        "base_entropy_std": float(np.std(base_entropy)),
        "base_conf_mean": float(np.mean(base_conf)),
        "base_conf_std": float(np.std(base_conf)),
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
