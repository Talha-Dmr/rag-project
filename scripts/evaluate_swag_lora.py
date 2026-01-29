#!/usr/bin/env python3
"""
Evaluate LoRA-SWAG (diagonal) by sampling LoRA weights and averaging predictions.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import DataLoader, Subset

from src.core.config_loader import load_config
from src.training.data.nli_dataset import NLIDataset, collate_nli_batch
from src.training.metrics.nli_metrics import NLIMetrics
from src.training.utils.model_utils import load_model_and_tokenizer


def load_swag_state(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def build_dataloader(
    data_path: str,
    tokenizer_name: str,
    max_length: int,
    batch_size: int,
    cache_dir: Optional[str],
    limit: int
) -> DataLoader:
    dataset = NLIDataset(
        data_path=data_path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        cache_dir=cache_dir
    )
    if limit and limit < len(dataset):
        dataset = Subset(dataset, list(range(limit)))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_nli_batch,
        pin_memory=True if torch.cuda.is_available() else False
    )


def sample_lora_weights(
    model: torch.nn.Module,
    swag_state: Dict[str, Any],
    param_cache: Dict[str, torch.nn.Parameter]
) -> None:
    param_names = swag_state.get("param_names") or list(swag_state.get("mean", {}).keys())
    mean = swag_state.get("mean", {})
    sq_mean = swag_state.get("sq_mean", {})

    for name in param_names:
        if name not in param_cache:
            continue
        if name not in mean or name not in sq_mean:
            continue
        mu = mean[name]
        second = sq_mean[name]
        var = torch.clamp(second - mu.pow(2), min=1e-12)
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        sample = mu + eps * std
        param_cache[name].data.copy_(sample.to(param_cache[name].device))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LoRA-SWAG samples")
    parser.add_argument("--config", required=True, help="Config name (without .yaml)")
    parser.add_argument("--swag-path", required=True, help="Path to swag_stats.pt")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional checkpoint dir with model.pt to load base weights"
    )
    parser.add_argument("--data", required=True, help="Path to test.jsonl")
    parser.add_argument("--num-samples", type=int, default=5, help="SWAG samples")
    parser.add_argument("--limit", type=int, default=0, help="Limit eval examples")
    args = parser.parse_args()

    config = load_config(args.config)
    training_config = config.get("training", {})
    model_config = training_config.get("model", {})
    data_config = training_config.get("data", {})
    hyper_config = training_config.get("hyperparameters", {})

    model, _tokenizer = load_model_and_tokenizer(
        model_name=model_config.get("base_model", "microsoft/deberta-v3-small"),
        num_labels=model_config.get("num_labels", 3),
        cache_dir=model_config.get("cache_dir"),
        device="cuda" if torch.cuda.is_available() else "cpu",
        lora_config=model_config.get("lora")
    )
    model.eval()

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        model_path = checkpoint_path / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"model.pt not found in {checkpoint_path}")
        state = torch.load(model_path, map_location=model.device)
        model.load_state_dict(state, strict=False)

    swag_state = load_swag_state(Path(args.swag_path))
    param_cache = {
        name: param for name, param in model.named_parameters()
        if param.requires_grad and "lora" in name.lower()
    }

    dataloader = build_dataloader(
        data_path=args.data,
        tokenizer_name=model_config.get("base_model", "microsoft/deberta-v3-small"),
        max_length=data_config.get("max_seq_length", 128),
        batch_size=hyper_config.get("batch_size", 16),
        cache_dir=model_config.get("cache_dir"),
        limit=args.limit
    )

    metrics = NLIMetrics()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(model.device)

        probs_sum = None

        for _ in range(args.num_samples):
            sample_lora_weights(model, swag_state, param_cache)
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                probs = torch.softmax(outputs.logits, dim=-1)
                probs_sum = probs if probs_sum is None else probs_sum + probs

        avg_probs = probs_sum / float(args.num_samples)
        preds = torch.argmax(avg_probs, dim=-1)

        metrics.update(
            predictions=preds.cpu().tolist(),
            labels=labels.cpu().tolist(),
            probabilities=avg_probs.cpu().tolist()
        )

    results = metrics.compute()
    print("LoRA-SWAG Evaluation")
    print(f"samples: {args.num_samples}")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
