#!/usr/bin/env python
"""Export the frozen base model weights from a PEFT training checkpoint.

The project's LoRA checkpoints store weights with PEFT prefixes such as
``base_model.model.*`` and wrapped projection keys such as ``*.base_layer.*``.
This utility writes a local HuggingFace-compatible base model directory so
offline continuation training can build the model before loading LoRA weights
with ``--init-from``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from transformers import DebertaV2Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir with model.pt")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer dir to copy")
    parser.add_argument("--output-dir", required=True, help="HF-compatible output dir")
    return parser.parse_args()


def unwrap_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    unwrapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not key.startswith("base_model.model."):
            continue
        if ".lora_" in key or ".original_module." in key:
            continue

        new_key = key[len("base_model.model.") :]
        new_key = new_key.replace(".base_layer.", ".")
        new_key = new_key.replace("classifier.modules_to_save.default.", "classifier.")
        new_key = new_key.replace("pooler.modules_to_save.default.", "pooler.")
        unwrapped[new_key] = value

    return unwrapped


def build_deberta_v3_small_config() -> DebertaV2Config:
    return DebertaV2Config(
        vocab_size=128100,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=0,
        initializer_range=0.02,
        layer_norm_eps=1e-7,
        relative_attention=True,
        max_relative_positions=-1,
        position_buckets=256,
        norm_rel_ebd="layer_norm",
        position_biased_input=False,
        pooler_hidden_size=768,
        pooler_dropout=0,
        num_labels=3,
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
        label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
    )


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint)
    tokenizer_dir = Path(args.tokenizer)
    output_dir = Path(args.output_dir)

    model_path = checkpoint_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint model file not found: {model_path}")
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")

    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    unwrapped = unwrap_state_dict(state_dict)
    if not unwrapped:
        raise ValueError(f"No PEFT base model weights found in {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(unwrapped, output_dir / "pytorch_model.bin")
    build_deberta_v3_small_config().save_pretrained(output_dir)

    for path in tokenizer_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, output_dir / path.name)

    print(f"Wrote {len(unwrapped)} weights to {output_dir}")


if __name__ == "__main__":
    main()
