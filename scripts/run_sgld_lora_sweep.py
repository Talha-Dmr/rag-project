#!/usr/bin/env python3
"""
Run a small SGLD LoRA hyperparameter sweep on the AmbigQA mini dataset.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from src.core.config_loader import load_config
from src.training.base_trainer import TrainerFactory
from src.training.trainers import hallucination_trainer  # noqa: F401


def run_sweep(args: argparse.Namespace) -> None:
    base_config = load_config(args.config)
    training_config = base_config.get("training", {})

    grid = [
        {"lr": 1e-5, "noise": 1e-3},
        {"lr": 5e-6, "noise": 5e-4},
        {"lr": 1e-6, "noise": 1e-4},
    ]

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = Path(args.output_root) / f"sgld_lora_sweep_{timestamp}"
    sweep_root.mkdir(parents=True, exist_ok=True)

    for idx, params in enumerate(grid, 1):
        run_dir = sweep_root / f"run_{idx}_lr{params['lr']}_noise{params['noise']}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Update training config
        training_config["hyperparameters"]["learning_rate"] = params["lr"]
        training_config["hyperparameters"]["noise_scale"] = params["noise"]
        training_config["hyperparameters"]["max_epochs"] = args.epochs
        training_config["checkpointing"]["save_dir"] = str(run_dir)
        training_config["output"]["checkpoint_dir"] = str(run_dir)

        trainer = TrainerFactory.create("hallucination", config=training_config)
        trainer.prepare_data(
            train_data_path=str(Path(args.data_dir) / "train.jsonl"),
            val_data_path=str(Path(args.data_dir) / "val.jsonl"),
        )
        trainer.build_model()

        history = trainer.train(
            num_epochs=args.epochs,
            output_dir=str(run_dir),
            resume_from_checkpoint=None,
        )

        final_metrics = history["val_metrics"][-1] if history["val_metrics"] else {}
        results.append(
            {
                "lr": params["lr"],
                "noise": params["noise"],
                "epochs": args.epochs,
                "metrics": final_metrics,
                "run_dir": str(run_dir),
            }
        )

    results_path = sweep_root / "sweep_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Sweep complete. Results saved to: {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SGLD LoRA sweep (mini).")
    parser.add_argument(
        "--config",
        default="sgld_lora_pilot_ambigqa_mini",
        help="Base config name (without .yaml)",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to data directory with train.jsonl/val.jsonl",
    )
    parser.add_argument(
        "--output-root",
        default="evaluation_results",
        help="Root directory to store sweep outputs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Epochs per sweep run",
    )

    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
