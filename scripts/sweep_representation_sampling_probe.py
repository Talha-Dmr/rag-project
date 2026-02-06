"""Grid sweep for representation-space sampling hyperparameters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument(
        "--output",
        default="evaluation_results/representation_sampling_sweep.json"
    )
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--max-runs", type=int, default=0)
    args = parser.parse_args()

    entropy_weights = [0.5, 1.0, 1.5]
    noise_stds = [0.05, 0.1]
    step_sizes = [0.02, 0.03, 0.04]

    results = []
    run_count = 0

    for ew in entropy_weights:
        for ns in noise_stds:
            for ss in step_sizes:
                out_path = Path("evaluation_results") / (
                    f"representation_probe_ew{ew}_ns{ns}_ss{ss}.json"
                )
                if args.skip_existing and out_path.exists():
                    print("Skipping existing:", out_path)
                    data = json.loads(out_path.read_text())
                    data.update(
                        {"entropy_weight": ew, "noise_std": ns, "step_size": ss}
                    )
                    results.append(data)
                    continue

                cmd = [
                    "PYTHONPATH=.",
                    "venv312/bin/python",
                    "scripts/representation_space_sampling_probe.py",
                    "--model", args.model,
                    "--config", args.config,
                    "--checkpoint", args.checkpoint,
                    "--data", args.data,
                    "--steps", str(args.steps),
                    "--step_size", str(ss),
                    "--noise_std", str(ns),
                    "--energy", "hybrid",
                    "--prior_strength", "1.0",
                    "--entropy_weight", str(ew),
                    "--limit", str(args.limit),
                    "--output", str(out_path),
                ]
                print("Running:", " ".join(cmd))
                subprocess.run(" ".join(cmd), shell=True, check=True)

                data = json.loads(out_path.read_text())
                data.update(
                    {"entropy_weight": ew, "noise_std": ns, "step_size": ss}
                )
                results.append(data)
                run_count += 1

                if args.max_runs and run_count >= args.max_runs:
                    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                    Path(args.output).write_text(json.dumps(results, indent=2))
                    print("Early stop: wrote partial sweep to", args.output)
                    return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print("Wrote sweep to", args.output)


if __name__ == "__main__":
    main()
