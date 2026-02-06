#!/usr/bin/env python3
import argparse
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RunResult:
    method: str
    eta: float
    burn_in: int
    steps: int
    mean: float
    std: float
    accept_rate: float | None


def ula_sample(eta: float, steps: int, burn_in: int, seed: int, x0: float = 5.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = x0
    samples: list[float] = []
    for t in range(steps):
        grad = -x  # grad log N(0,1)
        x = x + eta * grad + math.sqrt(2.0 * eta) * rng.normal()
        if t >= burn_in:
            samples.append(x)
    return np.asarray(samples, dtype=float)


def mala_sample(
    eta: float, steps: int, burn_in: int, seed: int, x0: float = 5.0
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    x = x0
    samples: list[float] = []
    accepts = 0

    for t in range(steps):
        grad = -x
        mean = x + eta * grad
        x_prop = mean + math.sqrt(2.0 * eta) * rng.normal()

        # log target (up to constant)
        logp = -0.5 * x * x
        logp_prop = -0.5 * x_prop * x_prop

        # proposal densities (constants cancel)
        mean_prop = x_prop + eta * (-x_prop)
        log_q_forward = -0.5 * ((x_prop - mean) ** 2) / (2.0 * eta)
        log_q_backward = -0.5 * ((x - mean_prop) ** 2) / (2.0 * eta)

        log_alpha = logp_prop + log_q_backward - logp - log_q_forward
        if math.log(rng.random()) < log_alpha:
            x = x_prop
            accepts += 1

        if t >= burn_in:
            samples.append(x)

    return np.asarray(samples, dtype=float), accepts / max(1, steps)


def run_suite(seed: int) -> list[RunResult]:
    results: list[RunResult] = []

    # ULA configs
    ula_runs = [
        ("ULA", 0.01, 1000, 6000),
        ("ULA", 0.005, 5000, 30000),
        ("ULA", 0.01, 5000, 30000),
    ]
    for method, eta, burn_in, steps in ula_runs:
        samples = ula_sample(eta, steps, burn_in, seed)
        results.append(
            RunResult(
                method=method,
                eta=eta,
                burn_in=burn_in,
                steps=steps,
                mean=float(samples.mean()),
                std=float(samples.std()),
                accept_rate=None,
            )
        )

    # MALA configs
    mala_runs = [
        ("MALA", 0.05, 2000, 20000),
        ("MALA", 0.1, 2000, 20000),
    ]
    for method, eta, burn_in, steps in mala_runs:
        samples, acc = mala_sample(eta, steps, burn_in, seed)
        results.append(
            RunResult(
                method=method,
                eta=eta,
                burn_in=burn_in,
                steps=steps,
                mean=float(samples.mean()),
                std=float(samples.std()),
                accept_rate=acc,
            )
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy Langevin sanity checks on N(0,1).")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    results = run_suite(args.seed)
    print("method\teta\tburn_in\tsteps\tmean\tstd\taccept")
    for r in results:
        acc = "-" if r.accept_rate is None else f"{r.accept_rate:.3f}"
        print(f"{r.method}\t{r.eta:.4f}\t{r.burn_in}\t{r.steps}\t{r.mean:.3f}\t{r.std:.3f}\t{acc}")


if __name__ == "__main__":
    main()
