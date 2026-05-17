"""Read result CSVs and produce plots for the milestone-2 report.

Expects each method's CSV to follow the schema produced by src.eval.run_eval:
    episode, image_idx, beta_pred, score_before, score_after, reward

Usage:
    python -m src.eval.report_plots \\
        --csv identity=results/identity.csv random=results/random.csv ppo=results/ppo.csv \\
        --out-dir reports/milestone2/figures
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> List[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _to_floats(rows: List[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=np.float64)


def bar_chart_mean_reward(method_rows: Dict[str, List[dict]], out_path: Path) -> None:
    methods = list(method_rows.keys())
    means = [_to_floats(method_rows[m], "reward").mean() for m in methods]
    stds = [_to_floats(method_rows[m], "reward").std() for m in methods]

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(len(methods))
    ax.bar(xs, means, yerr=stds, capsize=5, color="#4C72B0", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel("Mean TOPIQ delta (reward)")
    ax.set_title("Per-method mean reward on toy eval split")
    ax.axhline(0.0, color="k", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def beta_pred_histogram(method_rows: Dict[str, List[dict]], out_path: Path) -> None:
    """How does each method distribute its predicted beta? Useful sanity figure."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, rows in method_rows.items():
        beta = _to_floats(rows, "beta_pred")
        ax.hist(beta, bins=30, alpha=0.5, label=m, density=True)
    ax.set_xlabel("predicted beta")
    ax.set_ylabel("density")
    ax.set_title("Predicted beta distribution across methods")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_kv_pairs(items: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"--csv items must look like name=path/to.csv, got {it!r}")
        name, path = it.split("=", 1)
        out[name] = Path(path)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", nargs="+", required=True, help="name=path pairs")
    p.add_argument("--out-dir", default="reports/milestone2/figures")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = parse_kv_pairs(args.csv)
    method_rows: Dict[str, List[dict]] = {n: _read_csv(p) for n, p in csv_paths.items()}

    bar_chart_mean_reward(method_rows, out_dir / "mean_reward_bar.png")
    beta_pred_histogram(method_rows, out_dir / "beta_pred_hist.png")
    print(f"[report_plots] wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
