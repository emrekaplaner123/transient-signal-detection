from __future__ import annotations

from pathlib import Path
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from transient_glrt.config import load_config
from transient_glrt.io import get_run_dir, load_thresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/configs/paper.yaml")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    p = load_config(REPO_ROOT / args.config)
    run_dir = get_run_dir(REPO_ROOT, p.run_name)

    thresholds = load_thresholds(run_dir)
    alphas = np.array(sorted(thresholds.keys()), dtype=float)
    idx = int(np.argmin(np.abs(alphas - args.alpha)))
    alpha_used = float(alphas[idx])
    gamma = float(thresholds[alpha_used])

    T_h0_path = run_dir / "T_h0.npy"
    T_h0 = np.load(T_h0_path)
    # from script 01

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.hist(T_h0, bins=60, density=True)
    plt.axvline(gamma, linestyle="--", label=f"gamma (alpha={alpha_used:g})")
    plt.title("Null distribution of test statistic T under H0")
    plt.xlabel("T")
    plt.ylabel("density")
    plt.grid(True)
    plt.legend()

    out = fig_dir / f"null_dist_alpha{alpha_used:g}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

    print("=== NULL DISTRIBUTION FIGURE ===")
    print(f"Saved: {out}")
    print(f"alpha_used={alpha_used:g}, gamma={gamma:.6f}, len(T_h0)={len(T_h0)}")


if __name__ == "__main__":
    main()
