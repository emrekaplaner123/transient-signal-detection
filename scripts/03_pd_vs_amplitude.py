from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np

from transient_glrt.config import load_config
from transient_glrt.template_bank import build_template_bank
from transient_glrt.io import get_run_dir, load_thresholds, save_array
from transient_glrt.evaluation import estimate_pd
from transient_glrt.plotting import plot_pd_vs_amplitude


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/quick.yaml",
        help="Path to YAML config relative to repo root",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Target false alarm level used to pick gamma (closest available).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / args.config

    p = load_config(config_path)
    bank = build_template_bank(p)

    run_dir = get_run_dir(repo_root, p.run_name)
    thresholds = load_thresholds(run_dir)

    alpha_target = float(args.alpha)

    # pick closest available alpha from thresholds
    alphas = np.array(sorted(thresholds.keys()), dtype=float)
    idx = int(np.argmin(np.abs(alphas - alpha_target)))
    alpha_used = float(alphas[idx])
    gamma = float(thresholds[alpha_used])

    print(f"Using alpha_target={alpha_target:g}, closest available alpha={alpha_used:g}, gamma={gamma:.6f}")

    amplitudes = np.array(p.amplitudes, dtype=float)
    pd_hat = np.zeros_like(amplitudes)

    for i, A in enumerate(amplitudes):
        est = estimate_pd(p, bank, gamma, float(A), seed_offset=i)
        pd_hat[i] = est.rate
        print(f"[PD] A={A:g}  PD_hat={est.rate:.4f} ± {1.96*est.se:.4f}")

    save_array(run_dir, f"pd_vs_A_alpha{alpha_used:g}_Agrid", amplitudes)
    save_array(run_dir, f"pd_vs_A_alpha{alpha_used:g}_pd", pd_hat)

    fig_path = run_dir / "figures" / f"pd_vs_A_alpha{alpha_used:g}.png"
    plot_pd_vs_amplitude(amplitudes, pd_hat, title=f"PD vs A (alpha={alpha_used:g})", outpath=fig_path)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
