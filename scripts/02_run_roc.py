from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np

from transient_glrt.config import load_config
from transient_glrt.template_bank import build_template_bank
from transient_glrt.io import get_run_dir, load_thresholds, save_array
from transient_glrt.evaluation import estimate_pf, estimate_pd
from transient_glrt.plotting import plot_roc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/quick.yaml",
        help="Path to YAML config relative to repo root",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / args.config

    p = load_config(config_path)
    bank = build_template_bank(p)

    run_dir = get_run_dir(repo_root, p.run_name)
    thresholds = load_thresholds(run_dir)

    alphas = np.array(sorted(thresholds.keys()), dtype=float)
    gammas = np.array([thresholds[a] for a in alphas], dtype=float)

    # PF check (empirical)
    pf_hat = np.zeros_like(alphas)
    for i, a in enumerate(alphas):
        est = estimate_pf(p, bank, gammas[i], seed_offset=i)
        pf_hat[i] = est.rate
        print(f"[PF] alpha={a:g} gamma={gammas[i]:.6f}  PF_hat={est.rate:.4f} ± {1.96*est.se:.4f}")

    # ROC for each amplitude
    for A in p.amplitudes:
        pd_hat = np.zeros_like(alphas)
        for i, _a in enumerate(alphas):
            est = estimate_pd(p, bank, gammas[i], float(A), seed_offset=100 + i)
            pd_hat[i] = est.rate

        save_array(run_dir, f"roc_pf_A{A:g}", pf_hat)
        save_array(run_dir, f"roc_pd_A{A:g}", pd_hat)

        fig_path = run_dir / "figures" / f"roc_A{A:g}.png"
        plot_roc(pf_hat, pd_hat, title=f"ROC (A={A:g})", outpath=fig_path)
        print(f"[ROC] Saved {fig_path}")

    print("DONE")


if __name__ == "__main__":
    main()

