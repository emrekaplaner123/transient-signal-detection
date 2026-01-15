from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np

from transient_glrt.config import load_config
from transient_glrt.template_bank import build_template_bank
from transient_glrt.calibration import calibrate_thresholds_mc
from transient_glrt.io import (
    get_run_dir,
    save_params,
    save_thresholds,
    save_array,
    load_thresholds,
)
from transient_glrt.evaluation import estimate_pf, estimate_pd
from transient_glrt.plotting import plot_roc, plot_pd_vs_amplitude


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
        help="Target alpha used for PD-vs-A figure (closest available).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / args.config

    # Load params + build template bank
    p = load_config(config_path)
    bank = build_template_bank(p)

    # Output directory
    run_dir = get_run_dir(repo_root, p.run_name)

    # 1) Calibration (H0 Monte Carlo) -> thresholds + save diagnostics
    cal = calibrate_thresholds_mc(p, bank)
    save_params(run_dir, p)
    save_thresholds(run_dir, cal.thresholds)
    save_array(run_dir, "T_h0", cal.T_h0)

    print("=== STEP 1: CALIBRATION DONE ===")
    print(f"run_dir: {run_dir}")
    for a in sorted(cal.thresholds.keys()):
        print(f"alpha={a:g}  gamma={cal.thresholds[a]:.6f}")

    # Load thresholds (as the other scripts do)
    thresholds = load_thresholds(run_dir)
    alphas = np.array(sorted(thresholds.keys()), dtype=float)
    gammas = np.array([thresholds[a] for a in alphas], dtype=float)

    # 2) PF estimates (optional but useful to report)
    print("\n=== STEP 2: PF ESTIMATION ===")
    pf_hat = np.zeros_like(alphas)
    for i, a in enumerate(alphas):
        est = estimate_pf(p, bank, gammas[i], seed_offset=i)
        pf_hat[i] = est.rate
        print(f"[PF] alpha={a:g} gamma={gammas[i]:.6f}  PF_hat={est.rate:.4f} ± {1.96*est.se:.4f}")

    # 3) ROC curves for each amplitude
    print("\n=== STEP 3: ROC CURVES ===")
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

    # 4) PD vs amplitude at requested alpha (closest available)
    print("\n=== STEP 4: PD VS AMPLITUDE ===")
    alpha_target = float(args.alpha)
    idx = int(np.argmin(np.abs(alphas - alpha_target)))
    alpha_used = float(alphas[idx])
    gamma_used = float(thresholds[alpha_used])

    print(f"Using alpha_target={alpha_target:g}, closest available alpha={alpha_used:g}, gamma={gamma_used:.6f}")

    amps = np.array(p.amplitudes, dtype=float)
    pd_vs_A = np.zeros_like(amps)
    for i, A in enumerate(amps):
        est = estimate_pd(p, bank, gamma_used, float(A), seed_offset=500 + i)
        pd_vs_A[i] = est.rate
        print(f"[PD] A={A:g}  PD_hat={est.rate:.4f} ± {1.96*est.se:.4f}")

    save_array(run_dir, f"pd_vs_A_alpha{alpha_used:g}_Agrid", amps)
    save_array(run_dir, f"pd_vs_A_alpha{alpha_used:g}_pd", pd_vs_A)

    fig_path = run_dir / "figures" / f"pd_vs_A_alpha{alpha_used:g}.png"
    plot_pd_vs_amplitude(amps, pd_vs_A, title=f"PD vs A (alpha={alpha_used:g})", outpath=fig_path)
    print(f"Saved {fig_path}")

    print("\nDONE (all outputs produced).")


if __name__ == "__main__":
    main()
