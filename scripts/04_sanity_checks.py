from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np

from transient_glrt.config import load_config
from transient_glrt.template_bank import build_template_bank
from transient_glrt.io import get_run_dir, load_thresholds
from transient_glrt.evaluation import estimate_pf, estimate_pd


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
        help="Target alpha used for sanity checks (closest available).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / args.config

    p = load_config(config_path)
    bank = build_template_bank(p)
    run_dir = get_run_dir(repo_root, p.run_name)
    thresholds = load_thresholds(run_dir)

    alphas = np.array(sorted(thresholds.keys()), dtype=float)
    alpha_target = float(args.alpha)
    idx = int(np.argmin(np.abs(alphas - alpha_target)))
    alpha_used = float(alphas[idx])
    gamma = float(thresholds[alpha_used])

    pf_est = estimate_pf(p, bank, gamma)
    pf = pf_est.rate
    print(f"[CHECK] alpha_used={alpha_used:g}, gamma={gamma:.6f}, PF_hat={pf:.4f}")

    tol = 0.02 if p.M_eval <= 10_000 else 0.01
    if abs(pf - alpha_used) > tol:
        raise RuntimeError(f"PF check failed: |PF_hat-alpha|={abs(pf-alpha_used):.4f} > {tol}")

    pd0_est = estimate_pd(p, bank, gamma, A=0.0)
    pd0 = pd0_est.rate
    print(f"[CHECK] PD_hat(A=0)={pd0:.4f} (should be near PF_hat)")

    if abs(pd0 - pf) > tol:
        raise RuntimeError(f"PD(A=0) check failed: |PD0-PF|={abs(pd0-pf):.4f} > {tol}")

    amps = [float(a) for a in p.amplitudes]
    pd_list = []
    for i, A in enumerate(amps):
        pd_list.append(estimate_pd(p, bank, gamma, A=A, seed_offset=i).rate)

    print("[CHECK] PD vs A:", list(zip(amps, [round(x, 4) for x in pd_list])))

    max_allowed_drop = 0.03 if p.M_eval <= 10_000 else 0.01
    for i in range(1, len(pd_list)):
        if pd_list[i] + max_allowed_drop < pd_list[i - 1]:
            raise RuntimeError(
                f"Monotonicity check failed: PD dropped from {pd_list[i-1]:.4f} to {pd_list[i]:.4f} "
                f"(allowed drop {max_allowed_drop})"
            )

    print("ALL SANITY CHECKS PASSED")


if __name__ == "__main__":
    main()
