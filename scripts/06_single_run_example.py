from __future__ import annotations

from pathlib import Path
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

# --- Make `src/` importable when running from terminal ---
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from transient_glrt.config import load_config
from transient_glrt.template_bank import build_template_bank
from transient_glrt.io import get_run_dir, load_thresholds


def _compute_scan_statistic(y: np.ndarray, bank, sigma2: float):

    S = bank.S
    norm2 = bank.norm2

    dots = S @ y  # shape (num_shifts,)
    z = dots / np.sqrt(sigma2 * norm2)
    T_tau = z**2

    k = int(np.argmax(T_tau))
    T = float(T_tau[k])
    tau_hat = int(bank.taus[k])
    A_hat = float(dots[k] / norm2[k])

    return T, tau_hat, A_hat, T_tau


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/configs/paper.yaml")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--A_true", type=float, default=0.5)
    parser.add_argument("--tau_true", type=int, default=123)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config_path = REPO_ROOT / args.config
    p = load_config(config_path)
    bank = build_template_bank(p)

    run_dir = get_run_dir(REPO_ROOT, p.run_name)
    thresholds = load_thresholds(run_dir)

    # pick gamma for requested alpha (or closest available)
    alphas = np.array(sorted(thresholds.keys()), dtype=float)
    idx = int(np.argmin(np.abs(alphas - args.alpha)))
    alpha_used = float(alphas[idx])
    gamma = float(thresholds[alpha_used])

    # --- Build signals from the bank ---
    # Find the row corresponding to tau_true
    if args.tau_true not in set(bank.taus):
        raise ValueError(f"tau_true={args.tau_true} not in bank.taus range [{bank.taus[0]}, {bank.taus[-1]}].")
    k_true = int(list(bank.taus).index(args.tau_true))
    s_full = bank.S[k_true].astype(float)  # length N, pulse embedded at tau_true

    rng = np.random.default_rng(args.seed)
    v0 = rng.normal(0.0, np.sqrt(p.sigma2), size=p.N)
    v1 = rng.normal(0.0, np.sqrt(p.sigma2), size=p.N)

    y_h0 = v0
    y_h1 = args.A_true * s_full + v1

    # --- Run the test on the H1 example ---
    T, tau_hat, A_hat, T_tau = _compute_scan_statistic(y_h1, bank, p.sigma2)
    decision = "Reject H0 (declare signal)" if T > gamma else "Fail to reject H0"

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) Template pulse (within length L)
    # --- Get the short template pulse (length L) ---
    # Your Params doesn't store it, so we take it from the bank if available.
    if hasattr(bank, "s"):
        s_short = np.asarray(bank.s, dtype=float)  # length L (common)
    elif hasattr(bank, "template"):
        s_short = np.asarray(bank.template, dtype=float)  # length L (alternative name)
    else:
        # Fallback: rectangular pulse of ones (since that's your current model)
        s_short = np.ones(p.L, dtype=float)

    plt.figure()
    plt.plot(np.arange(len(s_short)), s_short)
    plt.title("Template pulse s (length L)")
    plt.xlabel("sample index (within pulse)")
    plt.ylabel("amplitude")
    plt.grid(True)
    out = fig_dir / "template_pulse.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

    # 2) y under H0
    plt.figure()
    plt.plot(np.arange(p.N), y_h0)
    plt.title("Observation y under H0 (noise only)")
    plt.xlabel("sample index")
    plt.ylabel("y")
    plt.grid(True)
    out = fig_dir / "y_h0.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

    # 3) y under H1 (single realization)
    plt.figure()
    plt.plot(np.arange(p.N), y_h1, label="y (H1)")
    plt.axvline(args.tau_true, linestyle="--", color ="green",label=f"true tau={args.tau_true}")
    plt.axvline(tau_hat, linestyle="--", color="red", label=f"tau_hat={tau_hat}")
    plt.title(f"Observation y under H1 (A={args.A_true:g})")
    plt.xlabel("sample index")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    out = fig_dir / "y_h1.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

    # 4) Scan statistic vs tau + threshold line
    plt.figure()
    plt.plot(np.array(bank.taus), T_tau)
    plt.axhline(gamma, linestyle="--", label=f"gamma (alpha={alpha_used:g})")
    plt.axvline(args.tau_true, linestyle="--", color ="red" ,label=f"true tau={args.tau_true}")
    plt.axvline(tau_hat, linestyle="--", color = "green", label=f"tau_hat={tau_hat}")
    plt.title("Scan statistic $T_\\tau$ across shifts")
    plt.xlabel("tau")
    plt.ylabel("T_tau")
    plt.grid(True)
    plt.legend()
    out = fig_dir / "scan_statistic_example.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

    # Save a JSON summary (easy to paste into paper/table)
    summary = {
        "alpha_used": alpha_used,
        "gamma": gamma,
        "A_true": float(args.A_true),
        "tau_true": int(args.tau_true),
        "A_hat": A_hat,
        "tau_hat": tau_hat,
        "T": T,
        "decision": decision,
    }
    with (run_dir / "single_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print one LaTeX-ready row
    print("=== SINGLE RUN EXAMPLE ===")
    print(f"alpha_used={alpha_used:g}, gamma={gamma:.6f}")
    print(f"A_true={args.A_true:g}, tau_true={args.tau_true}, A_hat={A_hat:.3f}, tau_hat={tau_hat}, T={T:.3f}")
    print(f"Decision: {decision}")
    print("LaTeX row:")
    print(f"{args.A_true:g} & {args.tau_true} & {A_hat:.3f} & {tau_hat} & {T:.3f} & {gamma:.3f} & {decision} \\\\")


if __name__ == "__main__":
    main()
