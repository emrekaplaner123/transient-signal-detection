from __future__ import annotations

from pathlib import Path
import argparse

from transient_glrt.config import load_config
from transient_glrt.template_bank import build_template_bank
from transient_glrt.calibration import calibrate_thresholds_mc
from transient_glrt.io import get_run_dir, save_params, save_thresholds, save_array


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

    result = calibrate_thresholds_mc(p, bank)

    run_dir = get_run_dir(repo_root, p.run_name)
    save_params(run_dir, p)
    save_thresholds(run_dir, result.thresholds)
    save_array(run_dir, "T_h0", result.T_h0)

    print("=== CALIBRATION DONE ===")
    print(f"run_dir: {run_dir}")
    for a in sorted(result.thresholds.keys()):
        print(f"alpha={a:g}  gamma={result.thresholds[a]:.6f}")


if __name__ == "__main__":
    main()
