from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from transient_glrt.config import Params


def get_run_dir(repo_root: Path, run_name: str) -> Path:
    """
    Outputs go to experiments/runs/<run_name>/ (create if needed).
    """
    run_dir = repo_root / "experiments" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_params(run_dir: Path, p: Params) -> None:
    """
    Save params snapshot (without huge arrays) for reproducibility.
    """
    # Manually build a JSON-safe dict (avoid numpy arrays)
    data: dict[str, Any] = {
        "run_name": p.run_name,
        "N": p.N,
        "L": p.L,
        "sigma2": p.sigma2,
        "template_type": p.template_type,
        "template_amplitude": p.template_amplitude,
        "tau_mode": p.tau_mode,
        "tau_fixed": p.tau_fixed,
        "M_cal": p.M_cal,
        "M_eval": p.M_eval,
        "alpha_grid": p.alpha_grid,
        "amplitudes": p.amplitudes,
        "seed": p.seed,
    }
    (run_dir / "params.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_thresholds(run_dir: Path, thresholds: dict[float, float]) -> None:
    """
    Save thresholds as JSON. Keys are stored as strings for JSON compatibility.
    """
    data = {str(k): float(v) for k, v in thresholds.items()}
    (run_dir / "thresholds.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_array(run_dir: Path, name: str, arr: np.ndarray) -> None:
    """
    Save numpy array to .npy
    """
    np.save(run_dir / f"{name}.npy", arr)

def load_thresholds(run_dir: Path) -> dict[float, float]:
    import json
    data = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))
    return {float(k): float(v) for k, v in data.items()}
