from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class Params:
    # From YAML
    run_name: str
    N: int
    L: int
    sigma2: float

    template_type: str
    template_amplitude: float

    tau_mode: str          # "uniform" or "fixed"
    tau_fixed: int         # only used if tau_mode == "fixed"

    M_cal: int
    M_eval: int

    alpha_grid: list[float]
    amplitudes: list[float]

    seed: int

    # Derived (computed in load_config)
    T_max: int
    taus: np.ndarray       # shape (T_max+1,)
    s: np.ndarray          # shape (L,)


def _require_key(d: dict[str, Any], key: str, context: str = "config") -> Any:
    if key not in d:
        raise KeyError(f"Missing required key '{key}' in {context}.")
    return d[key]


def _make_template(template_type: str, amplitude: float, L: int) -> np.ndarray:
    if template_type != "rect":
        raise ValueError(f"Unsupported template.type='{template_type}'. Only 'rect' is supported.")
    # Rectangular pulse: all entries equal to amplitude
    return amplitude * np.ones(L, dtype=float)


def _validate_core(
    *,
    N: int,
    L: int,
    sigma2: float,
    template_type: str,
    template_amplitude: float,
    tau_mode: str,
    tau_fixed: int,
    M_cal: int,
    M_eval: int,
    alpha_grid: list[float],
    amplitudes: list[float],
) -> None:
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")
    if not isinstance(L, int) or L <= 0:
        raise ValueError("L must be a positive integer.")
    if L > N:
        raise ValueError(f"Invalid lengths: L={L} cannot exceed N={N}.")
    if sigma2 <= 0:
        raise ValueError("sigma2 must be > 0 (known noise variance).")

    if template_type != "rect":
        raise ValueError("template.type must be 'rect' for this project setup.")
    if not np.isfinite(template_amplitude):
        raise ValueError("template.amplitude must be a finite number.")

    if tau_mode not in ("uniform", "fixed"):
        raise ValueError("tau_mode must be either 'uniform' or 'fixed'.")

    if not isinstance(M_cal, int) or M_cal <= 0:
        raise ValueError("M_cal must be a positive integer.")
    if not isinstance(M_eval, int) or M_eval <= 0:
        raise ValueError("M_eval must be a positive integer.")

    if not alpha_grid or not isinstance(alpha_grid, list):
        raise ValueError("alpha_grid must be a non-empty list of numbers.")
    for a in alpha_grid:
        if not (0.0 < float(a) < 1.0):
            raise ValueError(f"Each alpha must be in (0,1). Got alpha={a}.")

    if amplitudes is None or not isinstance(amplitudes, list) or len(amplitudes) == 0:
        raise ValueError("amplitudes must be a non-empty list of numbers.")


def load_config(config_path: str | Path) -> Params:
    """
    Load YAML config and return a fully-initialized Params object with derived fields:
    - T_max = N - L
    - taus = [0, 1, ..., T_max]
    - s = rectangular pulse of length L
    """
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config file did not parse to a dictionary.")

    run_name = str(_require_key(cfg, "run_name"))

    N = int(_require_key(cfg, "N"))
    L = int(_require_key(cfg, "L"))
    sigma2 = float(_require_key(cfg, "sigma2"))

    template_cfg = _require_key(cfg, "template")
    if not isinstance(template_cfg, dict):
        raise ValueError("template must be a mapping with keys: type, amplitude.")
    template_type = str(_require_key(template_cfg, "type", context="template"))
    template_amplitude = float(_require_key(template_cfg, "amplitude", context="template"))

    tau_mode = str(_require_key(cfg, "tau_mode"))
    tau_fixed = int(_require_key(cfg, "tau_fixed"))

    M_cal = int(_require_key(cfg, "M_cal"))
    M_eval = int(_require_key(cfg, "M_eval"))

    alpha_grid = [float(x) for x in _require_key(cfg, "alpha_grid")]
    amplitudes = [float(x) for x in _require_key(cfg, "amplitudes")]

    seed = int(_require_key(cfg, "seed"))

    # Validate core (before derived)
    _validate_core(
        N=N,
        L=L,
        sigma2=sigma2,
        template_type=template_type,
        template_amplitude=template_amplitude,
        tau_mode=tau_mode,
        tau_fixed=tau_fixed,
        M_cal=M_cal,
        M_eval=M_eval,
        alpha_grid=alpha_grid,
        amplitudes=amplitudes,
    )

    # Derived
    T_max = N - L
    taus = np.arange(0, T_max + 1, dtype=int)

    if tau_mode == "fixed":
        if not (0 <= tau_fixed <= T_max):
            raise ValueError(f"tau_fixed must be in [0, {T_max}] when tau_mode='fixed'.")

    s = _make_template(template_type, template_amplitude, L)

    return Params(
        run_name=run_name,
        N=N,
        L=L,
        sigma2=sigma2,
        template_type=template_type,
        template_amplitude=template_amplitude,
        tau_mode=tau_mode,
        tau_fixed=tau_fixed,
        M_cal=M_cal,
        M_eval=M_eval,
        alpha_grid=alpha_grid,
        amplitudes=amplitudes,
        seed=seed,
        T_max=T_max,
        taus=taus,
        s=s,
    )
