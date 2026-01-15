from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from transient_glrt.config import Params
from transient_glrt.template_bank import TemplateBank
from transient_glrt.detector import glrt_scan_statistic


@dataclass(frozen=True)
class CalibrationResult:
    thresholds: dict[float, float]   # alpha -> gamma
    T_h0: np.ndarray                 # simulated statistics under H0 (for diagnostics)


def calibrate_thresholds_mc(p: Params, bank: TemplateBank) -> CalibrationResult:
    """
    Monte Carlo calibration under H0:
      - simulate y ~ N(0, sigma2 I)
      - compute T(y)
      - set gamma(alpha) = empirical (1-alpha)-quantile of T(y)
    """
    rng = np.random.default_rng(p.seed)

    M = int(p.M_cal)
    T_h0 = np.empty(M, dtype=float)

    sigma = float(np.sqrt(p.sigma2))

    for i in range(M):
        y = rng.normal(0.0, sigma, size=p.N)
        T_h0[i] = glrt_scan_statistic(y, bank).T

    thresholds: dict[float, float] = {}
    for alpha in p.alpha_grid:
        a = float(alpha)
        gamma = float(np.quantile(T_h0, 1.0 - a, method="linear"))
        thresholds[a] = gamma

    return CalibrationResult(thresholds=thresholds, T_h0=T_h0)

