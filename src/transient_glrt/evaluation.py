from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from transient_glrt.config import Params
from transient_glrt.template_bank import TemplateBank
from transient_glrt.detector import glrt_scan_statistic


@dataclass(frozen=True)
class RateEstimate:
    rate: float
    se: float          # standard error (binomial)
    count_true: int
    M: int


def _binom_se(p_hat: float, M: int) -> float:
    if M <= 0:
        return float("nan")
    return float(np.sqrt(p_hat * (1.0 - p_hat) / M))


def estimate_pf(p: Params, bank: TemplateBank, gamma: float, *, seed_offset: int = 0) -> RateEstimate:
    """
    Estimate false alarm probability PF = P(T > gamma | H0).
    """
    rng = np.random.default_rng(p.seed + 10_000 + seed_offset)
    M = int(p.M_eval)
    sigma = float(np.sqrt(p.sigma2))

    hits = 0
    for _ in range(M):
        y = rng.normal(0.0, sigma, size=p.N)
        T = glrt_scan_statistic(y, bank).T
        if T > gamma:
            hits += 1

    rate = hits / M
    return RateEstimate(rate=rate, se=_binom_se(rate, M), count_true=hits, M=M)


def estimate_pd(p: Params, bank: TemplateBank, gamma: float, A: float, *, seed_offset: int = 0) -> RateEstimate:
    """
    Estimate detection probability PD = P(T > gamma | H1).
    H1: y = A*s_tau + v, where tau is either fixed or uniform.
    """
    rng = np.random.default_rng(p.seed + 20_000 + seed_offset)
    M = int(p.M_eval)
    sigma = float(np.sqrt(p.sigma2))

    hits = 0
    for _ in range(M):
        if p.tau_mode == "fixed":
            tau = int(p.tau_fixed)
        else:
            tau = int(rng.integers(0, p.T_max + 1))  # uniform over {0,...,T_max}

        y = A * bank.S[tau] + rng.normal(0.0, sigma, size=p.N)
        T = glrt_scan_statistic(y, bank).T
        if T > gamma:
            hits += 1

    rate = hits / M
    return RateEstimate(rate=rate, se=_binom_se(rate, M), count_true=hits, M=M)
