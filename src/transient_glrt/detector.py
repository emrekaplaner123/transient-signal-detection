from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from transient_glrt.template_bank import TemplateBank


@dataclass(frozen=True)
class DetectionResult:
    T: float
    tau_hat: int
    A_hat: float  # amplitude estimate at tau_hat (optional but useful)


def glrt_scan_statistic(y: np.ndarray, bank: TemplateBank) -> DetectionResult:
    """
    Compute GLRT scan statistic:
        T(y) = max_tau (s_tau^T y)^2 / ||s_tau||^2
    using the provided TemplateBank.

    Returns:
        T: max value
        tau_hat: argmax shift
        A_hat: (s_tau_hat^T y) / ||s_tau_hat||^2
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if y.shape[0] != bank.N:
        raise ValueError(f"y must have length N={bank.N}. Got {y.shape[0]}.")

    # Correlations for all shifts: c_tau = s_tau^T y
    c = bank.S @ y  # shape (K,)

    # Statistic per shift: (c_tau^2) / ||s_tau||^2
    stats = (c * c) / bank.norm2

    idx = int(np.argmax(stats))
    T = float(stats[idx])
    tau_hat = int(bank.taus[idx])

    # Optional amplitude estimate at maximizing shift
    A_hat = float(c[idx] / bank.norm2[idx])

    return DetectionResult(T=T, tau_hat=tau_hat, A_hat=A_hat)


def decide(T: float, gamma: float) -> bool:
    """
    Decision rule:
        decide H1 iff T > gamma
    Returns True if H1, False if H0.
    """
    return bool(T > gamma)
