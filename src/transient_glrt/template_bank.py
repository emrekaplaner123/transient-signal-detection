from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from transient_glrt.config import Params


@dataclass(frozen=True)
class TemplateBank:
    S: np.ndarray       # shape (K, N)
    taus: np.ndarray    # shape (K,)
    norm2: np.ndarray   # shape (K,)
    N: int
    L: int


def embed_shift(s: np.ndarray, N: int, tau: int) -> np.ndarray:
    """
    Create s_tau in R^N by zero-padding and shifting:
    s_tau[tau : tau+L] = s, and 0 elsewhere.
    """
    L = int(s.shape[0])
    if not (0 <= tau <= N - L):
        raise ValueError(f"tau must be in [0, {N-L}]. Got tau={tau}.")
    out = np.zeros(N, dtype=float)
    out[tau:tau + L] = s
    return out


def build_template_bank(p: Params) -> TemplateBank:
    """
    Build all shifted templates s_tau for tau in p.taus (0..N-L).
    Returns a TemplateBank with cached norms.
    """
    N, L = p.N, p.L
    taus = p.taus
    K = int(taus.shape[0])

    S = np.zeros((K, N), dtype=float)
    for i, tau in enumerate(taus):
        S[i, :] = embed_shift(p.s, N, int(tau))

    norm2 = np.sum(S * S, axis=1)  # ||s_tau||^2 for each tau

    # Safety: norms should be positive
    if np.any(norm2 <= 0):
        raise RuntimeError("Template norm2 contained non-positive values. Check template construction.")

    return TemplateBank(S=S, taus=taus.copy(), norm2=norm2, N=N, L=L)
