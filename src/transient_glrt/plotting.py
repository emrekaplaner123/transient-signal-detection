from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_roc(pf: np.ndarray, pd: np.ndarray, *, title: str, outpath: Path) -> None:
    plt.figure()
    plt.plot(pf, pd, marker="o")
    plt.xlabel("P_F")
    plt.ylabel("P_D")
    plt.title(title)
    plt.grid(True)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

def plot_pd_vs_amplitude(A: np.ndarray, pd: np.ndarray, *, title: str, outpath: Path) -> None:
    plt.figure()
    plt.plot(A, pd, marker="o")
    plt.xlabel("Amplitude A")
    plt.ylabel("P_D")
    plt.title(title)
    plt.grid(True)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
