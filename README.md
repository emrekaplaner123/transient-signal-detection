# transient-signal-detection

Detection of a transient **rectangular pulse** in Gaussian noise using a **GLRT scan statistic** with **unknown arrival time** (tau) and **unknown amplitude** (A).
Thresholds gamma(alpha) are calibrated by **Monte Carlo under H0**, and performance is evaluated with **ROC curves** and **PD vs amplitude**.

---

## Paper
- [PDF](paper/transient-signal-detection.pdf)

---

## Repository layout

- `src/transient_glrt/` — implementation (detector + calibration + evaluation)
- `scripts/` — runnable scripts
- `experiments/configs/` — YAML configs (`quick.yaml`, `paper.yaml`)
- `experiments/runs/` — generated outputs (ignored by git)

---

## Setup (Windows / PowerShell)

```powershell
python -m venv .venv1
.\.venv1\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt


