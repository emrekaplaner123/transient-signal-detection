# transient-signal-detection

GLRT scan detection of a transient **rectangular pulse** with **unknown arrival time** and **unknown amplitude**.
Thresholds are calibrated by **Monte Carlo under H0**, and performance is evaluated with **ROC curves** and **PD vs amplitude**.

---

## Project structure

- `src/transient_glrt/` — implementation (config, template bank, detector, calibration, evaluation, plotting)
- `scripts/` — runnable experiment scripts (01–05)
- `experiments/configs/` — YAML configs (`quick.yaml`, `paper.yaml`)
- `experiments/runs/` — generated outputs (ignored by git)

---

## Setup (Windows / PowerShell)

### 1) Create & activate a virtual environment (example)
If you already have a venv, you can skip this.

```powershell
python -m venv .venv1
.\.venv1\Scripts\Activate.ps1

