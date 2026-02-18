# src/paths.py
from pathlib import Path

# Repo root = one level above /src
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_SAMPLE = REPO_ROOT / "data_sample" / "sample.csv"
# Local proprietary data folder (ignored by git)
DATA_PRIVATE_DIR = REPO_ROOT / "Data"

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

from pathlib import Path

# Root of the repository (auto-detected)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -------------------- DATA --------------------
DATA_DIR = PROJECT_ROOT / "data_sample"

DATA_SAMPLE = DATA_DIR / "sample.csv"   # demo dataset

# -------------------- RESULTS --------------------
RESULTS_DIR = PROJECT_ROOT / "results"

LSTM_RESULTS_DIR = RESULTS_DIR / "lstm"
HMM_RESULTS_DIR  = RESULTS_DIR / "hmm"
DM_RESULTS_DIR   = RESULTS_DIR / "dm_tests"

# Create folders automatically if they don't exist
for p in [RESULTS_DIR, LSTM_RESULTS_DIR, HMM_RESULTS_DIR, DM_RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)