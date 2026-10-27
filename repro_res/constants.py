from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

BATCH_SIZE = 128
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42