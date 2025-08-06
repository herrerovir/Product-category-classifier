from pathlib import Path
import os

# === Project Root ===
ROOT_DIR = Path(__file__).resolve().parent

# === Data Directories ===
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# === Figures Directory ===
FIGURES_DIR = ROOT_DIR / "figures"

# === Model Directory ===
MODELS_DIR = ROOT_DIR / "models"

# === Notebooks Directory ===
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# === Results Directories ===
RESULTS_DIR = ROOT_DIR / "results"
METRICS_RESULTS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# === Ensure all required directories exist ===
for path in [
    RAW_DIR,
    PROCESSED_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    RESULTS_DIR,
    METRICS_RESULTS_DIR,
    PREDICTIONS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# === Set working directory ===
def setup(subdir = None):
    """
    Sets the working directory in Colab after Drive is mounted.
    
    Args:
    ----------
        subdir (str, optional): Subdirectory within the repo to `cd` into (e.g., 'notebooks')
    """
    if subdir:
        os.chdir(ROOT_DIR / subdir)
    else:
        os.chdir(ROOT_DIR)
    print(f"Working directory set to: {os.getcwd()}")