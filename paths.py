from pathlib import Path

ROOT_DIR   = Path(__file__).resolve().parent

EXPERIMENTS_DIR = ROOT_DIR / "experiments"
CONFIGS_DIR     = ROOT_DIR / "configs"

JOBS_DIR        = EXPERIMENTS_DIR / "jobs"
OUTPUTS_DIR     = EXPERIMENTS_DIR / "outputs"
LOGS_DIR        = EXPERIMENTS_DIR / "logs"
SCRIPTS_DIR     = EXPERIMENTS_DIR / "scripts"

GRAD_JOBS = JOBS_DIR / "grad"
GRAD_OUTPUTS = OUTPUTS_DIR / "grad"
GRAD_SCRIPTS = SCRIPTS_DIR / "grad"
GRAD_LOGS = LOGS_DIR / "grad"

SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
NSYNTH_DIR = DATA_DIR / "nsynth"