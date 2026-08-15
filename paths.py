"""Project paths, and the single point where the root ``.env`` enters Python.

Importing this module loads ``<repo root>/.env`` into ``os.environ``. Every entry
point already imports it at module scope, so the variables are in place before
Hydra resolves any ``${oc.env:...}`` interpolation.

Real environment variables always win over ``.env`` — a SLURM job launched with
``--export=ALL`` inherits the submitting shell, and that should not be silently
overridden by a file on a shared filesystem.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Minimal ``KEY=VALUE`` reader. Stdlib only, so no extra pixi dependency."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and value:
            os.environ.setdefault(key, value)


_load_dotenv(ROOT_DIR / ".env")

SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"

EXPERIMENTS_DIR = ROOT_DIR / "experiments"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
JOBS_DIR = EXPERIMENTS_DIR / "jobs"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"
EVALUATION_DIR = EXPERIMENTS_DIR / "evaluation"
OUTPUTS_DIR = EXPERIMENTS_DIR / "outputs"
LOGS_DIR = EXPERIMENTS_DIR / "logs"

# Where the NSynth splits (training/ validation/ test/) live. Set NSYNTH_DIR in
# .env to keep the dataset outside the repo — on scratch, or an external drive.
NSYNTH_DIR = Path(os.environ.get("NSYNTH_DIR") or DATA_DIR / "nsynth").expanduser()
