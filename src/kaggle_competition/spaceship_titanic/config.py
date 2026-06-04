import random
from pathlib import Path

import numpy as np
import torch


DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "spaceship_titanic" / "raw"
MODEL_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "spaceship_titanic" / "parameter"
)
DATA_CONFIG_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "spaceship_titanic" / "data_config"
)
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "spaceship_titanic"


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
