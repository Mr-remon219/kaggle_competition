from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "spaceship_titanic" / "raw"
MODEL_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "spaceship_titanic" / "parameter"
)
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "spaceship_titanic"
