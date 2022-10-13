from pathlib import Path

DATA_DIR = Path.cwd().parents[0] / "data"
DICOMS_DIR = DATA_DIR / "dicoms"
LABELS_PATH = DATA_DIR / "ground_truths.csv"
PREPROC_DIR = DATA_DIR / "preproc"

ID_COLUMN = "patientId"
TARGET_COLUMN = "Target"

IMG_SIZE = (224, 224)

SPLIT_SIZE = {"train": .8, "dev": .1, "test": .1}

SEED = 1234
