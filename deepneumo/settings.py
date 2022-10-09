from pathlib import Path

DATA_DIR = Path.cwd().parents[0] / "data"
DICOMS_DIR = DATA_DIR / "dicoms"
LABELS_PATH = DATA_DIR / "ground_truths.csv"
PREPROC_DIR = DATA_DIR / "preproc"

ID_COLUMN = "patientId"
TARGET_COLUMN = "Target"

IMG_SIZE = (224, 224)

SPLIT_SIZE = (.8, .1, .1)
SUBSET_NAMES = ("train", "dev", "test")

SEED = 1234
