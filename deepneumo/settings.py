from pathlib import Path

#DATA_DIR = Path.cwd().parents[0] / "data"
DATA_DIR = Path.cwd() / "data"
DICOMS_DIR = DATA_DIR / "dicoms"
LABELS_PATH = DATA_DIR / "ground_truths.csv"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"


ID_COLUMN = "patientId"
TARGET_COLUMN = "Target"
CONTROL_CLASS = "control"
CONDITION_CLASS = "pneumonia"

IMG_SIZE = (224, 224)

SPLIT_SIZE = {"train": .8, "val": .1, "test": .1}

SEED = 1234
