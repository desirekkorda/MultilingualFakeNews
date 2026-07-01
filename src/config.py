
# ==========================================
# MODEL CONFIGURATION
# ==========================================

MODEL_NAME = "xlm-roberta-base"

MAX_LEN = 384

NUM_CLASSES = 2

# ==========================================
# TRAINING CONFIGURATION
# ==========================================

BATCH_SIZE = 8

LEARNING_RATE = 2e-5

DROPOUT = 0.3

RANDOM_STATE = 42

# ==========================================
# LABELS
# ==========================================

LABEL_MAP = {
    "Legit": 0,
    "Fake": 1
}

ID2LABEL = {
    0: "Legit",
    1: "Fake"
}

# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = "/content/drive/MyDrive/MultilingualFakeNews"

DATA_DIR = f"{PROJECT_ROOT}/data"

MODEL_DIR = f"{PROJECT_ROOT}/models"

DATASET_ROOT = f"{DATA_DIR}/TALLIP-FakeNews-Dataset"

ENGLISH_DIR = f"{DATASET_ROOT}/English"

ENGLISH_TRAIN_PATH = (
    f"{ENGLISH_DIR}/Train/"
    "train_English_Data_Complete_FakeNews.txt"
)

ENGLISH_TEST_PATH = (
    f"{ENGLISH_DIR}/Test/"
    "test_English_Data_Complete_FakeNews.txt"
)
