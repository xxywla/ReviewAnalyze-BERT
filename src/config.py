from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

PRETRAINED_MODEL_NAME = Path(__file__).parent.parent / "pretrained" / "bert-base-chinese"

MODEL_PATH = Path(__file__).parent.parent / "model"
MODEL_FILE_NAME = MODEL_PATH / "model.pt"

LOGS_DIR = Path(__file__).parent.parent / "logs"

SEQ_LEN = 128

BATCH_SIZE = 16
LEARNING_RATE = 1e-5
EPOCHS = 5

if __name__ == '__main__':
    print(DATA_DIR)
