import json
import os
import random
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


def make_splits(config_path="configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["data"]["seed"]
    train_ratio = cfg["data"]["train_ratio"]
    val_ratio = cfg["data"]["val_ratio"]

    splits_file = Path(os.environ["SPLITS_DIR"]) / "splits.json"

    if splits_file.exists():
        with open(splits_file) as f:
            return json.load(f)

    data_root = Path(os.environ["DATA_ROOT"])
    patients = sorted([str(p) for p in data_root.iterdir() if p.is_dir()])

    random.seed(seed)
    random.shuffle(patients)

    n = len(patients)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": patients[:n_train],
        "val": patients[n_train:n_train + n_val],
        "test": patients[n_train + n_val:]
    }

    splits_file.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_file, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"saved splits to {splits_file}")
    return splits


if __name__ == "__main__":
    splits = make_splits()
    print(len(splits["train"]), len(splits["val"]), len(splits["test"]))