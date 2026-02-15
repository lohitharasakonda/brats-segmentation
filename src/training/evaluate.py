import os
import yaml
import torch
from pathlib import Path
from dotenv import load_dotenv

from src.data.splits import make_splits
from src.data.dataloader import get_dataloaders
from src.models.unet import build_model
from src.training.metrics import compute_dice

if __name__ == "__main__":
    load_dotenv()

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    splits = make_splits()
    _, _, test_loader = get_dataloaders(splits, cfg)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = build_model(cfg).to(device)
    ckpt_path = Path(os.environ.get("CHECKPOINT_DIR", "./checkpoints")) / "best_model.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    model.eval()
    test_dice = 0.0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            test_dice += compute_dice(logits, labels)

    test_dice /= len(test_loader)
    print(f"test dice: {test_dice:.4f}")