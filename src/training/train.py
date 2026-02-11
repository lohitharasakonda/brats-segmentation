import argparse
import yaml
import torch
from dotenv import load_dotenv

from src.data.splits import make_splits
from src.data.dataloader import get_dataloaders
from src.models.unet import build_model
from src.training.loss import build_loss
from src.training.trainer import train


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    splits = make_splits()

    if args.debug:
        n_patients = cfg["data"]["num_patients_debug"]
        splits = {
            "train": splits["train"][:n_patients],
            "val":   splits["val"][:2],
            "test":  splits["test"][:2],
        }
        epochs = cfg["training"]["epochs_debug"]
        print(f"debug mode: {n_patients} patients, {epochs} epochs")
    else:
        epochs = cfg["training"]["epochs_full"]
        print(f"full training: {len(splits['train'])} patients, {epochs} epochs")

    cfg["training"]["epochs_debug"] = epochs

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"using {device}")

    train_loader, val_loader, _ = get_dataloaders(splits, cfg)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = build_loss()

    train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, cfg, device)