import os
import torch
from pathlib import Path


def train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, cfg, device):
    from src.training.metrics import compute_dice

    epochs = cfg["training"]["epochs_debug"]
    checkpoint_dir = Path(os.environ.get("CHECKPOINT_DIR", "./checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_dice = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # validation
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(images)
                val_dice += compute_dice(logits, labels)

        val_dice /= len(val_loader)
        scheduler.step()

        print(f"epoch {epoch}/{epochs} loss {train_loss:.4f} dice {val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
            print("saved checkpoint")

    print(f"done, best dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    import yaml
    from dotenv import load_dotenv
    load_dotenv()

    from src.data.splits import make_splits
    from src.data.dataloader import get_dataloaders
    from src.models.unet import build_model
    from src.training.loss import build_loss

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    splits = make_splits()
    small_splits = {
        "train": splits["train"][:2],
        "val": splits["val"][:1],
        "test": splits["test"][:1],
    }
    cfg["training"]["epochs_debug"] = 2

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"using device: {device}")

    train_loader, val_loader, _ = get_dataloaders(small_splits, cfg)
    model = build_model(cfg).to(device)
    loss_fn = build_loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs_debug"])

    train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, cfg, device)