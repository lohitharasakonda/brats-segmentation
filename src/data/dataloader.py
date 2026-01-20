from torch.utils.data import DataLoader
from src.data.dataset import BraTSSliceDataset
from src.data.transforms import get_train_transforms, get_val_transforms

def get_dataloaders(splits, cfg):
    modalities = cfg["data"]["modalities"]
    seed = cfg["data"]["seed"]
    empty_ratio = cfg["data"]["empty_ratio"]
    batch_size = cfg["training"]["batch_size"]

    train_ds = BraTSSliceDataset(splits["train"], modalities, get_train_transforms(), empty_ratio=empty_ratio, seed=seed)
    val_ds = BraTSSliceDataset(splits["val"], modalities, get_val_transforms(), empty_ratio=empty_ratio, seed=seed)
    test_ds = BraTSSliceDataset(splits["test"], modalities, get_val_transforms(), empty_ratio=empty_ratio, seed=seed)

    # tried num_workers=4 but kept getting errors, 0 seems to work
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import yaml
    from src.data.splits import make_splits

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    splits = make_splits()

    small_splits = {
        "train": splits["train"][:3],
        "val": splits["val"][:1],
        "test": splits["test"][:1],
    }

    train_loader, val_loader, test_loader = get_dataloaders(small_splits, cfg)

    batch = next(iter(train_loader))
    print(batch["image"].shape)
    print(batch["label"].shape)
    print(len(train_loader), len(val_loader), len(test_loader))