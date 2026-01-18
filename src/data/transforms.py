from monai.transforms import Compose, RandFlipd, RandRotate90d, ToTensord


def get_train_transforms():
    return Compose([
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        ToTensord(keys=["image", "label"]),
    ])


def get_val_transforms():
    return Compose([
        ToTensord(keys=["image", "label"]),
    ])