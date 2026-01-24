import yaml
import torch
from monai.networks.nets import UNet


def build_model(cfg):
    m = cfg["model"]
    model = UNet(
        spatial_dims=2,
        in_channels=m["in_channels"],
        out_channels=m["out_channels"],
        channels=m["channels"],
        strides=m["strides"],
        dropout=m["dropout_prob"],
    )
    return model


if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg)

    x = torch.randn(2, 6, 240, 240)
    out = model(x)
    print(out.shape)