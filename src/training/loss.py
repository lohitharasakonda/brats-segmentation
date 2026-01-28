from monai.losses import DiceCELoss


def build_loss():
    return DiceCELoss(to_onehot_y=True, softmax=True)


if __name__ == "__main__":
    import torch

    loss_fn = build_loss()
    pred = torch.randn(2, 2, 240, 240)
    label = torch.randint(0, 2, (2, 1, 240, 240)).float()
    loss = loss_fn(pred, label)
    print(loss.item())