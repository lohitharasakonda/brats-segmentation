import torch


def compute_dice(logits, labels, eps=1e-6):
    preds = torch.argmax(logits, dim=1)
    targets = labels.squeeze(1).long()

    pred_tumor = (preds == 1).float()
    true_tumor = (targets == 1).float()

    intersection = (pred_tumor * true_tumor).sum()
    dice = (2.0 * intersection + eps) / (pred_tumor.sum() + true_tumor.sum() + eps)

    return dice.item()


if __name__ == "__main__":
    import torch

    pred = torch.zeros(2, 2, 10, 10)
    pred[:, 1, 3:7, 3:7] = 10.0
    label = torch.zeros(2, 1, 10, 10)
    label[:, :, 3:7, 3:7] = 1.0

    print(compute_dice(pred, label))