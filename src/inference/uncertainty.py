import torch
import torch.nn.functional as F


def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_dropout_predict(model, image, n_passes=20):
    model.eval()
    enable_dropout(model)

    all_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(image)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)

    stacked = torch.stack(all_probs, dim=0)
    mean_probs = stacked.mean(dim=0)
    mean_mask = torch.argmax(mean_probs, dim=1)

    eps = 1e-8
    entropy_map = -(mean_probs * torch.log(mean_probs + eps)).sum(dim=1)

    return mean_mask, mean_probs, entropy_map