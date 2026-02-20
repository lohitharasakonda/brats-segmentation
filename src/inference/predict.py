import torch
import torch.nn.functional as F


def predict(model, image):
    model.eval()
    with torch.no_grad():
        logits = model(image)
        probs = F.softmax(logits, dim=1)
        mask = torch.argmax(probs, dim=1)
    return mask, probs