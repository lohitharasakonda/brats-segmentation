import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.models.unet import build_model
from src.inference.predict import predict
from src.inference.uncertainty import mc_dropout_predict

CACHE_DIR = Path(os.environ.get("PREPROCESSED_DIR", "data/cache"))
CKPT_PATH = Path(os.environ.get("CHECKPOINT_DIR", "./checkpoints")) / "best_model.pth"


@st.cache_resource
def load_model():
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    return model, device, cfg


def load_cache(patient_id):
    images_path = CACHE_DIR / f"{patient_id}_images.npy"
    seg_path = CACHE_DIR / f"{patient_id}_seg.npy"
    if not images_path.exists() or not seg_path.exists():
        return None, None
    images = np.load(images_path, mmap_mode="r")
    seg = np.load(seg_path, mmap_mode="r")
    return images, seg


def compute_patient_dice(images, seg, model, device):
    eps = 1e-6
    dice_total = 0.0
    n_batches = 0
    batch_size = 32
    for start in range(0, len(images), batch_size):
        chunk = torch.from_numpy(
            images[start:start + batch_size].astype(np.float32)
        ).to(device)
        labels = torch.from_numpy(
            (seg[start:start + batch_size] > 0).astype(np.float32)
        ).unsqueeze(1).to(device)
        mask, _ = predict(model, chunk)
        pred_t = (mask == 1).float()
        true_t = labels.squeeze(1)
        dice = (2.0 * (pred_t * true_t).sum() + eps) / (pred_t.sum() + true_t.sum() + eps)
        dice_total += dice.item()
        n_batches += 1
    return dice_total / n_batches


def run_uncertainty(images, model, device, n_passes=20):
    all_entropy = []
    batch_size = 32
    for start in range(0, len(images), batch_size):
        chunk = torch.from_numpy(
            images[start:start + batch_size].astype(np.float32)
        ).to(device)
        _, _, ent = mc_dropout_predict(model, chunk, n_passes=n_passes)
        all_entropy.append(ent.cpu().numpy())
    return np.concatenate(all_entropy, axis=0)


def middle_tumor_slice(seg):
    tumor = [i for i in range(len(seg)) if seg[i].any()]
    return tumor[len(tumor) // 2] if tumor else len(seg) // 2


# ── app ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="BraTS Segmentation", layout="wide")
st.markdown("""
<style>
    .stApp {
        background-color: #fde8ef;
    }
    .stApp, .stMarkdown, p, label, h1, h2, h3 {
        color: #3a2030 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #f5d0de;
    }
    /* Run button */
    .stButton > button {
        background-color: #c2607a;
        color: white;
        border: none;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #a04060;
        color: white;
    }
    /* Top header bar */
    header[data-testid="stHeader"] {
        background-color: #f5d0de;
     }
    header[data-testid="stHeader"] * {
        color: #3a2030 !important;
    }
    [data-testid="stSidebarCollapseButton"] button {
    background-color: #c2607a !important;
    border-radius: 50% !important;
}
[data-testid="stSidebarCollapseButton"] span {
    color: white !important;
}
    
</style>
""", unsafe_allow_html=True)
st.title("🧠 Brain Tumor Segmentation")
st.caption("MONAI UNet · BraTS 2020 · Test Dice 0.8651")

model, device, cfg = load_model()

# get all available patient ids from cache
patient_ids = sorted(set(
    p.stem.replace("_images", "").replace("_seg", "").replace("_entropy", "")
    for p in CACHE_DIR.glob("*.npy")
))

patient_id = st.sidebar.selectbox("Select Patient", patient_ids)
run = st.sidebar.button("Run")

if run and patient_id:
    images, seg = load_cache(patient_id)
    if images is None:
        st.error("Cache not found for this patient.")
        st.stop()

    with st.spinner("Running inference..."):
        dice = compute_patient_dice(images, seg, model, device)

    with st.spinner("Running MC-Dropout uncertainty (20 passes)..."):
        entropy = run_uncertainty(images, model, device, n_passes=20)

    st.session_state.images = images
    st.session_state.seg = seg
    st.session_state.entropy = entropy
    st.session_state.dice = dice
    st.session_state.patient_id = patient_id
    st.session_state.default_slice = middle_tumor_slice(seg)
    st.rerun()

if "images" in st.session_state:
    images = st.session_state.images
    seg = st.session_state.seg
    entropy = st.session_state.entropy
    dice = st.session_state.dice
    pid = st.session_state.patient_id

    st.sidebar.markdown("---")
    s = st.sidebar.slider("Slice", 0, len(images) - 1, st.session_state.default_slice)
    show_mask = st.sidebar.checkbox("Tumor overlay", value=True)
    show_ent = st.sidebar.checkbox("Uncertainty overlay", value=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("Patient", pid)
    col2.metric("Dice Score", f"{dice:.4f}")
    col3.metric("Mean Entropy", f"{entropy.mean():.4f}")

    st.divider()

    flair = images[s, 1].astype(np.float32)
    mask = seg[s]
    ent_slice = entropy[s]

    n_panels = 1 + int(show_mask) + int(show_ent)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    idx = 0
    axes[idx].imshow(flair.T, cmap="gray", origin="lower")
    axes[idx].set_title("FLAIR")
    axes[idx].axis("off")

    if show_mask:
        idx += 1
        axes[idx].imshow(flair.T, cmap="gray", origin="lower")
        overlay = np.ma.masked_where(mask.T == 0, mask.T.astype(float))
        axes[idx].imshow(overlay, cmap="Reds", alpha=0.5, origin="lower")
        axes[idx].set_title("Tumor Overlay")
        axes[idx].axis("off")

    if show_ent:
        idx += 1
        axes[idx].imshow(flair.T, cmap="gray", origin="lower")
        axes[idx].imshow(ent_slice.T, cmap="plasma", alpha=0.6, origin="lower")
        axes[idx].set_title("Uncertainty")
        axes[idx].axis("off")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)