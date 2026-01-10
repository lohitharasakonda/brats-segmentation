import os
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from dotenv import load_dotenv
from scipy.ndimage import zoom
from tqdm import tqdm

load_dotenv()


def preprocess_patient(patient_dir, modalities, cache_dir):
    patient_id = Path(patient_dir).name
    images_path = cache_dir / f"{patient_id}_images.npy"
    seg_path = cache_dir / f"{patient_id}_seg.npy"

    if images_path.exists() and seg_path.exists():
        return

    vols = {}
    for mod in modalities:
        p = Path(patient_dir) / f"{patient_id}_{mod}.nii"
        if not p.exists():
            p = Path(patient_dir) / f"{patient_id}_{mod}.nii.gz"
        vols[mod] = nib.load(p).get_fdata()

    seg_p = Path(patient_dir) / f"{patient_id}_seg.nii"
    if not seg_p.exists():
        seg_p = Path(patient_dir) / f"{patient_id}_seg.nii.gz"
    seg_vol = nib.load(seg_p).get_fdata()

    # normalize
    for mod in modalities:
        vol = vols[mod]
        mask = vol > 0
        vols[mod] = (vol - vol[mask].mean()) / (vol[mask].std() + 1e-8)

    # resize to 128x128
    h, w, _ = seg_vol.shape
    for mod in modalities:
        vols[mod] = zoom(vols[mod], (128/h, 128/w, 1.0), order=1)
    seg_vol = zoom(seg_vol, (128/h, 128/w, 1.0), order=0)

    n = seg_vol.shape[2]
    images_arr = np.zeros((n - 2, len(modalities) * 3, 128, 128), dtype=np.float16)
    seg_arr = np.zeros((n - 2, 128, 128), dtype=np.uint8)

    for i, s in enumerate(range(1, n - 1)):
        ch = 0
        for mod in modalities:
            for offset in (-1, 0, 1):
                images_arr[i, ch] = vols[mod][:, :, s + offset]
                ch += 1
        seg_arr[i] = seg_vol[:, :, s]

    np.save(images_path, images_arr)
    np.save(seg_path, seg_arr)


if __name__ == "__main__":
    from src.data.splits import make_splits

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    modalities = cfg["data"]["modalities"]
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    splits = make_splits()
    all_patients = splits["train"] + splits["val"] + splits["test"]

    print(f"preprocessing {len(all_patients)} patients...")

    for patient_dir in tqdm(all_patients):
        try:
            preprocess_patient(patient_dir, modalities, cache_dir)
        except Exception as e:
            print(f"skipped {patient_dir}: {e}")

    print("done")