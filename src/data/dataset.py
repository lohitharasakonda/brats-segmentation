import os
import random
from pathlib import Path

import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


class BraTSSliceDataset(Dataset):

    def __init__(self, patient_dirs, modalities, transform=None, empty_ratio=2.0, seed=42):
        self.modalities = modalities
        self.transform = transform
        self.cache_dir = Path(os.environ.get("PREPROCESSED_DIR", "data/cache"))
        self.samples = []

        rng = random.Random(seed)

        for patient_dir in patient_dirs:
            patient_id = Path(patient_dir).name
            seg_path = self.cache_dir / f"{patient_id}_seg.npy"

            if seg_path.exists():
                seg = np.load(seg_path, mmap_mode="r")
                tumor_slices = [(patient_dir, i) for i in range(len(seg)) if seg[i].any()]
                empty_slices = [(patient_dir, i) for i in range(len(seg)) if not seg[i].any()]
            else:
                # fallback to nifti
                try:
                    seg_vol = nib.load(Path(patient_dir) / f"{patient_id}_seg.nii").get_fdata()
                except FileNotFoundError:
                    print(f"[Dataset] SKIP {patient_id}: seg file not found")
                    continue
                n = seg_vol.shape[2]
                tumor_slices, empty_slices = [], []
                for s in range(1, n - 1):
                    if seg_vol[:, :, s].any():
                        tumor_slices.append((patient_dir, s - 1))
                    else:
                        empty_slices.append((patient_dir, s - 1))

            # dont want too many empty slices
            max_empty = int(len(tumor_slices) * empty_ratio)
            empty_slices = empty_slices[:max_empty]  # lazy, should probably sample but whatever

            self.samples += tumor_slices + empty_slices

        random.shuffle(self.samples)
        print(f"total slices: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient_dir, i = self.samples[idx]
        patient_id = Path(patient_dir).name

        seg_path = self.cache_dir / f"{patient_id}_seg.npy"

        if seg_path.exists():
            images = np.load(self.cache_dir / f"{patient_id}_images.npy", mmap_mode="r")
            seg = np.load(seg_path, mmap_mode="r")
            image = images[i].astype(np.float32)
            label = (seg[i] > 0).astype(np.float32)[np.newaxis]
        else:
            s = i + 1
            channels = []
            for mod in self.modalities:
                vol = nib.load(Path(patient_dir) / f"{patient_id}_{mod}.nii").get_fdata()
                for offset in (-1, 0, 1):
                    channels.append(vol[:, :, s + offset].astype(np.float32))
            image = np.stack(channels, axis=0)
            seg_vol = nib.load(Path(patient_dir) / f"{patient_id}_seg.nii").get_fdata()
            label = (seg_vol[:, :, s] > 0).astype(np.float32)[np.newaxis]

        sample = {"image": image, "label": label}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    from src.data.splits import make_splits

    splits = make_splits()
    ds = BraTSSliceDataset(splits["train"][:2], modalities=["flair", "t1ce"])
    s = ds[0]
    print(s["image"].shape, s["label"].shape)