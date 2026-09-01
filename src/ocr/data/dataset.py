import os
from typing import List, Optional, Tuple, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset


class IIITIndicHWDataset(Dataset):
    """
    PyTorch Dataset for IIIT-Indic-HW-UC Handwritten Word Corpus (Devanagari / Marathi).

    Args:
        image_paths: List of absolute or relative file paths to target word images.
        labels: List of ground-truth character strings corresponding to image_paths.
        transform: PyTorch torchvision transform pipeline.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[str],
        transform: Optional[Callable] = None,
    ):
        assert len(image_paths) == len(labels), (
            f"Mismatched image paths ({len(image_paths)}) and labels ({len(labels)})"
        )
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        # Handle unreadable/corrupt files gracefully by cycling through the dataset
        for attempt in range(len(self)):
            sample_idx = (idx + attempt) % len(self)
            img_path = self.image_paths[sample_idx]
            label = self.labels[sample_idx]

            if not label or label.strip() == "":
                continue

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            if self.transform:
                image = self.transform(image)

            return image, label

        raise RuntimeError("No readable or valid samples found in the dataset.")


# Alias for backward compatibility
MarathiDataset = IIITIndicHWDataset
