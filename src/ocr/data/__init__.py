from src.ocr.data.dataset import IIITIndicHWDataset, MarathiDataset
from src.ocr.data.transforms import get_train_transforms, get_val_transforms

__all__ = [
    "IIITIndicHWDataset",
    "MarathiDataset",
    "get_train_transforms",
    "get_val_transforms",
]
