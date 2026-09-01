"""
Dataset definition wrapper for Devanagari OCR.
Imports IIITIndicHWDataset and MarathiDataset from src.ocr.data.dataset.
"""

from src.ocr.data.dataset import IIITIndicHWDataset, MarathiDataset

__all__ = ["IIITIndicHWDataset", "MarathiDataset"]