"""
Model definition wrapper for Devanagari OCR.
Imports CRNN and ChannelAttention from src.ocr.models.crnn.
"""

from src.ocr.models.crnn import ChannelAttention, CRNN, OCRModel

__all__ = ["ChannelAttention", "CRNN", "OCRModel"]