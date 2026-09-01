"""
Utility functions wrapper for Devanagari OCR.
Provides load_data, create_vocab, SimpleConverter, decode, and compute_cer.
"""

from typing import List, Tuple
from src.ocr.utils.vocab import DevanagariVocab, SimpleConverter, create_vocab
from src.ocr.metrics.evaluator import compute_cer, compute_word_accuracy, evaluate_metrics


def load_data(img_file: str, label_file: str) -> Tuple[List[str], List[str]]:
    """
    Helper function to load image file paths and ground-truth text labels.
    """
    with open(img_file, "r", encoding="utf-8") as f:
        images = f.read().splitlines()

    with open(label_file, "r", encoding="utf-8") as f:
        labels = f.read().splitlines()

    return images, labels


def decode(preds, converter: SimpleConverter) -> List[str]:
    """
    CTC greedy decoding wrapper.
    """
    return converter.decode_greedy(preds)


__all__ = [
    "load_data",
    "create_vocab",
    "SimpleConverter",
    "DevanagariVocab",
    "decode",
    "compute_cer",
    "compute_word_accuracy",
    "evaluate_metrics",
]