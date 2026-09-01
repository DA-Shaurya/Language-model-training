from typing import List, Dict, Any
import Levenshtein


def compute_cer(predictions: List[str], targets: List[str]) -> float:
    """
    Computes Character Error Rate (CER) using normalized Levenshtein distance.
    CER = Sum(EditDistance(pred, target)) / Sum(Len(target))
    """
    total_dist = 0
    total_chars = 0

    for p, g in zip(predictions, targets):
        total_dist += Levenshtein.distance(p, g)
        total_chars += len(g)

    return total_dist / total_chars if total_chars > 0 else 0.0


def compute_word_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Computes exact-match Word Accuracy percentage.
    Word Accuracy = Count(pred == target) / Total Samples
    """
    if not targets:
        return 0.0

    exact_matches = sum(1 for p, g in zip(predictions, targets) if p.strip() == g.strip())
    return exact_matches / len(targets)


def evaluate_metrics(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """
    Computes comprehensive evaluation metrics dictionary.
    Returns:
        - cer: Character Error Rate (e.g. 0.0464)
        - char_accuracy: Character Accuracy (1 - CER, e.g. 0.9536 / 95.4%)
        - word_accuracy: Word Accuracy percentage (e.g. 0.8390 / 83.9%)
        - wer: Word Error Rate (1 - word_accuracy)
    """
    cer = compute_cer(predictions, targets)
    word_acc = compute_word_accuracy(predictions, targets)

    return {
        "cer": cer,
        "char_accuracy": max(0.0, 1.0 - cer),
        "word_accuracy": word_acc,
        "wer": max(0.0, 1.0 - word_acc),
    }
