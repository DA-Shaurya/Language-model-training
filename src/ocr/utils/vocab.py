from typing import List, Tuple, Dict
import torch


class DevanagariVocab:
    """
    Vocabulary & Label Converter for Devanagari / Indic Handwritten Character Recognition.
    Maps characters to continuous indices (1..N) and reserves index 0 for the CTC blank token.
    """

    def __init__(self, vocab_chars: List[str]):
        # Reserve index 0 for CTC blank
        self.vocab = sorted(list(set(vocab_chars)))
        self.char2idx: Dict[str, int] = {c: i + 1 for i, c in enumerate(self.vocab)}
        self.idx2char: Dict[int, str] = {i + 1: c for i, c in enumerate(self.vocab)}
        self.blank_idx = 0

    def __len__(self) -> int:
        # Total classes = vocabulary length + 1 (blank token)
        return len(self.vocab) + 1

    def encode_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes a list of string targets into a 1D target tensor and a 1D length tensor for CTCLoss.
        """
        targets = []
        lengths = []

        for text in texts:
            encoded = [self.char2idx[c] for c in text if c in self.char2idx]
            targets.extend(encoded)
            lengths.append(len(encoded))

        target_tensor = torch.tensor(targets, dtype=torch.long)
        length_tensor = torch.tensor(lengths, dtype=torch.long)

        return target_tensor, length_tensor

    def decode_greedy(self, preds: torch.Tensor) -> List[str]:
        """
        Performs CTC greedy decoding on output logits or softmax probabilities.
        Args:
            preds: Model predictions of shape (T, B, num_classes) or (B, T, num_classes)
        Returns:
            Decoded text strings for each sample in the batch.
        """
        if preds.ndim == 3 and preds.size(1) != preds.size(0):
            # Ensure shape is (B, T, num_classes)
            if preds.size(0) > preds.size(1):  # (T, B, C)
                preds = preds.permute(1, 0, 2)

        argmax_preds = preds.argmax(dim=-1)  # (B, T)
        results = []

        for seq in argmax_preds:
            prev = -1
            text_chars = []
            for token_idx in seq:
                idx = token_idx.item()
                if idx != prev and idx != self.blank_idx:
                    if idx in self.idx2char:
                        text_chars.append(self.idx2char[idx])
                prev = idx
            results.append("".join(text_chars))

        return results


def create_vocab(labels: List[str]) -> List[str]:
    """
    Extracts sorted unique character vocabulary from a list of text labels.
    """
    chars = set()
    for word in labels:
        for c in word:
            chars.add(c)
    return sorted(list(chars))


# Alias for backward compatibility
SimpleConverter = DevanagariVocab
