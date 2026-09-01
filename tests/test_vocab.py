import unittest
import torch
from src.ocr.utils.vocab import DevanagariVocab, create_vocab


class TestVocab(unittest.TestCase):
    def test_vocab_creation(self):
        labels = ["मराठी", "देवनागरी", "भारत"]
        vocab_list = create_vocab(labels)
        self.assertIn("म", vocab_list)
        self.assertIn("र", vocab_list)
        self.assertIn("ा", vocab_list)

    def test_vocab_encoding_decoding(self):
        vocab_chars = ["क", "ख", "ग", "घ"]
        vocab = DevanagariVocab(vocab_chars)

        texts = ["कख", "गघ"]
        targets, lengths = vocab.encode_batch(texts)

        self.assertEqual(len(lengths), 2)
        self.assertEqual(lengths[0].item(), 2)
        self.assertEqual(lengths[1].item(), 2)
        self.assertEqual(len(targets), 4)

        # Test dummy logits tensor for decoding (T=5, B=2, C=5)
        logits = torch.zeros(5, 2, len(vocab))
        logits[0, 0, 1] = 10.0
        logits[1, 0, 1] = 10.0  # duplicate frame to test CTC collapse
        logits[2, 0, 2] = 10.0

        decoded = vocab.decode_greedy(logits)
        self.assertEqual(decoded[0], "कख")


if __name__ == "__main__":
    unittest.main()
