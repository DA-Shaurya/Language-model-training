import unittest
from src.ocr.metrics.evaluator import compute_cer, compute_word_accuracy, evaluate_metrics


class TestMetrics(unittest.TestCase):
    def test_cer_calculation(self):
        preds = ["मराठी", "देवनागरी"]
        targets = ["मराठी", "देवनागर"]  # 1 edit distance
        cer = compute_cer(preds, targets)
        self.assertGreater(cer, 0.0)
        self.assertEqual(compute_cer(["मराठी"], ["मराठी"]), 0.0)

    def test_word_accuracy(self):
        preds = ["मराठी", "भारत", "गणेश"]
        targets = ["मराठी", "भारत", "शिव"]
        acc = compute_word_accuracy(preds, targets)
        self.assertAlmostEqual(acc, 2 / 3, places=2)

    def test_evaluate_metrics(self):
        preds = ["मराठी", "भारत"]
        targets = ["मराठी", "भारत"]
        m = evaluate_metrics(preds, targets)
        self.assertEqual(m["cer"], 0.0)
        self.assertEqual(m["char_accuracy"], 1.0)
        self.assertEqual(m["word_accuracy"], 1.0)
        self.assertEqual(m["wer"], 0.0)


if __name__ == "__main__":
    unittest.main()
