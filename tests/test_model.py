import unittest
import torch
from src.ocr.models.crnn import CRNN, ChannelAttention


class TestModel(unittest.TestCase):
    def test_channel_attention(self):
        attn = ChannelAttention(channels=512, reduction=16)
        x = torch.randn(20, 4, 512)  # (T=20, B=4, C=512)
        out = attn(x)
        self.assertEqual(out.shape, (20, 4, 512))

    def test_crnn_forward_pass(self):
        model = CRNN(num_classes=118, use_attention=True)
        images = torch.randn(2, 3, 32, 640)  # Batch=2, Height=32, Width=640
        logits = model(images)
        self.assertEqual(logits.ndim, 3)
        self.assertEqual(logits.size(1), 2)  # Batch dimension
        self.assertEqual(logits.size(2), 118)  # Classes dimension

    def test_crnn_ablation_toggle(self):
        model_no_attn = CRNN(num_classes=118, use_attention=False)
        images = torch.randn(1, 3, 32, 640)
        logits = model_no_attn(images)
        self.assertEqual(logits.ndim, 3)
        self.assertEqual(logits.size(2), 118)


if __name__ == "__main__":
    unittest.main()
