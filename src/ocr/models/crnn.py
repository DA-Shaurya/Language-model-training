import torch
import torch.nn as nn
import torchvision.models as models


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation (SE) channel attention module applied over the feature-map
    temporal/width dimension, enabling the downstream BiLSTM to focus on salient feature steps.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x input shape: (T, B, C) where T = sequence width, B = batch, C = channels (512)
        x_t = x.permute(1, 2, 0)  # (B, C, T)
        attn = self.avg_pool(x_t).squeeze(-1)  # (B, C)
        attn = self.fc(attn).unsqueeze(0)  # (1, B, C)
        return x * attn  # Broadcast over time dimension T


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network (CRNN) for Devanagari OCR.

    Architecture:
    - Backbone: ResNet-34 pre-trained feature extractor (conv1 through layer4)
    - Attention: Optional Squeeze-and-Excitation (SE) Channel Attention
    - Recurrent: 2-layer Bidirectional LSTM (BiLSTM)
    - Classifier: Fully connected linear projection to vocabulary size (+ CTC blank)
    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        num_rnn_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention

        # ResNet-34 Feature Extractor (removes avgpool & fc)
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

        # Squeeze-and-Excitation Channel Attention
        if self.use_attention:
            self.attention = ChannelAttention(512, reduction=16)

        # Regularization
        self.dropout = nn.Dropout(p=dropout)

        # 2-layer Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            bidirectional=True,
            num_layers=num_rnn_layers,
            dropout=dropout if num_rnn_layers > 1 else 0.0,
        )

        # Fully-connected linear classifier head
        # 2 * hidden_size because BiLSTM is bidirectional
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input images tensor of shape (B, 3, H, W)
        Returns:
            Output logit tensor of shape (T, B, num_classes) for CTCLoss
        """
        features = self.cnn(x)  # (B, 512, H_feat, W_feat)
        features = features.mean(2)  # (B, 512, W_feat) - collapse height dimension
        features = features.permute(2, 0, 1)  # (W_feat, B, 512) - time-first for PyTorch LSTM

        if self.use_attention:
            features = self.attention(features)

        features = self.dropout(features)
        recurrent_out, _ = self.rnn(features)  # (W_feat, B, 512)
        logits = self.fc(recurrent_out)  # (W_feat, B, num_classes)

        return logits


# Alias for backward compatibility
OCRModel = CRNN
