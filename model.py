import torch
import torch.nn as nn
import torchvision.models as models


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention applied over the feature-map
    width dimension so the LSTM can focus on the most informative time steps.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (T, B, C)
        x_t = x.permute(1, 2, 0)          # (B, C, T)
        attn = self.avg_pool(x_t).squeeze(-1)  # (B, C)
        attn = self.fc(attn).unsqueeze(0)  # (1, B, C)
        return x * attn                    # broadcast over T


class OCRModel(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Remove avgpool and fc — keep spatial feature maps
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

        # Channel attention between CNN and RNN
        self.attention = ChannelAttention(512)

        # Dropout before RNN to regularise
        self.dropout = nn.Dropout(p=dropout)

        # KEEP THIS FIXED (as per original comment)
        self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2,
                           dropout=dropout)

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)        # (B, 512, H, W)
        x = x.mean(2)          # (B, 512, W)  — collapse height
        x = x.permute(2, 0, 1) # (W, B, 512)  — time-first for LSTM

        x = self.attention(x)  # (W, B, 512)
        x = self.dropout(x)

        x, _ = self.rnn(x)     # (W, B, 512)

        x = self.fc(x)         # (W, B, num_classes)

        return x