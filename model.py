import torch.nn as nn
import torchvision.models as models

class OCRModel(nn.Module):
    def __init__(self, num_classes):
        super(OCRModel, self).__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Remove last layers
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

        # KEEP THIS FIXED
        self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2)

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)  # (B, 512, H, W)

        x = x.mean(2)   # (B, 512, W)

        x = x.permute(2, 0, 1)  # (W, B, 512)

        x, _ = self.rnn(x)

        x = self.fc(x)

        return x