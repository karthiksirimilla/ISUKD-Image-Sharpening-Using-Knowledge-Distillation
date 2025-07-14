import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoSharpeningStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 8, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 64, 1),
            nn.Sigmoid()
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 3, 3, padding=1)
        )
        
        self.skip = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        residual = self.skip(x)
        x = self.encoder(x)
        x = x * self.attention(x)
        x = self.decoder(x)
        return torch.sigmoid(x + residual[:, :x.shape[1], :, :])