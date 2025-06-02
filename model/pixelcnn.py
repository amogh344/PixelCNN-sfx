import torch
import torch.nn as nn

class PixelCNN1D(nn.Module):
    def __init__(self, channels=256, hidden_dim=64, kernel_size=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            *[nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2), nn.ReLU()) for _ in range(7)],
            nn.Conv1d(hidden_dim, channels, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1).float() / 255.0  # Shape: [B, 1, T]
        return self.net(x).transpose(1, 2)  # Shape: [B, T, C]

