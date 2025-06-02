import torch
import torch.nn as nn

class PixelCNN1D(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64, kernel_size=7, layers=10):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.conv_stack = nn.ModuleList()
        for i in range(layers):
            self.conv_stack.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2))
        self.out = nn.Conv1d(hidden_dim, input_dim, 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (B, H, T)
        for conv in self.conv_stack:
            x = torch.relu(conv(x))
        return self.out(x).permute(0, 2, 1)  # (B, T, C)
