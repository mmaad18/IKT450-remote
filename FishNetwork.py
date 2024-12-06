import torch
from torch import nn


class FishNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network_stack = nn.Sequential(
            # 64x64x3
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 32x32x6
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 16x16x12
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 8x8x24
            nn.Flatten(),
            nn.Linear(8*8*24, 4*4*24),
            nn.ReLU(),
            nn.Linear(4*4*24, 23),
        )

        self._initialize_weights()


    def forward(self, x):
        return self.network_stack(x)


    def _initialize_weights(self):
        for layer in self.network_stack:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)


