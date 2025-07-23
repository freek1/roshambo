import math
import torch
import torch.nn as nn


class Simple1DCNN(nn.Module):
    def __init__(
        self,
        num_sensors=312,
        num_classes=3,
        dropout_prob=0.2,  # bumped up
        base_width=32,  # start smaller
    ):
        super().__init__()

        def conv_block(in_ch, out_ch, k):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch) if torch.cuda.is_available() else nn.Identity(),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_block(num_sensors, base_width, 3),
            nn.MaxPool1d(2),
            nn.Dropout1d(dropout_prob),
            conv_block(base_width, base_width * 2, 5),
            nn.MaxPool1d(2),
            nn.Dropout1d(dropout_prob),
            conv_block(base_width * 2, base_width * 4, 7),
            nn.MaxPool1d(2),
            nn.Dropout1d(dropout_prob),
            conv_block(base_width * 4, base_width * 8, 9),
            # switch to avg-pool
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout1d(dropout_prob),
        )

        self.classifier = nn.Sequential(
            # more dropout before classifier
            nn.Conv1d(base_width * 8, base_width * 4, 1),
            (
                nn.BatchNorm1d(base_width * 4)
                if torch.cuda.is_available()
                else nn.Identity()
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv1d(base_width * 4, num_classes, 1),
        )

    def forward(self, x):
        # x: (B, num_sensors, T)
        if x.size(0) == 1:  # If batch size is 1
            # Temporarily disable batch norm
            for module in self.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.eval()
        else:
            # Enable batch norm for batch size > 1
            for module in self.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.train()

        x = self.features(x)  # → (B, base_width*8, 1)
        x = self.classifier(x)  # → (B, num_classes, 1)
        return x.squeeze(-1)  # → (B, num_classes)

    def _conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
            (
                nn.BatchNorm1d(out_channels)
                if torch.cuda.is_available()
                else nn.Identity()
            ),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding="same"),
            (
                nn.BatchNorm1d(out_channels)
                if torch.cuda.is_available()
                else nn.Identity()
            ),
            nn.ReLU(),
        )
