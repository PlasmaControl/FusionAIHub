import torch.nn as nn


class SpectrogramProcessor(nn.Module):
    """2D CNN for processing spectrograms."""

    def __init__(self, in_channels, out_features=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, channels, freq_bins, time_frames)
        Returns:
            features: (batch, out_features)
        """
        x = self.conv(x)
        return self.fc(x)


class TimeSeriesProcessor(nn.Module):
    """1D CNN for processing time series."""

    def __init__(self, in_channels, out_features=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time_frames)
        Returns:
            features: (batch, out_features)
        """
        x = self.conv(x)
        return self.fc(x)


class VideoProcessor(nn.Module):
    """3D CNN for processing video data (handles grayscale)."""

    def __init__(self, in_channels=1, out_features=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),  # Don't pool temporal dimension too much
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, time, height, width) - grayscale video
        Returns:
            features: (batch, out_features)
        """
        # Add channel dimension: (batch, time, height, width) -> (batch, 1, time, height, width)
        x = x.unsqueeze(1)
        x = self.conv(x)
        return self.fc(x)
