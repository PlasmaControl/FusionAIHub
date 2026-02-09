import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class CrossModalAttention(nn.Module):
    """Cross-modal attention fusion layer."""

    def __init__(self, feature_dim, num_modalities, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features):
        """
        Args:
            features: List of tensors, each (batch, feature_dim)
        Returns:
            fused: Tensor (batch, feature_dim)
        """
        # Stack features: (batch, num_modalities, feature_dim)
        stacked = torch.stack(features, dim=1)

        # Self-attention across modalities
        attended, _ = self.attention(stacked, stacked, stacked)

        # Residual connection
        attended = self.norm(attended + stacked)

        # Aggregate (mean pooling across modalities)
        fused = attended.mean(dim=1)  # (batch, feature_dim)

        return fused

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


class MultiModalTokamakModel(nn.Module):
    """
    Multi-modal tokamak model with switchable fusion strategies.

    Processes 9 modalities independently:
    - 3 spectrograms (mhr, ece, co2)
    - 2 time series groups (actuators, diagnostics)
    - 3 videos (bolo, irtv, tangtv)
    - 1 text
    """

    def __init__(
        self,
        feature_dim=64,
        fusion_mode="attention",  # "concat" or "attention"
        text_model_name="distilbert-base-uncased",
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.fusion_mode = fusion_mode

        # ====== Spectrogram Processors ======
        self.mhr_processor = SpectrogramProcessor(
            in_channels=8, out_features=feature_dim
        )
        self.ece_processor = SpectrogramProcessor(
            in_channels=48, out_features=feature_dim
        )
        self.co2_processor = SpectrogramProcessor(
            in_channels=4, out_features=feature_dim
        )

        # ====== Time Series Processors ======
        # Actuators: gas(5) + ech(11) + pin(8) + tin(8) = 32 channels
        self.actuator_processor = TimeSeriesProcessor(
            in_channels=32, out_features=feature_dim
        )

        # Diagnostics: d_alpha(1) + mse(1) + ts_core_density(1) = 3 channels
        self.diagnostic_processor = TimeSeriesProcessor(
            in_channels=119, out_features=feature_dim
        )

        # ====== Video Processors ======
        self.bolo_processor = VideoProcessor(in_channels=1, out_features=feature_dim)
        self.irtv_processor = VideoProcessor(in_channels=1, out_features=feature_dim)
        self.tangtv_processor = VideoProcessor(in_channels=1, out_features=feature_dim)

        # ====== Text Processor ======
        print(f"Loading pre-trained text model: {text_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Project BERT embedding (768) to feature space
        self.text_projector = nn.Sequential(
            nn.Linear(768, feature_dim),
            nn.ReLU(),
        )

        # ====== Fusion ======
        self.num_modalities = 9

        if fusion_mode == "attention":
            self.fusion = CrossModalAttention(
                feature_dim=feature_dim, num_modalities=self.num_modalities, num_heads=4
            )
            final_input_dim = feature_dim
        else:  # concatenation
            final_input_dim = feature_dim * self.num_modalities

        # ====== Final Predictor ======
        self.predictor = nn.Sequential(
            nn.Linear(final_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def set_fusion_mode(self, mode):
        """Switch between fusion modes."""
        assert mode in ["concat", "attention"], "Mode must be 'concat' or 'attention'"
        self.fusion_mode = mode

    def forward(self, batch):
        """
        Args:
            batch: Dictionary with keys:
                - mhr: (batch, 8, freq_bins, time_frames)
                - ece: (batch, 48, freq_bins, time_frames)
                - co2: (batch, 4, freq_bins, time_frames)
                - gas: (batch, 5, 1, time_frames)
                - ech: (batch, 11, 1, time_frames)
                - pin: (batch, 8, 1, time_frames)
                - tin: (batch, 8, 1, time_frames)
                - d_alpha: (batch, 1, 1, time_frames)
                - mse: (batch, 1, 1, time_frames)
                - ts_core_density: (batch, 1, 1, time_frames)
                - bolo: (batch, 10, 3, 64, 64)
                - irtv: (batch, 10, 3, 64, 64)
                - tangtv: (batch, 10, 3, 64, 64)
                - text: List[str]
        Returns:
            output: (batch, 1)
        """
        features = []
        device = next(self.parameters()).device

        # ====== Process Spectrograms ======
        if "mhr" in batch:
            features.append(self.mhr_processor(batch["mhr"]))

        if "ece" in batch:
            features.append(self.ece_processor(batch["ece"]))

        if "co2" in batch:
            features.append(self.co2_processor(batch["co2"]))

        # ====== Process Time Series ======
        # Concatenate actuators: gas, ech, pin, tin
        actuators = torch.cat(
            [
                batch["gas"].squeeze(2),  # (batch, 5, time_frames)
                batch["ech"].squeeze(2),  # (batch, 11, time_frames)
                batch["pin"].squeeze(2),  # (batch, 8, time_frames)
                batch["tin"].squeeze(2),  # (batch, 8, time_frames)
            ],
            dim=1,
        )  # (batch, 32, time_frames)
        features.append(self.actuator_processor(actuators))

        # Concatenate diagnostics: d_alpha, mse, ts_core_density
        diagnostics = torch.cat(
            [
                batch["d_alpha"].squeeze(2),  # (batch, 1, time_frames)
                batch["mse"].squeeze(2),  # (batch, 1, time_frames)
                batch["ts_core_density"].squeeze(2),  # (batch, 1, time_frames)
            ],
            dim=1,
        )  # (batch, 3, time_frames)
        features.append(self.diagnostic_processor(diagnostics))

        # ====== Process Videos ======
        if "bolo" in batch:
            features.append(self.bolo_processor(batch["bolo"]))

        if "irtv" in batch:
            features.append(self.irtv_processor(batch["irtv"]))

        if "tangtv" in batch:
            features.append(self.tangtv_processor(batch["tangtv"]))

        # ====== Process Text ======
        if "text" in batch and batch["text"] is not None:
            # Prepare text batch
            text_batch = []
            for text_item in batch["text"]:
                if isinstance(text_item, list):
                    text_batch.append(" ".join([str(s) for s in text_item]))
                else:
                    text_batch.append(str(text_item))

            # Tokenize
            encoded = self.tokenizer(
                text_batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Encode
            with torch.no_grad():
                output = self.text_encoder(input_ids, attention_mask=attention_mask)

            # Project CLS token
            cls_token = output.last_hidden_state[:, 0, :]
            features.append(self.text_projector(cls_token))

        # ====== Fusion ======
        if self.fusion_mode == "attention":
            fused = self.fusion(features)
        else:  # concatenation
            fused = torch.cat(features, dim=1)

        # ====== Prediction ======
        output = self.predictor(fused)

        return output


class MultiModalPredictionModel(nn.Module):
    """
    Multi-modal model for predicting future diagnostics.

    Input: All modalities at time t (input_frames)
    Output: Diagnostic signals at time t + horizon (target_frames)
    """

    def __init__(
            self,
            feature_dim=64,
            fusion_mode="concat",  # "concat" or "attention"
            text_model_name="distilbert-base-uncased",
            # Prediction output specs: (channels, target_frames)
            # These should match the target shapes from your dataset
            target_frames=50,  # Number of frames to predict (adjust based on your setup)
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.fusion_mode = fusion_mode
        self.target_frames = target_frames

        # ====== Input Encoders ======
        # Spectrogram processors
        self.mhr_processor = SpectrogramProcessor(in_channels=8, out_features=feature_dim)
        self.ece_processor = SpectrogramProcessor(in_channels=48, out_features=feature_dim)
        self.co2_processor = SpectrogramProcessor(in_channels=4, out_features=feature_dim)

        # Time series processors
        self.actuator_processor = TimeSeriesProcessor(in_channels=32, out_features=feature_dim)
        self.diagnostic_processor = TimeSeriesProcessor(in_channels=119, out_features=feature_dim)

        # Video processors
        self.bolo_processor = VideoProcessor(in_channels=1, out_features=feature_dim)
        self.irtv_processor = VideoProcessor(in_channels=1, out_features=feature_dim)
        self.tangtv_processor = VideoProcessor(in_channels=1, out_features=feature_dim)

        # Text processor
        print(f"Loading pre-trained text model: {text_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_projector = nn.Sequential(
            nn.Linear(768, feature_dim),
            nn.ReLU(),
        )

        # ====== Fusion ======
        self.num_modalities = 9

        if fusion_mode == "attention":
            self.fusion = CrossModalAttention(
                feature_dim=feature_dim,
                num_modalities=self.num_modalities,
                num_heads=4
            )
            fused_dim = feature_dim
        else:
            fused_dim = feature_dim * self.num_modalities

        # ====== Prediction Heads ======
        # Separate head for each diagnostic signal
        # d_alpha: 6 channels
        self.d_alpha_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 6 * target_frames),  # 6 channels * target_frames
        )

        # mse: 69 channels
        self.mse_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 69 * target_frames),  # 69 channels * target_frames
        )

        # ts_core_density: 44 channels
        self.ts_core_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 44 * target_frames),  # 44 channels * target_frames
        )

    def set_fusion_mode(self, mode):
        """Switch between fusion modes."""
        assert mode in ["concat", "attention"], "Mode must be 'concat' or 'attention'"
        self.fusion_mode = mode

    def encode(self, inputs):
        """Encode all input modalities to fused representation.

        Args:
            inputs: Dictionary of input modalities

        Returns:
            fused: Encoded representation (batch, fused_dim)
        """
        features = []
        device = next(self.parameters()).device

        # Process spectrograms
        features.append(self.mhr_processor(inputs["mhr"]))
        features.append(self.ece_processor(inputs["ece"]))
        features.append(self.co2_processor(inputs["co2"]))

        # Process actuators
        actuators = torch.cat([
            inputs["gas"].squeeze(2),      # (batch, 5, time_frames)
            inputs["ech"].squeeze(2),      # (batch, 11, time_frames)
            inputs["pin"].squeeze(2),      # (batch, 8, time_frames)
            inputs["tin"].squeeze(2),      # (batch, 8, time_frames)
        ], dim=1)  # (batch, 32, time_frames)
        features.append(self.actuator_processor(actuators))

        # Process diagnostics
        diagnostics = torch.cat([
            inputs["d_alpha"].squeeze(2),           # (batch, 6, time_frames)
            inputs["mse"].squeeze(2),               # (batch, 69, time_frames)
            inputs["ts_core_density"].squeeze(2),  # (batch, 44, time_frames)
        ], dim=1)  # (batch, 119, time_frames)
        features.append(self.diagnostic_processor(diagnostics))

        # Process videos
        features.append(self.bolo_processor(inputs["bolo"]))
        features.append(self.irtv_processor(inputs["irtv"]))
        features.append(self.tangtv_processor(inputs["tangtv"]))

        # Process text
        if "text" in inputs and inputs["text"] is not None:
            text_batch = []
            for text_item in inputs["text"]:
                if isinstance(text_item, list):
                    text_batch.append(" ".join([str(s) for s in text_item]))
                else:
                    text_batch.append(str(text_item))

            encoded = self.tokenizer(
                text_batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                output = self.text_encoder(input_ids, attention_mask=attention_mask)

            cls_token = output.last_hidden_state[:, 0, :]
            features.append(self.text_projector(cls_token))

        # Fusion
        if self.fusion_mode == "attention":
            fused = self.fusion(features)
        else:
            fused = torch.cat(features, dim=1)

        return fused

    def forward(self, batch):
        """
        Args:
            batch: Dictionary with:
                - 'inputs': Input modalities at time t
                - 'targets': Target diagnostics at time t+horizon (only during training)

        Returns:
            predictions: Dictionary with predicted diagnostic signals
                - 'd_alpha': (batch, 6, 1, target_frames)
                - 'mse': (batch, 69, 1, target_frames)
                - 'ts_core_density': (batch, 44, 1, target_frames)
        """
        # Get inputs (handle both training and inference formats)
        if 'inputs' in batch:
            inputs = batch['inputs']
        else:
            inputs = batch

        # Encode inputs
        fused = self.encode(inputs)
        batch_size = fused.shape[0]

        # Generate predictions for each diagnostic
        predictions = {}

        # d_alpha prediction
        d_alpha_flat = self.d_alpha_head(fused)  # (batch, 6 * target_frames)
        predictions['d_alpha'] = d_alpha_flat.view(batch_size, 6, 1, self.target_frames)

        # mse prediction
        mse_flat = self.mse_head(fused)  # (batch, 69 * target_frames)
        predictions['mse'] = mse_flat.view(batch_size, 69, 1, self.target_frames)

        # ts_core_density prediction
        ts_flat = self.ts_core_head(fused)  # (batch, 44 * target_frames)
        predictions['ts_core_density'] = ts_flat.view(batch_size, 44, 1, self.target_frames)

        return predictions
