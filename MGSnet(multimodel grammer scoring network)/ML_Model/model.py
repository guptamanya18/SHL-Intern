import torch
import torch.nn as nn

class MGSNetModel(nn.Module):
    def __init__(self, audio_channels=1, text_embedding_dim=768, grammar_dim=6, output_dim=1):
        super(MGSNetModel, self).__init__()

        # 1D CNN for audio waveform (raw audio) processing
        self.audio_cnn = nn.Sequential(
            nn.Conv1d(audio_channels, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global pooling
        )

        # 2D CNN for Mel-spectrogram processing
        self.spec_cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global pooling
        )

        # Text embedding layer (using a pre-trained embedding model such as BERT)
        self.text_fc = nn.Sequential(
            nn.Linear(text_embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )

        # Grammar feature processing (fully connected layer)
        self.grammar_fc = nn.Sequential(
            nn.Linear(grammar_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )

        # Fusion layer to combine audio, text, and grammar features
        self.fusion_fc = nn.Sequential(
            nn.Linear(256 + 256 + 256 + 64, 512),  # audio + spectrogram + text + grammar
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Output layer (final prediction)
        self.output_fc = nn.Linear(256, output_dim)

    def forward(self, waveform, spectrogram, text_embedding, grammar_vector):
        # Process waveform through audio CNN
        audio_features = self.audio_cnn(waveform)  # Shape: [batch_size, 256, 1]
        audio_features = audio_features.squeeze(-1)  # Remove extra dimension, shape: [batch_size, 256]

        # Process Mel-spectrogram through 2D CNN
        spec_features = self.spec_cnn(spectrogram)  # Shape: [batch_size, 256, 1, 1]
        spec_features = spec_features.squeeze(-1).squeeze(-1)  # Shape: [batch_size, 256]

        # Process text embedding
        text_features = self.text_fc(text_embedding)  # Shape: [batch_size, 256]

        # Process grammar features
        grammar_features = self.grammar_fc(grammar_vector)  # Shape: [batch_size, 64]

        # Concatenate all features
        combined_features = torch.cat((audio_features, spec_features, text_features, grammar_features), dim=1)

        # Pass through the fusion layer
        fused_features = self.fusion_fc(combined_features)

        # Output prediction
        output = self.output_fc(fused_features)  # Shape: [batch_size, 1]

        return output
