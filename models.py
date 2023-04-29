import torch
import torch.nn as nn
from torchvision import models
import torchaudio.transforms as T

from constants import AUDIO_SAMPLE_RATE


class VideoAudioMatchingModel(nn.Module):
    def __init__(self):
        super(VideoAudioMatchingModel, self).__init__()
        
        # Video sub-model (ResNet with GRU)
        self.video_cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.video_cnn.fc.in_features
        self.video_cnn.fc = nn.Identity()
        self.video_gru = nn.GRU(input_size=num_features, hidden_size=512, num_layers=1, batch_first=True)
        
        # Audio sub-model (Transfer learning from torchaudio)
        self.audio_preprocess = nn.Sequential(
            T.MelSpectrogram(sample_rate=AUDIO_SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128),
            T.AmplitudeToDB()
        )
        self.audio_cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.audio_cnn.fc = nn.Linear(num_features, 512)
        
        # Combine sub-models
        self.fc = nn.Linear(1024, 1)

    def forward(self, video, audio):
        batch_size, seq_len, c, h, w = video.size()
        video = video.view(batch_size * seq_len, c, h, w)
        video_features = self.video_cnn(video)
        video_features = video_features.view(batch_size, seq_len, -1)
        _, hidden = self.video_gru(video_features)
        video_output = hidden[-1]

        spectrogram = self.audio_preprocess(audio)
        spectrogram_3channel = spectrogram.repeat(1, 3, 1, 1)
        audio_features = self.audio_cnn(spectrogram_3channel)

        combined = torch.cat((video_output, audio_features), dim=1)
        output = torch.sigmoid(self.fc(combined))
        return output
