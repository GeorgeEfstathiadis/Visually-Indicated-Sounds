import torch
import torch.nn as nn
from torchvision import models
import torchaudio.transforms as T
import torchvision.models.video as video_models

from constants import AUDIO_SAMPLE_RATE


class VideoAudioMatchingModel(nn.Module):
    """
    Model that combines video and audio sub-models to predict whether a video and audio clip match

    ...

    Description
    -----------
    The video sub-model combines:
    * A ResNet18 model (pre-trained on ImageNet) to extract features from each frame with the last 
    fully connected layer removed
    * A GRU to uses the features from each frame to extract a single feature vector for the video

    The audio sub-model combines:
    * A MelSpectrogram to extract a spectrogram from the audio clip
    * A ResNet18 model (pre-trained on ImageNet) to extract features from the spectrogram with the last
    fully connected layer removed

    The video and audio sub-models are combined by concatenating the video feature vector and audio feature
    and passing it through a fully connected layer to get a single output value.
    """
    def __init__(self):
        super(VideoAudioMatchingModel, self).__init__()
        
        # Video sub-model (ResNet with GRU)
        self.video_cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.video_cnn.fc.in_features
        self.video_cnn.fc = nn.Identity()
        # freeze all layers of resnet except last layer
        for param in self.video_cnn.parameters():
            param.requires_grad = False
        for param in self.video_cnn.fc.parameters():
            param.requires_grad = True
        self.video_gru = nn.GRU(input_size=num_features, hidden_size=512, num_layers=1, batch_first=True)

        # Audio sub-model (Transfer learning from torchaudio)
        self.audio_preprocess = nn.Sequential(
            T.MelSpectrogram(sample_rate=AUDIO_SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128),
            T.AmplitudeToDB()
        )
        self.audio_cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.audio_cnn.fc = nn.Linear(num_features, 512)
        for param in self.audio_cnn.parameters():
            param.requires_grad = False
        for param in self.audio_cnn.fc.parameters():
            param.requires_grad = True
        
        # Combine sub-models
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, video, audio):
        batch_size, seq_len, c, h, w = video.size()
        video = video.view(batch_size * seq_len, c, h, w)
        video_features = self.video_cnn(video)
        video_features = video_features.view(batch_size, seq_len, -1)
        _, hidden = self.video_gru(video_features)
        video_output = hidden[-1] # 512

        spectrogram = self.audio_preprocess(audio)
        spectrogram = spectrogram.unsqueeze(1) # add channel dimension
        spectrogram_3channel = spectrogram.repeat(1, 3, 1, 1)
        audio_features = self.audio_cnn(spectrogram_3channel) # 512

        combined = torch.cat((video_output, audio_features), dim=1)
        output = torch.sigmoid(self.fc(combined))
        return output

class VideoAudioMatchingModel2(nn.Module):
    """
    Model that combines video and audio sub-models to predict whether a video and audio clip match

    ...

    Description
    -----------
    The video sub-model combines:
    * A ResNet18 model (pre-trained on ImageNet) to extract features from each frame with the last 
    fully connected layer removed
    * A GRU to uses the features from each frame to extract a single feature vector for the video

    The audio sub-model combines:
    * A MelSpectrogram to extract a spectrogram from the audio clip
    * A ResNet18 model (pre-trained on ImageNet) to extract features from the spectrogram with the last
    fully connected layer removed

    The video and audio sub-models are combined by concatenating the video feature vector and audio feature
    and passing it through a fully connected layer to get a single output value.
    """
    def __init__(self):
        super(VideoAudioMatchingModel2, self).__init__()
        
        # Video sub-model (ResNet3D)
        self.video_cnn = video_models.r3d_18(pretrained=True)
        num_features = self.video_cnn.fc.in_features
        self.video_cnn.fc = nn.Identity()
        # freeze all layers of resnet except last layer
        for param in self.video_cnn.parameters():
            param.requires_grad = False
        for param in self.video_cnn.fc.parameters():
            param.requires_grad = True
        self.video_gru = nn.GRU(input_size=num_features, hidden_size=512, num_layers=1, batch_first=True)

        # Audio sub-model (Transfer learning from torchaudio)
        self.audio_preprocess = nn.Sequential(
            T.MelSpectrogram(sample_rate=AUDIO_SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128),
            T.AmplitudeToDB()
        )
        self.audio_cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.audio_cnn.fc = nn.Linear(num_features, 512)
        for param in self.audio_cnn.parameters():
            param.requires_grad = False
        for param in self.audio_cnn.fc.parameters():
            param.requires_grad = True
        
        # Combine sub-models
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, video, audio):
        batch_size, seq_len, c, h, w = video.size()
        video = torch.transpose(video, 1, 2) # (batch_size, channels, time, height, width)
        video_features = self.video_cnn(video) 
        video_output = video_features.view(video_features.size(0), -1)  # flattening

        spectrogram = self.audio_preprocess(audio)
        spectrogram = spectrogram.unsqueeze(1) # add channel dimension
        spectrogram_3channel = spectrogram.repeat(1, 3, 1, 1)
        audio_features = self.audio_cnn(spectrogram_3channel) # 512

        combined = torch.cat((video_output, audio_features), dim=1)
        output = torch.sigmoid(self.fc(combined))
        return output



class VideoAudioMatchingModelConv(nn.Module):
    """
    Model that applies convolution layer to initial layer of resnet in the time dimension
    to predict whether a video and audio clip match

    ...

    Description
    -----------
    @TODO
    """
    # model that adds convolution layer to initial layer of resnet
    def __init__(self):
        super(VideoAudioMatchingModelConv, self).__init__()
        
        # Video sub-model (Conv -> ResNet)
        # @TODO not enough memory to time-convolute 3*w*h, we need to convolve spatially first
        self.conv1 = nn.Conv1d(3*256*456, 3*256*456, kernel_size=24, stride=11)
        self.video_cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.video_cnn.fc.in_features

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
        # shape (N, n_frames, c, w, h) to (N, c*w*h, n_frames)
        video = torch.reshape(torch.transpose(video, 1, 2), (batch_size, c*h*w, seq_len))
        video = self.conv1(video)
        # shape (N, c*w*h, n_frames) to (N, n_frames, c, w, h)
        video = torch.reshape(torch.transpose(video, 1, 2), (batch_size, video.size(2), c, h, w))

        new_seq_len = video.size(1)

        video = video.view(batch_size * new_seq_len, c, h, w)
        video_features = self.video_cnn(video)
        video_features = video_features.view(batch_size, new_seq_len, -1)

        spectrogram = self.audio_preprocess(audio)
        spectrogram_3channel = spectrogram.repeat(1, 3, 1, 1)
        audio_features = self.audio_cnn(spectrogram_3channel) # 512

        combined = torch.cat((video_features, audio_features), dim=1)
        output = torch.sigmoid(self.fc(combined))
        return output