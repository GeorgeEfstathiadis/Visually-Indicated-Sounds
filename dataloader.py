import os
import torch
import numpy as np
from decord import VideoReader
from torch.utils.data import Dataset
from scipy.io import wavfile

VIDEO_FRAME_RATE = 29.97002997002997
AUDIO_SAMPLE_RATE = 96000

class VideoAudioDataset(Dataset):
    def __init__(self, video_files, audio_files, transform=None):
        self.video_files = video_files
        self.audio_files = audio_files
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (video, audio) where video is a numpy array of shape (n_frames, height, width, n_channels)
            and audio is a numpy array of shape (n_frames, n_channels)
        """

        video_path = self.video_files[idx]
        audio_path = self.audio_files[idx]

        # Load video
        vr = VideoReader(video_path, num_threads=1)
        video = vr.get_batch(range(len(vr))).asnumpy()

        # Load audio
        _, audio = wavfile.read(audio_path) # (n_frames, 2)

        if self.transform:
            video, audio = self.transform(video, audio, seconds=5)

        return video, audio


def get_random_segment(video, audio, seconds):
    """Extract a random segment of the same time length from video and audio.

    Args:
        video (numpy array): of shape (n_frames, height, width, n_channels)
        audio (numpy array): of shape (n_frames, n_channels)
    Returns:
        tuple: (video, audio) where video is a numpy array of shape (n_frames, height, width, n_channels)
        and audio is a numpy array of shape (n_frames, n_channels)
    """
    
    # n_frames_video = int(seconds * VIDEO_FRAME_RATE)
    # n_frames_audio = int(seconds * AUDIO_SAMPLE_RATE)

    time_length = video.shape[0] / VIDEO_FRAME_RATE
    time_start = np.random.uniform(0, time_length - seconds)
    
    video_frame_start = int(time_start * VIDEO_FRAME_RATE)
    video_frame_end = video_frame_start + int(seconds * VIDEO_FRAME_RATE)
    audio_frame_start = int(time_start * AUDIO_SAMPLE_RATE)
    audio_frame_end = audio_frame_start + int(seconds * AUDIO_SAMPLE_RATE)

    video = video[video_frame_start:video_frame_end]
    audio = audio[audio_frame_start:audio_frame_end]

    return video, audio

