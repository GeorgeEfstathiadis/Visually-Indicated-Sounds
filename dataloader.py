import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from scipy.io import wavfile

VIDEO_FRAME_RATE = 29.97002997002997
AUDIO_SAMPLE_RATE = 96000

class VideoAudioDataset(Dataset):
    def __init__(self, dataset, device, filepath_prefix = '', transform=None):
        self.dataset = dataset
        self.device = device
        self.transform = transform
        self.filepath_prefix = filepath_prefix

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (video, audio) where video is a numpy array of shape (n_frames, height, width, n_channels)
            and audio is a numpy array of shape (n_frames, n_channels)
        """

        video_path = self.filepath_prefix + self.dataset[idx, 0]
        audio_path = self.filepath_prefix + self.dataset[idx, 1]

        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(audio_path) 
        
        # Load video
        cap = cv2.VideoCapture(video_path)

        # Read frames and store them in a list
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            frame = torch.from_numpy(frame).permute(2, 0, 1)  # Convert to CHW format
            frames.append(frame)

        # Release the VideoCapture object
        cap.release()

        # Stack the list of frames into a single tensor
        video = torch.stack(frames)
        video.to(self.device)


        # Load audio
        _, audio = wavfile.read(audio_path) # (n_frames, 2)
        audio = torch.from_numpy(audio)
        audio.to(self.device)

        if self.transform:
            video, audio = self.transform(video, audio, seconds=5)

        label = self.dataset[idx, 2].astype(np.int64)

        return video, audio, label


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

