import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from scipy.io import wavfile

from constants import VIDEO_FRAME_RATE, AUDIO_SAMPLE_RATE


class VideoAudioDataset(Dataset):
    def __init__(self, dataset, device, filepath_prefix = '', transform=None, use_cache=False):
        self.dataset = dataset
        self.device = device
        self.transform = transform
        self.filepath_prefix = filepath_prefix
        self.use_cache = use_cache
        self.cache = {}

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
        
        if self.use_cache and idx in self.cache:
            return self.cache[idx], self.dataset[idx, 2].astype(np.int64)

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
        video = torch.stack(frames).float() # (n_frames, height, width, n_channels)
        video = video / 255.0 # Normalize pixel values to [0, 1]
        video = video.to(self.device)


        # Load audio
        _, audio = wavfile.read(audio_path) # (n_frames, 2)
        # average the two channels
        audio = np.mean(audio, axis=1)
        audio = torch.from_numpy(audio).float() # (n_frames,)
        audio = audio.to(self.device)

        if self.transform:
            video, audio = self.transform(video, audio, seconds=2)

        # Add to cache
        if self.use_cache:
            self.cache[idx] = (video, audio)

        label = self.dataset[idx, 2].astype(np.int64)

        if video.shape[0]==0:
            raise ValueError("Video [", video_path, "] has 0 frames")
        if audio.shape[0]==0:
            raise ValueError("Audio [", audio_path, "] has 0 frames")

        return video, audio, label

    def empty_cache(self):
        assert self.use_cache, "Cannot empty cache if DataLoader was initialized with parameter use_cache = False"

        del self.cache
        self.cache = {}


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

    time_length = min(video.shape[0] / VIDEO_FRAME_RATE, audio.shape[0] / AUDIO_SAMPLE_RATE)
    assert time_length >= seconds, "Video and audio must be at least as long as the requested segment length (even for non-matching tracks!)"
    
    time_start = np.random.uniform(0, time_length - seconds)
    
    video_frame_start = int(time_start * VIDEO_FRAME_RATE)
    video_frame_end = video_frame_start + int(seconds * VIDEO_FRAME_RATE)
    audio_frame_start = int(time_start * AUDIO_SAMPLE_RATE)
    audio_frame_end = audio_frame_start + int(seconds * AUDIO_SAMPLE_RATE)

    video = video[video_frame_start:video_frame_end]
    audio = audio[audio_frame_start:audio_frame_end]

    return video, audio

