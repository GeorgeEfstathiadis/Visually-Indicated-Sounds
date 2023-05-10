import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from scipy.io import wavfile

from constants import VIDEO_FRAME_RATE, AUDIO_SAMPLE_RATE


class VideoAudioDataset(Dataset):

    class Transform:
        NONE = 0
        RANDOM_SEGMENT = 1

        def get_random_segment(n_frames_video, n_samples_audio, seconds):
            time_length = min(n_frames_video / VIDEO_FRAME_RATE, n_samples_audio / AUDIO_SAMPLE_RATE)
            assert time_length >= seconds, "Video and audio must be at least as long as the requested segment length (even for non-matching tracks!)"
            
            time_start = np.random.uniform(0, time_length - seconds)
            
            video_frame_start = int(time_start * VIDEO_FRAME_RATE)
            video_frame_end = video_frame_start + int(seconds * VIDEO_FRAME_RATE)
            audio_frame_start = int(time_start * AUDIO_SAMPLE_RATE)
            audio_frame_end = audio_frame_start + int(seconds * AUDIO_SAMPLE_RATE)
            
            return (video_frame_start, video_frame_end), (audio_frame_start, audio_frame_end)

    def __init__(
        self, dataset, device, filepath_prefix = '',
        transform=Transform.NONE, downsample_factor=1,
        use_cache = False, **transform_args
    ):
        self.dataset = dataset
        self.device = device
        self.transform = transform
        self.downsample_factor = downsample_factor
        self.transform_args = transform_args
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

        # Load audio first
        _, audio = wavfile.read(audio_path) # (n_samples, 2)

        # Average the two channels
        audio = np.mean(audio, axis=1)
        audio = torch.from_numpy(audio).float() # (n_samples,)

        # Load video
        frames = []
        cap = cv2.VideoCapture(video_path)

        if self.transform == VideoAudioDataset.Transform.RANDOM_SEGMENT:
            n_frames_video = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            n_samples_audio = audio.shape[0]
            (video_frame_start, video_frame_end), (audio_frame_start, audio_frame_end) = VideoAudioDataset.Transform.get_random_segment(
                n_frames_video, n_samples_audio,
                self.transform_args['random_segment_seconds'] if 'random_segment_seconds' in self.transform_args else 5
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_start)

        # Read frames and store them in a list
        while True:
            ret, frame = cap.read()
            if not ret or (self.transform == VideoAudioDataset.Transform.RANDOM_SEGMENT and cap.get(cv2.CAP_PROP_POS_FRAMES) > video_frame_end):
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            if self.downsample_factor > 1:
                frame = downsample_frame(frame, factor=self.downsample_factor)
            frame = torch.from_numpy(frame).permute(2, 0, 1)  # Convert to CHW format
            frames.append(frame)

        # Release the VideoCapture object
        cap.release()

        # Stack the list of frames into a single tensor
        video = torch.stack(frames).float() # (n_frames, height, width, n_channels)
        video = video / 255.0 # Normalize pixel values to [0, 1]
        
        if self.transform == VideoAudioDataset.Transform.RANDOM_SEGMENT:
            audio = audio[audio_frame_start:audio_frame_end]

        if video.shape[0]==0:
            raise ValueError("Video [", video_path, "] - found 0 frames when loaded.")
        if audio.shape[0]==0:
            raise ValueError("Audio [", audio_path, "] - found 0 frames when loaded.")

        # Add to cache
        if self.use_cache:
            self.cache[idx] = (video, audio)

        # Device storage
        video = video.to(self.device)
        audio = audio.to(self.device)

        label = self.dataset[idx, 2].astype(np.int64)

        return video, audio, label

    def empty_cache(self):
        assert self.use_cache, "Cannot empty cache if DataLoader was initialized with parameter use_cache = False"

        del self.cache
        self.cache = {}


def downsample_frame(frame, factor=2):
    """Downsample image height and width
    
    Args:
        frame (np ndarray): of shape (height, width, n_channels)
        factor (int): factor to downsample by
    Returns:
        np ndarray: of shape (height // factor, width // factor, n_channels)
    """

    # first apply gaussian blur to every frame
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # then downsample
    frame = frame[::factor, ::factor, :]

    return frame
