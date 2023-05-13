import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from matplotlib import pyplot as plt
from scipy.io import wavfile

from constants import VIDEO_FRAME_RATE, AUDIO_SAMPLE_RATE

class VATransform:
    """Transforms for video and audio data"""

    NONE = 0
    RANDOM_SEGMENT = 1
    IMG_DOWNSAMPLE = 2
    FRAME_DOWNSAMPLE = 3
    DATA_AUGMENT = 4

    def get_random_segment(n_frames_video, n_samples_audio, seconds):
        """Get a random segment of the video and audio of the specified length

        Args:
            n_frames_video (int): number of frames in the video
            n_samples_audio (int): number of samples in the audio
            seconds (int): length of the segment in seconds

        Returns:
            tuple: (video_frame_start, video_frame_end), (audio_frame_start, audio_frame_end)
        """

        time_length = min(n_frames_video / VIDEO_FRAME_RATE, n_samples_audio / AUDIO_SAMPLE_RATE)
        assert time_length >= seconds, "Video and audio must be at least as long as the requested segment length (even for non-matching tracks!)"
            
        time_start = np.random.uniform(0, time_length - seconds)
            
        video_frame_start = int(time_start * VIDEO_FRAME_RATE)
        video_frame_end = video_frame_start + int(seconds * VIDEO_FRAME_RATE)
        audio_frame_start = int(time_start * AUDIO_SAMPLE_RATE)
        audio_frame_end = audio_frame_start + int(seconds * AUDIO_SAMPLE_RATE)
            
        return (video_frame_start, video_frame_end), (audio_frame_start, audio_frame_end)
        
    def downsample_img(frame, factor=2):
        """Downsample image height and width
        
        Args:
            frame (np ndarray): of shape (height, width, n_channels)
            factor (int): factor to downsample by
        Returns:
            np ndarray: of shape (height // factor, width // factor, n_channels)
        """

        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        frame = frame[::factor, ::factor, :]

        return frame
    
    def data_augment(video, **transform_args):
        """Uses torchvision.transform to randomly change brightness, flip, or rotate the image.
        
        Args:
            video (np ndarray): of shape (n_frames, height, width, n_channels), mode='RGB'
        Returns:
            np ndarray: of shape (n_frames, height, width, n_channels)
        """

        augment_args = {
            'brightness_lower': transform_args.get('brightness_lower', 0.75),
            'brightness_upper': transform_args.get('brightness_upper', 1.25),
            'p_hflip': transform_args.get('p_hflip', 0.5),
            'degrees_range': transform_args.get('degrees_range', (-10, 10))
        }

        rng_brightness = np.random.uniform(augment_args['brightness_lower'], augment_args['brightness_upper'])
        rng_hflip = np.random.uniform()
        rng_rotate = np.random.uniform(augment_args['degrees_range'][0], augment_args['degrees_range'][1])


        frames = []

        for frame in video:

            # reshape from chw to hwc
            frame = np.transpose(frame, (1, 2, 0)).cpu().numpy()
            plt.imshow(frame)
            plt.show()

            # print("in: ", len(video), "x", frame.shape)

            frame = transforms.functional.to_pil_image(frame, mode='RGB')
            frame = transforms.functional.adjust_brightness(frame, rng_brightness)
            frame = transforms.functional.hflip(frame) if rng_hflip < 0.5 else frame
            frame = transforms.functional.rotate(frame, rng_rotate)

            frame = transforms.functional.to_tensor(frame)
            plt.imshow(frame.cpu().numpy().transpose(1, 2, 0))
            plt.show()
            frames.append(frame)

        # print("out: ", len(frames), "x", frames[0].shape)
        return frames



class VideoAudioDataset(Dataset):
    """Video and audio dataset"""

    def __init__(
        self, dataset, device, filepath_prefix = '',
        use_cache = False,
        transform=[VATransform.NONE],
        **transform_args
    ):
        """
        Args:
            dataset (np ndarray): of shape (n_samples, 3) where each row is (video_filename, audio_filename, label)
            device (torch.device): device to store the data on
            filepath_prefix (str): prefix to add to the video and audio filenames
            use_cache (bool): whether to cache the data in memory
            transform (list): list of transforms to apply to the data
            transform_args (dict): arguments for the transforms

            transform is a list that can contain the following values:
                VATransform.NONE: no transform
                VATransform.RANDOM_SEGMENT: random segment of the video and audio
                VATransform.IMG_DOWNSAMPLE: downsample the image
                VATransform.FRAME_DOWNSAMPLE: downsample the frames
                VATransform.DATA_AUGMENT: data augmentation

            transform_args can contain the following keys:
                random_segment_seconds (int): length of the random segment in seconds
                img_downsample_factor (int): factor to downsample the image by
                frame_downsample_factor (int): factor to downsample the frames by
        """
        self.dataset = dataset
        self.device = device
        self.transform = transform if isinstance(transform, list) else [transform]
        self.transform_args = transform_args
        self.filepath_prefix = filepath_prefix
        self.use_cache = use_cache
        self.cache = {}

        if VATransform.RANDOM_SEGMENT in self.transform and not 'random_segment_seconds' in self.transform_args:
            self.transform_args['random_segment_seconds'] = 5
            print("WARNING: random_segment_seconds not specified, defaulting to 5 seconds")
        if VATransform.IMG_DOWNSAMPLE in self.transform and not 'img_downsample_factor' in self.transform_args:
            self.transform_args['img_downsample_factor'] = 2
            print("WARNING: img_downsample_factor not specified, defaulting to 2")
        if VATransform.FRAME_DOWNSAMPLE in self.transform and not 'frame_downsample_factor' in self.transform_args:
            self.transform_args['frame_downsample_factor'] = 2
            print("WARNING: frame_downsample_factor not specified, defaulting to 2")

    def __len__(self):
        """
        Returns:
            int: number of samples in the dataset
        """
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

        if VATransform.RANDOM_SEGMENT in self.transform:
            n_frames_video = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            n_samples_audio = audio.shape[0]
            (video_frame_start, video_frame_end), (audio_frame_start, audio_frame_end) = VATransform.get_random_segment(
                n_frames_video, n_samples_audio, self.transform_args['random_segment_seconds']
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_start)

        # Read frames and store them in a list
        f_pos = 0
        while True:
            ret, frame = cap.read()
            f_pos += 1
            if not ret or (VATransform.RANDOM_SEGMENT in self.transform and cap.get(cv2.CAP_PROP_POS_FRAMES) > video_frame_end):
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            if VATransform.IMG_DOWNSAMPLE in self.transform:
                frame = VATransform.downsample_img(frame, factor=self.transform_args['img_downsample_factor'])
            frame = torch.from_numpy(frame).permute(2, 0, 1)  # Convert to CHW format
            frames.append(frame)

        # Release the VideoCapture object
        cap.release()
        
        if VATransform.FRAME_DOWNSAMPLE in self.transform:
            frames = frames[::self.transform_args['frame_downsample_factor']]

        if VATransform.DATA_AUGMENT in self.transform:
            frames = VATransform.data_augment(frames, **self.transform_args) # shape n x h x w x c

        # Stack the list of frames into a single tensor
        video = torch.stack(frames).float() # (n_frames, height, width, n_channels)
        video = video / 255.0 # Normalize pixel values to [0, 1]
        
        if VATransform.RANDOM_SEGMENT in self.transform:
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
        """
        Empties the cache of the DataLoader.
        """
        assert self.use_cache, "Cannot empty cache if DataLoader was initialized with parameter use_cache = False"

        del self.cache
        self.cache = {}


class ResamplingSampler(Sampler):
    """
    Torch sampler for training data segmentation.
    """
    def __init__(self, data_source, batch_size, no_batches=None, replacement=True):
        """
        Args:
            data_source (Dataset): dataset to sample from
            batch_size (int): batch size
            no_batches (int): number of batches to sample
            replacement (bool): whether to sample with replacement
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.replacement = replacement
        self.no_batches = no_batches if no_batches is not None else len(data_source) // batch_size

    def __iter__(self):
        """
        Returns:
            iterator: iterator over batch indices
        """
        n = len(self.data_source)
        for _ in range(self.no_batches):
            batch_indices = np.random.choice(n, self.batch_size, replace=self.replacement)
            yield batch_indices

    def __len__(self):
        """
        Returns:
            int: number of batches
        """
        return self.no_batches
