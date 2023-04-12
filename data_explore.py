# open .mat and .pk files
import scipy.io
import numpy as np

data = scipy.io.loadmat('vis-sfs/vis-data/2015-02-16-17-27-53_sf.mat')
data['sfs'].shape # (time, 45, 42)

# open corresponding .pk file
import pickle
with open('vis-sfs/vis-data/2015-02-16-17-27-53_sf.pk', 'rb') as f:
    data2 = pickle.load(f, encoding='latin1')
np.sum(data2 != data['sfs']) # 0 # same data

# open .mp4 file
import cv2
cap = cv2.VideoCapture('vis-data-256/vis-data-256/2015-02-16-16-49-06_mic.mp4')
ret, frame = cap.read()
frame.shape # (256, 456, 3)
# play video
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        cv2.imshow('frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

# open .wav file
import wave

filename = 'vis-data-256/vis-data-256/2015-02-16-16-49-06_mic.wav'

with wave.open(filename, 'rb') as wav_file:
    n_channels = wav_file.getnchannels()
    sample_width = wav_file.getsampwidth()
    frame_rate = wav_file.getframerate()
    n_frames = wav_file.getnframes()
    duration = n_frames / float(frame_rate)
    
    print(f'Channels: {n_channels}')
    print(f'Sample width: {sample_width}')
    print(f'Frame rate: {frame_rate}')
    print(f'Number of frames: {n_frames}')
    print(f'Duration: {duration} seconds')

    frames = wav_file.readframes(n_frames)

import numpy as np
audio_data = np.frombuffer(frames, dtype=np.int16 if sample_width == 2 else np.int8)
audio_data.shape # (n_frames * n_channels,)

if n_channels == 2:
    left_channel = audio_data[::2]
    right_channel = audio_data[1::2]
else:
    left_channel = audio_data

import matplotlib.pyplot as plt
# plot 0.1 seconds of audio
time = np.linspace(0, len(left_channel) / frame_rate, num=len(left_channel))[:10000]

plt.figure(figsize=(10, 4))
plt.plot(time, left_channel[:10000], label='Left channel')
if n_channels == 2:
    plt.plot(time, right_channel[:10000], label='Right channel', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Signal')
plt.legend()
plt.show()
