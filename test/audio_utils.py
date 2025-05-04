import librosa
import numpy as np


def load_audio(file_path, sr=16000, duration=5.0):
    waveform, _ = librosa.load(file_path, sr=sr)
    target_length = int(sr * duration)

    if len(waveform) < target_length:
        pad = target_length - len(waveform)
        waveform = np.pad(waveform, (0, pad))
    else:
        waveform = waveform[:target_length]

    waveform = waveform / np.max(np.abs(waveform))  # normalize
    return waveform

