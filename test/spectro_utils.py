import librosa
import numpy as np

def audio_to_melspectrogram(waveform, sr=16000, n_mels=128):
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec
