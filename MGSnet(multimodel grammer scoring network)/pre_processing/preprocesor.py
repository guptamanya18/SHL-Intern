# Import required libraries
import librosa
import numpy as np
import whisper
import warnings
from transformers import AutoTokenizer
import torch
from transformers import AutoModel
import language_tool_python
from langdetect import detect, DetectorFactory


# For consistent results in language detection
DetectorFactory.seed = 0

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Function to load audio and preprocess it
def load_audio(file_path, sr=16000, duration=5.0):
    waveform, _ = librosa.load(file_path, sr=sr)
    target_length = int(sr * duration)

    if len(waveform) < target_length:
        pad = target_length - len(waveform)
        waveform = np.pad(waveform, (0, pad))
    else:
        waveform = waveform[:target_length]

    waveform = waveform / np.max(np.abs(waveform))  # Normalize
    return waveform, sr

# Function to convert audio to mel spectrogram
def audio_to_melspectrogram(waveform, sr=16000, n_mels=128):
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

# Load the Whisper model once
model = whisper.load_model("base")
warnings.filterwarnings("ignore", category=UserWarning, message="FP16 is not supported on CPU")

# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# Function to transcribe and filter audio based on language
def transcribe_audio_with_whisper(file_path):
    result = model.transcribe(file_path)
    text = result['text'].strip()

    lang = detect_language(text)
    if lang != 'en':
        return f"[AUDIO_REJECTED] Non-English content detected: {lang}"

    return text

# Load BERT model for embeddings
bert_model = AutoModel.from_pretrained("bert-base-uncased")
bert_model.eval()

# Load grammar tool
grammar_tool = language_tool_python.LanguageTool('en-US')

# Function to tokenize transcribed text
def tokenize_text(text):
    return tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")

# Function to get BERT-based embedding for the full sentence
def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # Take [CLS] token
    return cls_embedding.squeeze(0)  # Shape: (768,)

# Function to generate grammar error vector and score
def get_grammar_features(text):
    matches = grammar_tool.check(text)

    # Error vector — binary vector of categories
    categories = ['GRAMMAR', 'PUNCTUATION', 'TYPOGRAPHY', 'STYLE', 'CASING', 'OTHER']
    error_vector = torch.zeros(len(categories))

    for match in matches:
        category = match.ruleIssueType.upper()
        if category in categories:
            idx = categories.index(category)
            error_vector[idx] += 1
        else:
            error_vector[-1] += 1  # Count as OTHER

    # Grammar score = 1 - (#errors / #words) ∈ [0,1]
    num_errors = len(matches)
    num_words = len(text.split())
    grammar_score = 1.0 - (num_errors / max(1, num_words))
    grammar_score = torch.tensor(grammar_score, dtype=torch.float)

    return {
        'error_vector': error_vector,           # Tensor: [GRAMMAR, PUNCTUATION, TYPO, STYLE, CASING, OTHER]
        'score': grammar_score                  # Float tensor between 0–1
    }
