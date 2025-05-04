'''
from pydub import AudioSegment
import speech_recognition as sr


# Function to preprocess audio
def preprocess_audio(input_audio_path, output_audio_path):
    # Load the audio file
    audio = AudioSegment.from_wav(input_audio_path)

    # Optional: Normalize audio volume (reduce audio clipping)
    audio = audio.apply_gain(-audio.dBFS)

    # Optional: Trim silence (you can adjust silence thresholds as needed)
    audio = audio.strip_silence(silence_len=1000, silence_thresh=-40)

    # Save the processed audio file
    audio.export(output_audio_path, format="wav")
    return output_audio_path


# Path to the raw audio and pre-processed audio
raw_audio_path = "E:\\Hackathon\\SHL_Intern\\assets\\Dataset\\audios\\train\\audio_4.wav"
processed_audio_path = "audio_4_processed.wav"

# Preprocess the audio
processed_audio = preprocess_audio(raw_audio_path, processed_audio_path)

# Now you can pass this preprocessed audio to the transcription function
recognizer = sr.Recognizer()


def transcribe_audio(file_path):
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)  # Capture the audio data

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"


# Transcribe the preprocessed audio
transcription = transcribe_audio(processed_audio_path)
print("Transcript:", transcription)

'''

import whisper
import warnings

# Load the Whisper model
model = whisper.load_model("base")  # You can change to "small", "medium", or "large" as per your need

warnings.filterwarnings("ignore", category=UserWarning, message="FP16 is not supported on CPU")

def transcribe_audio_with_whisper(file_path):
    # Transcribe audio using Whisper
    result = model.transcribe(file_path)
    return result['text']
