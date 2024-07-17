# utils.py
import numpy as np
import soundfile as sf
import librosa
import acoustid

# Function to preprocess audio
def preprocess_audio(file_path, target_length=30):
    audio, rate = librosa.load(file_path, sr=None, mono=True)
    if len(audio) > target_length * rate:
        audio = audio[:target_length * rate]  # Truncate to the target length
    elif len(audio) < target_length * rate:
        padding = np.zeros((target_length * rate - len(audio),))
        audio = np.concatenate((audio, padding))  # Pad with zeros to the target length
    return audio, rate

# Function to generate chroma fingerprint
def generate_chroma_fingerprint(data, rate):
    chroma = librosa.feature.chroma_stft(data, sr=rate)
    return chroma

# Function to match fingerprints
def match_fingerprints(query_fingerprint, db_fingerprint, threshold=0.8):
    similarity_matrix = np.dot(query_fingerprint.T, db_fingerprint)
    max_similarity = np.max(similarity_matrix, axis=1)
    match_indices = np.where(max_similarity >= threshold)[0]
    return match_indices
