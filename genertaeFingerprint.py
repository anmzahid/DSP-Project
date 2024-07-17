import os
import numpy as np
import acoustid
from pydub import AudioSegment
from pydub.utils import which
import glob

# Ensure ffmpeg and fpcalc are set correctly
# os.environ["PATH"] += os.pathsep + r"C:\path\to\ffmpeg\bin"
# os.environ["PATH"] += os.pathsep + r"C:\path\to\chromaprint"

# # Verify that pydub can find ffmpeg
# if which("ffmpeg") is None:
#     print("ffmpeg is not found. Please check your PATH.")
# else:
#     print("ffmpeg is found.")

# # Verify that acoustid can find fpcalc
# if which("fpcalc") is None:
#     print("fpcalc is not found. Please check your PATH.")
# else:
#     print("fpcalc is found.")

def preprocess_audio(file_path, target_length=30):
    audio = AudioSegment.from_file(file_path)
    
    if len(audio) > target_length * 1000:
        audio = audio[:target_length * 1000]  # Truncate to the target length
    elif len(audio) < target_length * 1000:
        padding = AudioSegment.silent(duration=(target_length * 1000 - len(audio)))
        audio = audio + padding  # Pad with silence to the target length
    
    return audio

def generate_fingerprint(file_path):
    try:
        preprocessed_audio = preprocess_audio(file_path)
        preprocessed_file_path = file_path.replace('.wav', '_preprocessed.wav')
        preprocessed_audio.export(preprocessed_file_path, format='wav')
        
        duration, fingerprint = acoustid.fingerprint_file(preprocessed_file_path)
        fingerprint = np.array(fingerprint, dtype=np.float32)  # Ensure correct data type
        return fingerprint
    except Exception as e:
        print(f"Error generating fingerprint for {file_path}: {e}")
        return None

def save_fingerprint(file_path, fingerprint):
    npy_file_path = file_path.replace('.wav', '.npy')
    np.save(npy_file_path, fingerprint)
    return npy_file_path

def main():
    audio_files_dir = 'audio_files'
    audio_files = glob.glob(os.path.join(audio_files_dir, '*.wav'))
    fingerprints = {}

    for file in audio_files:
        fingerprint = generate_fingerprint(file)
        if fingerprint is not None:
            npy_file_path = save_fingerprint(file, fingerprint)
            fingerprints[file] = npy_file_path
            print(f"Fingerprint saved for {file} -> {npy_file_path}")
        else:
            print(f"Failed to generate fingerprint for {file}")

if __name__ == "__main__":
    main()