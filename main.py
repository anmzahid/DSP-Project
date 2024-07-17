# main.py
import os
import numpy as np
from utils import preprocess_audio, generate_chroma_fingerprint, match_fingerprints

# Constants
RECORDED_FILE = "recorded_audio.wav"
DB_DIR = "audio_files"

# Function to recognize audio
def recognize_audio(query_file, db_dir):
    # Preprocess query audio
    query_audio, query_rate = preprocess_audio(query_file)

    # Iterate through the database
    match_results = []
    for db_file in os.listdir(db_dir):
        if db_file.endswith(".wav"):
            # Load and preprocess database audio
            db_audio, db_rate = preprocess_audio(os.path.join(db_dir, db_file))

            # Generate chroma fingerprint for query and database audio
            query_chroma = generate_chroma_fingerprint(query_audio, query_rate)
            db_chroma = generate_chroma_fingerprint(db_audio, db_rate)

            # Match fingerprints
            match_indices = match_fingerprints(query_chroma, db_chroma)

            # Append match results to list
            match_results.append((db_file, len(match_indices)))

    # Sort match results by number of matches
    match_results.sort(key=lambda x: x[1], reverse=True)

    # Return the best match
    if match_results:
        return match_results[0][0]
    else:
        return None

# Main function
if __name__ == "__main__":
    result = recognize_audio(RECORDED_FILE, DB_DIR)
    print("Recognized Audio:", result)
