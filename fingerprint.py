import numpy as np
from utils import get_chroma
from sklearn.metrics.pairwise import cosine_similarity

def generate_chroma_fingerprint(data, rate):
    chroma = get_chroma(data, rate)
    return chroma


def match_chroma_fingerprints(query_chroma, db_chroma, threshold=0.8):
    similarity_matrix = cosine_similarity(query_chroma.T, db_chroma.T)
    max_similarity = np.max(similarity_matrix, axis=1)
    match_indices = np.where(max_similarity >= threshold)[0]
    return match_indices

