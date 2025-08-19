import json
import pickle


def steer_evil_vector(path="cached_vectors.pkl"):
    with open(path, "rb") as f:
        cached_vectors = pickle.load(f)

    for idx in range(len(cached_vectors)):
        cached_vectors[idx] = cached_vectors[idx]
    
    return cached_vectors

