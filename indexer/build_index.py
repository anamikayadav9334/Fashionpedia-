"""
Builds a FAISS vector index from image embeddings.
Uses IndexFlatIP (inner product) for exact cosine-similarity search,
since embeddings are L2-normalized beforehand.
"""
import faiss
import numpy as np


def build_faiss_index(embeddings):
    """
    embeddings: np.ndarray of shape (N, D), L2-normalized, dtype float32
    Returns a FAISS index with all vectors added.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    return index


def save_index(index, path):
    faiss.write_index(index, path)


def load_index(path):
    return faiss.read_index(path)
