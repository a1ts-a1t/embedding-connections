from numpy.typing import NDArray
from common import WordEmbedding, Metric
import numpy as np
from numpy.linalg import norm


def normalize(embedding: NDArray) -> NDArray:
    embedding_norm = norm(embedding)

    # prevents divide by zero errors. should only happen for the 0 vector which cannot be noramlized
    if embedding_norm == 0:
        return embedding

    return embedding / embedding_norm


def cosine_metric(embedding1: NDArray, embedding2: NDArray) -> float:
    return 1 - np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))


def l2_metric(embedding1: NDArray, embedding2: NDArray) -> float:
    diff = embedding1 - embedding2
    return np.sum(np.square(diff))


def mean_centroid_distance_scorer(word_embeddings: list[WordEmbedding], metric: Metric) -> np.floating:
    embeddings = map(lambda word_embedding: np.array(word_embedding.embedding), word_embeddings)
    centroid = np.sum(list(embeddings)) / len(word_embeddings)
    distances = map(lambda embedding: metric(embedding, centroid), embeddings)
    return np.mean(list(distances))


# ref: https://skeptric.com/projective-centroid/
def mean_projective_centroid_distance_scorer(word_embeddings: list[WordEmbedding], metric: Metric) -> np.floating:
    normalized_embeddings = list(map(lambda word_embedding: normalize(np.array(word_embedding.embedding)), word_embeddings))
    centroid = normalize(np.sum(normalized_embeddings, axis=0) / len(word_embeddings))

    distances = map(lambda embedding: metric(embedding, centroid), normalized_embeddings)
    return np.mean(list(distances))


METRICS_MAP = { "cosine": cosine_metric, "l2": l2_metric }
SCORER_MAP = { "mean_centroid_distance": mean_centroid_distance_scorer, "mean_projective_centroid_distance": mean_projective_centroid_distance_scorer }

