from itertools import combinations
from numpy.typing import NDArray
from common import ClusterScorer, WordEmbedding
import numpy as np
from numpy.linalg import norm


def normalize(embedding: NDArray) -> NDArray:
    embedding_norm = norm(embedding)

    # prevents divide by zero errors. should only happen for the 0 vector which cannot be noramlized
    if embedding_norm == 0:
        return embedding

    return embedding / embedding_norm


def cosine_metric(embedding1: NDArray, embedding2: NDArray) -> np.floating:
    return 1 - np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))


def l2_metric(embedding1: NDArray, embedding2: NDArray) -> np.floating:
    diff = embedding1 - embedding2
    return np.sum(np.square(diff))


def mean_centroid_distance_scorer(word_embeddings: list[WordEmbedding]) -> np.floating:
    embeddings = map(lambda word_embedding: np.array(word_embedding.embedding), word_embeddings)
    centroid = np.sum(list(embeddings)) / len(word_embeddings)
    distances = map(lambda embedding: l2_metric(embedding, centroid), embeddings)
    return np.mean(list(distances))


# ref: https://skeptric.com/projective-centroid/
def mean_projective_centroid_distance_scorer(word_embeddings: list[WordEmbedding]) -> np.floating:
    normalized_embeddings = list(map(lambda word_embedding: normalize(np.array(word_embedding.embedding)), word_embeddings))
    centroid = normalize(np.sum(normalized_embeddings, axis=0) / len(word_embeddings))

    distances = map(lambda embedding: cosine_metric(embedding, centroid), normalized_embeddings)
    return np.mean(list(distances))


def max_pairwise_l2_distance_scorer(word_embeddings: list[WordEmbedding]) -> np.floating:
    embeddings = map(lambda word_embedding: np.array(word_embedding.embedding), word_embeddings)
    embedding_pairs = combinations(embeddings, 2)
    distances = map(lambda pair: l2_metric(pair[0], pair[1]), embedding_pairs)
    return np.max(list(distances))


def mean_pairwise_l2_distance_scorer(word_embeddings: list[WordEmbedding]) -> np.floating:
    embeddings = map(lambda word_embedding: np.array(word_embedding.embedding), word_embeddings)
    embedding_pairs = combinations(embeddings, 2)
    distances = map(lambda pair: l2_metric(pair[0], pair[1]), embedding_pairs)
    return np.mean(list(distances))


def max_pairwise_cosine_distance_scorer(word_embeddings: list[WordEmbedding]) -> np.floating:
    embeddings = map(lambda word_embedding: np.array(word_embedding.embedding), word_embeddings)
    embedding_pairs = combinations(embeddings, 2)
    distances = map(lambda pair: cosine_metric(pair[0], pair[1]), embedding_pairs)
    return np.max(list(distances))


def mean_pairwise_cosine_distance_scorer(word_embeddings: list[WordEmbedding]) -> np.floating:
    embeddings = map(lambda word_embedding: np.array(word_embedding.embedding), word_embeddings)
    embedding_pairs = combinations(embeddings, 2)
    distances = map(lambda pair: cosine_metric(pair[0], pair[1]), embedding_pairs)
    return np.mean(list(distances))


SCORER_MAP: dict[str, ClusterScorer] = {
    "mean_centroid_distance": mean_centroid_distance_scorer,
    "mean_projective_centroid_distance": mean_projective_centroid_distance_scorer,
    "max_pairwise_cosine_distance": max_pairwise_cosine_distance_scorer,
    "mean_pairwise_cosine_distance": mean_pairwise_cosine_distance_scorer,
    "max_pairwise_l2_distance": max_pairwise_l2_distance_scorer,
    "mean_pairwise_l2_distance": mean_pairwise_l2_distance_scorer,
}

