from collections.abc import Callable
from itertools import combinations, combinations_with_replacement
from numpy.typing import NDArray
from common import Scorer, WordEmbedding
import numpy as np
from numpy.linalg import norm

Metric = Callable[[NDArray, NDArray], np.floating]
Aggregator = Callable[[list[np.floating]], np.floating]


def normalize(embedding: NDArray) -> NDArray:
    embedding_norm = norm(embedding)

    # prevents divide by zero errors. should only happen for the 0 vector which cannot be noramlized
    if embedding_norm == 0:
        return embedding

    return embedding / embedding_norm


def cosine_similarity(embedding1: NDArray, embedding2: NDArray) -> np.floating:
    return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))


def cosine_metric(embedding1: NDArray, embedding2: NDArray) -> np.floating:
    return 1 - cosine_similarity(embedding1, embedding2)


def l2_metric(embedding1: NDArray, embedding2: NDArray) -> np.floating:
    diff = embedding1 - embedding2
    return np.sum(np.square(diff))


class CentroidDistanceScorer(Scorer):
    aggregator: Aggregator
    def __init__(self, aggregator: Aggregator):
        self.aggregator = aggregator

    def score(self, word_embeddings: list[WordEmbedding]) -> np.floating:
        embeddings = list(map(lambda word_embedding: np.array(word_embedding.embedding), word_embeddings))
        centroid = np.sum(embeddings, axis=0) / len(word_embeddings)
        distances = list(map(lambda embedding: l2_metric(embedding, centroid), embeddings))
        return self.aggregator(list(distances))


# ref: https://skeptric.com/projective-centroid/
class ProjectiveCentroidDistanceScorer(Scorer):
    aggregator: Aggregator
    def __init__(self, aggregator: Aggregator):
        self.aggregator = aggregator

    def score(self, word_embeddings: list[WordEmbedding]) -> np.floating:
        normalized_embeddings = list(map(lambda word_embedding: normalize(np.array(word_embedding.embedding)), word_embeddings))
        centroid = normalize(np.sum(normalized_embeddings, axis=0) / len(word_embeddings))

        distances = map(lambda embedding: cosine_metric(embedding, centroid), normalized_embeddings)
        return self.aggregator(list(distances))


class PairwiseDistanceScorer(Scorer):
    metric: Metric
    aggregator: Aggregator

    def __init__(self, metric: Metric, aggregator: Aggregator):
        self.metric = metric
        self.aggregator = aggregator
        pass

    def score(self, word_embeddings: list[WordEmbedding]) -> np.floating:
        embeddings = map(lambda word_embedding: np.array(word_embedding.embedding), word_embeddings)
        embedding_pairs = combinations(embeddings, 2)
        distances = map(lambda pair: self.metric(pair[0], pair[1]), embedding_pairs)
        return self.aggregator(list(distances))


# https://en.wikipedia.org/wiki/Silhouette_(clustering)
class SilhouetteCoefficientScorer(Scorer):
    metric: Callable[[NDArray, NDArray], np.floating]
    pairwise_distance_map: dict[frozenset[str], np.floating] = {}
    all_words: set[str] = set()

    def __init__(self, metric: Callable[[NDArray, NDArray], np.floating]):
        self.metric = metric

    # compute all pairwise distances between embeddings
    def precompute(self, word_embeddings: list[WordEmbedding]) -> None:
        self.all_words = set([word_embedding.word for word_embedding in word_embeddings])
        for combination in combinations_with_replacement(word_embeddings, 2):
            word1 = combination[0].word
            word2 = combination[1].word
            if word1 == word2:
                self.pairwise_distance_map[frozenset([word1, word2])] = 0 # type: ignore
                continue

            distance = self.metric(np.array(combination[0].embedding), np.array(combination[1].embedding))
            self.pairwise_distance_map[frozenset([word1, word2])] = distance

    def get_distance(self, word1: str, word2: str) -> np.floating:
        key = frozenset([word1, word2])
        return self.pairwise_distance_map[key]

    def score(self, word_embeddings: list[WordEmbedding]) -> np.floating:
        # intracluster
        def a(word_embedding: WordEmbedding) -> np.floating:
            distances = [self.get_distance(word_embedding.word, cluster_embedding.word) for cluster_embedding in word_embeddings]
            return np.sum(distances) / 3

        # intercluster
        non_cluster_words = self.all_words.difference(set([word_embedding.word for word_embedding in word_embeddings]))
        def b(word_embedding: WordEmbedding) -> np.floating:
            distances = [self.get_distance(word_embedding.word, non_cluster_word) for non_cluster_word in non_cluster_words]
            return np.sum(distances) / 12

        def s(word_embedding: WordEmbedding) -> np.floating:
            a_val = a(word_embedding)
            b_val = b(word_embedding)
            return 1 - a_val / b_val if a_val < b_val else b_val / a_val - 1

        return -np.mean([s(word_embedding) for word_embedding in word_embeddings])


# https://arxiv.org/pdf/2412.01621
class LopezMcDonaldEmamiScorer(Scorer):
    word_embedding_map: dict[str, NDArray] = {}

    def __init__(self):
        pass

    def precompute(self, word_embeddings: list[WordEmbedding]) -> None:
        self.word_embedding_map = dict([(we.word, np.array(we.embedding)) for we in word_embeddings])

    def partition_cluster(self, word_embeddings: list[WordEmbedding]) -> tuple[list[NDArray], list[NDArray]]:
        cluster_words = set([we.word for we in word_embeddings])
        cluster_embeddings = []
        non_cluster_embeddings = []

        for word, embedding in self.word_embedding_map.items():
            if word in cluster_words:
                cluster_embeddings.append(embedding)
            else:
                non_cluster_embeddings.append(embedding)

        return (cluster_embeddings, non_cluster_embeddings)

    def score(self, word_embeddings: list[WordEmbedding]) -> np.floating:
        # group similarity score
        cluster_embeddings, non_cluster_embeddings = self.partition_cluster(word_embeddings)
        centroid = np.sum(cluster_embeddings, axis=0) / len(word_embeddings)
        i = -np.sum([l2_metric(embedding, centroid) for embedding in cluster_embeddings])

        pairwise_similarities = [cosine_similarity(pair[0], pair[1]) for pair in combinations(cluster_embeddings, 2)]
        s = np.min(pairwise_similarities)
        v = np.mean(pairwise_similarities) / (1 + np.var(pairwise_similarities))

        g = 0.4 * i + 0.3 * s + 0.3 * v

        # penalty score
        p = np.sum([cosine_similarity(centroid, r) for r in non_cluster_embeddings]) / 12

        return p - g # signs swapped from paper to reward low scores



SCORER_MAP: dict[str, Callable[[], Scorer]] = {
    "mean_centroid_distance": lambda: CentroidDistanceScorer(np.mean),
    "mean_projective_centroid_distance": lambda: ProjectiveCentroidDistanceScorer(np.mean),
    "max_centroid_distance": lambda: CentroidDistanceScorer(np.max),
    "max_projective_centroid_distance": lambda: ProjectiveCentroidDistanceScorer(np.max),
    "max_pairwise_cosine_distance": lambda: PairwiseDistanceScorer(cosine_metric, np.max),
    "mean_pairwise_cosine_distance": lambda: PairwiseDistanceScorer(cosine_metric, np.mean),
    "max_pairwise_l2_distance": lambda: PairwiseDistanceScorer(l2_metric, np.max),
    "mean_pairwise_l2_distance": lambda: PairwiseDistanceScorer(l2_metric, np.mean),
    "cosine_silhouette_coefficient": lambda: SilhouetteCoefficientScorer(cosine_metric),
    "l2_silhouette_coefficient": lambda: SilhouetteCoefficientScorer(l2_metric),
    "lopez_mcdonald_emami": lambda: LopezMcDonaldEmamiScorer(),
}

