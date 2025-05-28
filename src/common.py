from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Choice = frozenset[str]


@dataclass
class GameState:
    choices: set[Choice]
    answers_remaining: set[Choice]
    turns_remaining: int = 1820 # 16 choose 4

    def is_terminal(self) -> bool:
        return self.turns_remaining <= 0


@dataclass
class WordEmbedding:
    word: str
    embedding: list[float]

    @staticmethod
    def from_tuple(t: tuple[str, list[float]]):
        return WordEmbedding(t[0], t[1])

    @staticmethod
    def from_dict(d: dict):
        return WordEmbedding(d['word'], d['embedding'])


Metric = Callable[[NDArray, NDArray], float]
ClusterScorer = Callable[[list[WordEmbedding], Metric], np.floating]

