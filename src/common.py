from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


Choice = frozenset[str]


@dataclass
class GameState:
    choices: set[Choice]
    answers_remaining: set[Choice]
    turns_taken: int = 0

    def is_terminal(self) -> bool:
        return len(self.answers_remaining) <= 0


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


@dataclass
class GameDatum:
    id: str
    words: set[str]
    answer_key: set[Choice]

    @staticmethod
    def from_json(j: Any):
        id = j['id']
        words = set(j['words'])
        answer_key = set([frozenset(answer) for answer in j['answer_key']])
        return GameDatum(id, words, answer_key)


ClusterScorer = Callable[[list[WordEmbedding]], np.floating]

