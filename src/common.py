from dataclasses import dataclass


Choice = frozenset[str]


@dataclass
class GameState:
    choices: set[Choice]
    answers_remaining: set[Choice]
    turns_remaining: int = 4

    def is_terminal(self) -> bool:
        return self.turns_remaining <= 0


@dataclass
class WordEmbedding:
    word: str
    embedding: list[float]

    @staticmethod
    def from_tuple(t: tuple[str, list[float]]):
        return WordEmbedding(t[0], t[1])

