from common import Choice, Scorer, WordEmbedding
import json
import random
import numpy as np
from abc import ABC, abstractmethod

from utils import create_all_choices_from_words


class Player(ABC):
    @abstractmethod
    def make_choice(self, choices: set[Choice]) -> Choice:
        pass


class HumanPlayer(Player):
    def __init__(self) -> None:
        pass

    def make_choice(self, choices: set[Choice]) -> Choice:
        choices_list = list(choices)
        for idx, choice in enumerate(choices_list):
            print(f"{idx + 1}. {choice}")

        choice_index = int(input("Which choice do you chose: ")) - 1
        return choices_list[choice_index]


class RandomPlayer(Player):
    def __init__(self) -> None:
        pass

    def make_choice(self, choices: set[Choice]) -> Choice:
        return random.choice(list(choices))


class WordEmbeddingPlayer(Player):
    choice_scores: dict[Choice, np.floating]


    def __init__(self, word_embeddings: list[WordEmbedding], scorer: Scorer) -> None:
        word_to_word_embedding = dict([(we.word, we) for we in word_embeddings])
        choices = create_all_choices_from_words(set(word_to_word_embedding.keys()))
        scorer.precompute(word_embeddings)

        def get_choice_score(choice: Choice):
            choices_as_word_embeddings = [word_to_word_embedding[word] for word in choice]
            return scorer.score(choices_as_word_embeddings)

        self.choice_scores = dict(map(lambda choice: (choice, get_choice_score(choice)), choices))


    def make_choice(self, choices: set[Choice]) -> Choice:
        return min(choices, key=lambda choice: self.choice_scores[choice].item())

