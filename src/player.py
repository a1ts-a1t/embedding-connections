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


def _load_word_embeddings_dict(file_name: str) -> dict[str, WordEmbedding]:
    with open(file_name, mode='r') as file:
        data = json.load(file)
        word_embeddings = map(WordEmbedding.from_dict, data)
        return dict(map(lambda word_embedding: (word_embedding.word, word_embedding), word_embeddings))


class WordEmbeddingPlayer(Player):
    choice_scores: dict[Choice, np.floating]


    def __init__(self, file_name: str, words: set[str], scorer: Scorer) -> None:
        choices = create_all_choices_from_words(words)
        word_embeddings = _load_word_embeddings_dict(file_name)
        word_embeddings = dict([(word, word_embeddings[word]) for word in words])

        scorer.precompute(list(word_embeddings.values()))

        def get_choice_score(choice: Choice):
            choices_as_word_embeddings = map(lambda word: word_embeddings[word], choice)
            return scorer.score(list(choices_as_word_embeddings))

        self.choice_scores = dict(map(lambda choice: (choice, get_choice_score(choice)), choices))


    def make_choice(self, choices: set[Choice]) -> Choice:
        return min(choices, key=lambda choice: self.choice_scores[choice].item())

