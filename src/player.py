from common import Choice
from abc import ABC, abstractmethod


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
