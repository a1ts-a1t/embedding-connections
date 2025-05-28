from common import Choice, GameState
from utils import create_initial_game_state, get_next_game_state
from player import Player
from typing import Set


class Game:
    words: Set[str]
    answer_key: Set[Choice]
    player: Player

    current_state: GameState


    def __init__(self, words: Set[str], answer_key: Set[Choice], player: Player) -> None:
        self.words = words
        self.answer_key = answer_key
        self.player = player

        initial_state = create_initial_game_state(words, answer_key)
        self.current_state = initial_state


    def reset(self) -> None:
        self.current_state = create_initial_game_state(self.words, self.answer_key)


    def print_current_state(self) -> None:
        print(f"Current state\n{self.current_state}")


    def take_turn(self) -> None:
        if self.current_state.is_terminal():
            return

        player_choice = self.player.make_choice(self.current_state.choices)
        new_state = get_next_game_state(self.current_state, player_choice)
        self.current_state = new_state


    def play(self) -> bool:
        while not self.current_state.is_terminal():
            self.print_current_state()
            self.take_turn()

        return len(self.current_state.answers_remaining) == 0

