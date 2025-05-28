from common import Choice, GameState
from utils import create_initial_game_state, get_next_game_state
from player import Player


class Game:
    words: set[str]
    answer_key: set[Choice]
    player: Player

    current_state: GameState


    def __init__(self, words: set[str], answer_key: set[Choice], player: Player) -> None:
        self.words = words
        self.answer_key = answer_key
        self.player = player

        initial_state = create_initial_game_state(words, answer_key)

        print(f"Words: {words}")
        print(f"Answer key: {answer_key}")
        self.current_state = initial_state


    def take_turn(self) -> None:
        if self.current_state.is_terminal():
            return

        player_choice = self.player.make_choice(self.current_state.choices)
        print(f"Player choice: {player_choice}")
        new_state = get_next_game_state(self.current_state, player_choice)
        self.current_state = new_state


    def play(self) -> int:
        turn_count = 0
        while not self.current_state.is_terminal():
            if len(self.current_state.answers_remaining) == 0:
                break

            turn_count += 1
            self.take_turn()

        return turn_count

