from argparse import ArgumentParser
from common import GameDatum
from game import Game
import json

from player import RandomPlayer

def provide_args() -> dict:
    parser = ArgumentParser()
    parser.add_argument("--game-data-file", default="./data/game_data.json")
    return vars(parser.parse_args())


def load_game_data(file_name: str):
    with open(file_name, mode='r') as file:
        return json.load(file)

if __name__ == "__main__":
    args = provide_args()
    game_data = load_game_data(args['game_data_file'])
    turn_counts = []
    for game_json in game_data:
        game_datum = GameDatum.from_json(game_json)
        player = RandomPlayer()
        game = Game(player=player, words=game_datum.words, answer_key=game_datum.answer_key)
        final_game_state = game.play()
        turns_taken = final_game_state.turns_taken
        print(f"Random selection took {turns_taken} turns to complete game {game_datum.id}")
        turn_counts.append(turns_taken)

    print(f"Randomness took {sum(turn_counts) / len(turn_counts)} turns, on average")

