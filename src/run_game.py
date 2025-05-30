from typing import Any
from common import GameDatum
from game import Game
from scorers import SCORER_MAP
from player import WordEmbeddingPlayer
import json
from argparse import ArgumentParser, BooleanOptionalAction
from tqdm import tqdm
from utils import sanitize_model_name


def provide_args() -> dict:
    parser = ArgumentParser()
    parser.add_argument("--scorer", choices=SCORER_MAP.keys(), default="mean_projective_centroid_distance") # scorer
    parser.add_argument("--model", default="all-MiniLM-L6-v2") # model
    parser.add_argument("--game-data-file", default="./data/game_data.json")
    parser.add_argument("--output-file-name")
    parser.add_argument("--verbose", action=BooleanOptionalAction)
    return vars(parser.parse_args())


def load_game_data(file_name: str):
    with open(file_name, mode='r') as file:
        return json.load(file)


def dump_results(file_name: str, results: Any):
    with open(file_name, mode='w') as file:
        return json.dump(results, file)


if __name__ == "__main__":
    args = provide_args()
    scorer = SCORER_MAP.get(args['scorer'])
    
    if scorer is None:
        raise Exception(f"Invalid scorer[{args['scorer']}]")

    game_data = load_game_data(args['game_data_file'])
    embedding_file_name = f"./data/{sanitize_model_name(args['model'])}.json"
    verbose = args.get("verbose")

    game_results = []
    for game_json in tqdm(game_data):
        game_datum = GameDatum.from_json(game_json)
        player = WordEmbeddingPlayer(embedding_file_name, game_datum.words, scorer=scorer())
        game = Game(player=player, words=game_datum.words, answer_key=game_datum.answer_key)
        final_game_state = game.play()
        turns_taken = final_game_state.turns_taken

        if verbose:
            print(f"Model took {turns_taken} turns to complete game {game_datum.id}")

        game_results.append({"game_id": game_datum.id, "turns_taken": turns_taken})

    results = {"model": args['model'], "scorer": args['scorer'], "game_results": game_results}
    output_file_name = f"./data/results/{sanitize_model_name(args['model'])}_{args['scorer']}.json" if args.get('output_file_name') is None else args['output_file_name']
    dump_results(output_file_name, results)

