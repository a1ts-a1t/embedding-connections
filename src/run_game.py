from typing import Any
from common import GameDatum, WordEmbedding
from game import Game
from scorers import SCORER_MAP
from player import WordEmbeddingPlayer
import json
from argparse import ArgumentParser, BooleanOptionalAction
from tqdm.contrib.concurrent import process_map
from utils import sanitize_model_name


def provide_args() -> dict:
    parser = ArgumentParser()
    parser.add_argument("--scorer", choices=SCORER_MAP.keys(), default="mean_projective_centroid_distance") # scorer
    parser.add_argument("--model", default="all-MiniLM-L6-v2") # model
    parser.add_argument("--game-data-file", default="./data/game_data.json")
    parser.add_argument("--output-file-name")
    return vars(parser.parse_args())


def load_game_data(file_name: str):
    with open(file_name, mode='r') as file:
        return json.load(file)


def dump_results(file_name: str, results: Any):
    with open(file_name, mode='w') as file:
        return json.dump(results, file)


def load_word_embeddings_dict(file_name: str) -> dict[str, WordEmbedding]:
    with open(file_name, mode='r') as file:
        data = json.load(file)
        word_embeddings = map(WordEmbedding.from_dict, data)
        return dict(map(lambda word_embedding: (word_embedding.word, word_embedding), word_embeddings))


if __name__ == "__main__":
    args = provide_args()
    scorer = SCORER_MAP.get(args['scorer'])
    
    if scorer is None:
        raise Exception(f"Invalid scorer[{args['scorer']}]")

    game_data: list = load_game_data(args['game_data_file'])
    embedding_file_name = f"./data/{sanitize_model_name(args['model'])}.json"
    word_embeddings_dict = load_word_embeddings_dict(embedding_file_name)

    def run(game_json: Any):
        game_datum = GameDatum.from_json(game_json)
        game_word_embeddings = [word_embeddings_dict[word] for word in game_datum.words]

        player = WordEmbeddingPlayer(game_word_embeddings, scorer=scorer()) # type: ignore
        game = Game(player=player, words=game_datum.words, answer_key=game_datum.answer_key)

        final_game_state = game.play()
        turns_taken = final_game_state.turns_taken

        return { "game_id": game_datum.id, "turns_taken": turns_taken }

    game_results = process_map(run, game_data)
    results = {"model": args['model'], "scorer": args['scorer'], "game_results": game_results}
    output_file_name = f"./data/results/{sanitize_model_name(args['model'])}_{args['scorer']}.json" if args.get('output_file_name') is None else args['output_file_name']
    dump_results(output_file_name, results)

