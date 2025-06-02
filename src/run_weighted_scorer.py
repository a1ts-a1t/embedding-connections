import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from utils import sanitize_model_name
import json
from common import GameDatum, WordEmbedding, Scorer
from scorers import WeightedInterclusterScorer
from concurrent.futures import ProcessPoolExecutor
from typing import Any
from game import Game
from player import WordEmbeddingPlayer
from tqdm.contrib.concurrent import process_map


def provide_args() -> dict:
    parser = ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--fidelity", default=10)
    parser.add_argument("--output-file-name")
    parser.add_argument("--game-data-file", default="./data/game_data.json")
    return vars(parser.parse_args())


def weights_iterator(maximum: int):
    for i in range(0, maximum + 1):
        for j in range(0, maximum + 1 - i):
            yield (i, j, maximum - i - j)


def load_game_data(file_name: str):
    with open(file_name, mode='r') as file:
        return json.load(file)

def load_word_embeddings_dict(file_name: str) -> dict[str, WordEmbedding]:
    with open(file_name, mode='r') as file:
        data = json.load(file)
        word_embeddings = map(WordEmbedding.from_dict, data)
        return dict(map(lambda word_embedding: (word_embedding.word, word_embedding), word_embeddings))


def run_game(game_json: Any, scorer: Scorer):
    game_datum = GameDatum.from_json(game_json)
    game_word_embeddings = [word_embeddings_dict[word] for word in game_datum.words]
    player = WordEmbeddingPlayer(game_word_embeddings, scorer)
    game = Game(player=player, words=game_datum.words, answer_key=game_datum.answer_key)
    return game.play().turns_taken

if __name__ == "__main__":
    args = provide_args()
    fidelity = int(args['fidelity'])

    game_data: list = load_game_data(args['game_data_file'])
    embedding_file_name = f"./data/{sanitize_model_name(args['model'])}.json"
    word_embeddings_dict = load_word_embeddings_dict(embedding_file_name)

    def evaluate_scorer(scorer: Scorer):
        return np.mean(list(process_map(run_game, game_data, [scorer] * len(game_data), desc="Current run", leave=False)))


    results = []
    for inertia_weight, pairwise_min_weight, pairwise_var_weight in tqdm(weights_iterator(fidelity), total=(fidelity+2)*(fidelity+1)/2, desc="Overall run"):
        scorer = WeightedInterclusterScorer(inertia_weight, pairwise_min_weight, pairwise_var_weight)
        results.append({
            "inertia_weight": inertia_weight / fidelity,
            "pairwise_min_weight": pairwise_min_weight / fidelity,
            "pairwise_var_weight": pairwise_var_weight / fidelity,
            "average_turns_taken": evaluate_scorer(scorer)
        })

    results = {"model": args['model'], "results": results}
    output_file_name = f"./data/results/{sanitize_model_name(args['model'])}_weighted-scorer.json" if args.get('output_file_name') is None else args['output_file_name']

    with open(output_file_name, mode="w") as fp:
        json.dump(results, fp)

