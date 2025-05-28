from dataclasses import asdict
from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser
import json

from common import WordEmbedding


def provide_args() -> dict:
    parser = ArgumentParser()
    parser.add_argument("--input-file-name", default="./data/all_words.json") # filename
    parser.add_argument("--model", default="all-MiniLM-L6-v2") # model name
    parser.add_argument("--output-file-name") # output file name
    return vars(parser.parse_args())


def read_words_from_file(file_name: str) -> list[str]:
    with open(file_name, mode='r') as file:
        return list(json.loads(file.read()))


def write_word_embeddings_to_file(word_embeddings: list[WordEmbedding], file_name: str) -> None:
    word_embedding_dicts = list(map(asdict, word_embeddings))
    with open(file_name, mode='w') as file:
        json.dump(word_embedding_dicts, file)


if __name__ == "__main__":
    args = provide_args()
    input_file_name = args['input_file_name']
    model_name = args['model']
    output_file_name = f"./data/{model_name}.json" if args.get('output_file_name') is None else str(args.get('output_file_name'))

    words = read_words_from_file(input_file_name)
    model = SentenceTransformer(model_name)

    embeddings: list[list[float]] = model.encode(words).tolist()
    word_embeddings = list(map(WordEmbedding.from_tuple, zip(words, embeddings)))

    write_word_embeddings_to_file(word_embeddings, output_file_name)

