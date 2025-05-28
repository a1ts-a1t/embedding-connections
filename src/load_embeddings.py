from dataclasses import asdict
from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser
import json

from common import WordEmbedding


def provide_args() -> dict:
    parser = ArgumentParser()
    parser.add_argument("-f", default="./data/all_words.json") # filename
    parser.add_argument("-m", default="sentence-transformers/all-MiniLM-L6-v2") # model name
    parser.add_argument("-o", default="./data/all-MiniLM-L6-v2.json") # output file name
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
    file_name = args['f']
    model_name = args['m']
    output_file_name = args['o']

    words = read_words_from_file(file_name)
    model = SentenceTransformer(model_name)

    embeddings: list[list[float]] = model.encode(words).tolist()
    word_embeddings = list(map(WordEmbedding.from_tuple, zip(words, embeddings)))

    write_word_embeddings_to_file(word_embeddings, output_file_name)

