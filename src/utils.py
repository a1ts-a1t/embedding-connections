from common import Choice, GameState
from itertools import chain, combinations
from collections import Counter


def get_intersection_size(choice1: Choice, choice2: Choice) -> int:
    return len(choice1.intersection(choice2))


def create_initial_game_state(words: set[str], answer_key: set[Choice]) -> GameState:
    # validate inputs
    if len(words) != 16:
        raise Exception(f"Games must have 16 words, found {len(words)}.")

    if len(answer_key) != 4:
        raise Exception("Games must have 4 answers, found {len(answer_key)}.")

    illegal_words = list(filter(lambda word: word not in words, chain.from_iterable(answer_key)))
    if len(illegal_words) > 0:
        raise Exception(f"Answer key cannot contain words not present in the game, found {illegal_words}.")

    counter = Counter(chain.from_iterable(answer_key))
    repeated_words = list(map(lambda item: item[0], filter(lambda item: item[1] > 1, counter.items())))
    if len(repeated_words) > 0:
        raise Exception(f"Answer key cannot repeat words in different answers, found {repeated_words}.")

    # create choices
    choices = set(map(lambda t: frozenset(t), combinations(words, 4)))
    return GameState(choices, answer_key)


def get_next_game_state(game_state: GameState, choice: Choice) -> GameState:
    intersection_size = min(map(lambda answer: get_intersection_size(choice, answer), game_state.answers_remaining))

    # player picked a correct choice
    if intersection_size == 4:
        removed_choices = filter(lambda c: get_intersection_size(c, choice) > 0, game_state.choices)
        new_choices = game_state.choices.difference(removed_choices)
        new_answers_remaining = game_state.answers_remaining.difference(set([choice]))
        return GameState(new_choices, new_answers_remaining, game_state.turns_remaining)

    # player was one off
    if intersection_size == 3:
        removed_choices = filter(lambda c: get_intersection_size(c, choice) == 2, game_state.choices)
        new_choices = game_state.choices.difference(removed_choices)
        new_choices.remove(choice)
        new_answers_remaining = set(game_state.answers_remaining)
        return GameState(new_choices, new_answers_remaining, game_state.turns_remaining - 1)
        
    # player was completely off
    new_choices = game_state.choices.difference(set([choice]))
    new_answers_remaining = set(game_state.answers_remaining)
    return GameState(new_choices, new_answers_remaining, game_state.turns_remaining - 1)

