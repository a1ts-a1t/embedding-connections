from game import Game
from player import HumanPlayer


player = HumanPlayer()

words = set([
    'a', 'b', 'c', 'd',
    'e', 'f', 'g', 'h',
    'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p',
])

answer_key = set([
    frozenset(['a', 'b', 'c', 'd']),
    frozenset(['e', 'f', 'g', 'h']),
    frozenset(['i', 'j', 'k', 'l']),
    frozenset(['m', 'n', 'o', 'p']),
])

if __name__ == "__main__":
    game = Game(words, answer_key, player)
    game.play()

