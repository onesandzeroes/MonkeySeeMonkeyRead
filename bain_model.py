from __future__ import division
import collections
import copy
import itertools
import random


def score_word(word, weights):
    score = sum(weights[index][char] for index, char in enumerate(word))
    return score


def train_model(weights, training_words, training_nonwords,
                n_iterations, learning_rate=5):
    for iteration_num in range(n_iterations):
        current_score = fitness_score(training_words, training_nonwords,
                                      weights)
        # Create a new set of weights by randomly perturbing the existing ones:
        new_weights = copy.deepcopy(weights)
        all_pos_char_pairs = list(itertools.product(
            [0, 1, 2, 3],
            list('abcdefghijklmnopqrstuvwxyz')))
        weights_to_change = random.sample(
            all_pos_char_pairs,
            5)
        for n, c in weights_to_change:
            new_weights[n][c] = random.choice([0, 1, -1])
        new_score = fitness_score(
            training_words, training_nonwords, new_weights)
        if new_score > current_score:
            print("Fitness score: {f}".format(f=new_score))
            weights = new_weights
    return weights


def sign(n):
    if n > 0:
        return 1
    if n == 0:
        return 0
    if n < 0:
        return -1


def fitness_score(words, nonwords, weights):
    word_scores = [score_word(w, weights) for w in words]
    word_signs = [sign(s) for s in word_scores]
    word_total = sum(word_signs) / len(words)
    nonword_scores = [score_word(w, weights) for w in nonwords]
    nonword_signs = [sign(s) for s in nonword_scores]
    nonword_total = sum(nonword_signs) / len(nonwords)
    nonzero_weights = sum(1 for pos in weights
                          for char in weights[pos]
                          if weights[pos][char] != 0)
    score = (1000 * (word_total - nonword_total)) / (300 + nonzero_weights)
    return score


train_words = []
test_words = []
with open('dan_learned_words.txt') as words_file:
    for n, line in enumerate(words_file):
        word = line.strip()
        if n % 2 == 0:
            train_words.append(word)
        else:
            test_words.append(word)

train_nonwords = []
test_nonwords = []
with open('dan_nonwords.txt') as nonwords_file:
    for n, line in enumerate(nonwords_file):
        nonword = line.strip()
        if n % 2 == 0:
            train_nonwords.append(nonword)
        else:
            test_nonwords.append(nonword)

print(len(train_words))

weights = {}

for n in [0, 1, 2, 3]:
    weights[n] = {}
    for c in 'abcdefghijklmnopqrstuvwxyz':
        weights[n][c] = random.choice([0, 1, -1])

weights = train_model(weights, train_words, train_nonwords, n_iterations=2000)
words_correct = sum(score_word(w, weights) > 0 for w in test_words)
words_correct_percent = words_correct / len(test_words)
nonwords_correct = sum(score_word(w, weights) < 0 for w in test_nonwords)
nonwords_correct_percent = nonwords_correct / len(test_nonwords)
unclassified = (sum(score_word(w, weights) == 0 for w in test_words) +
    sum(score_word(w, weights) == 0 for w in test_nonwords))