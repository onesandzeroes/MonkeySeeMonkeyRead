from __future__ import division
import copy
import doctest
import itertools
import random
import matplotlib.pyplot as plt
import pandas as pd


def score_word(word, weights):
    """
    Calculate the score for the current word, using the provided
    weights.

    >>> w = {0: {'a': 1}, 1: {'r': -1}, 2: {'t': 0}, 3: {'y': 1}}
    >>> score_word('arty', w)
    1
    """
    score = sum(weights[index][char] for index, char in enumerate(word))
    return score


def train_model(weights, training_words, training_nonwords,
                n_iterations, learning_rate=0.05):
    """
    Train the word recognition model by calculating the current fitness
    score, randomly changing ~5% of the weights, and keeping the new
    weights if the fitness score is improved.
    """
    for iteration_num in range(n_iterations):
        current_score = fitness_score(training_words, training_nonwords,
                                      weights)
        # Create a new set of weights by randomly perturbing the existing ones:
        new_weights = copy.deepcopy(weights)
        all_pos_char_pairs = itertools.product(
            [0, 1, 2, 3],
            list('abcdefghijklmnopqrstuvwxyz')
        )
        for n, c in all_pos_char_pairs:
            random_roll = random.random()
            if random_roll <= learning_rate:
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


def create_random_weights():
    weights = {}
    for n in [0, 1, 2, 3]:
        weights[n] = {}
        for c in 'abcdefghijklmnopqrstuvwxyz':
            weights[n][c] = random.choice([0, 1, -1])
    return weights


def full_model_run(train_words, train_nonwords, test_words, test_nonwords,
                   n_iterations):
    weights = create_random_weights()
    model_results = {'iteration': [], 'words_correct': [],
                     'nonwords_correct': [], 'unclassified': []}
    for i in range(n_iterations):
        weights = train_model(
            weights, train_words, train_nonwords, n_iterations=1)
        words_correct = sum(score_word(w, weights) > 0 for w in test_words)
        words_correct_percent = words_correct / len(test_words)
        nonwords_correct = sum(score_word(w, weights) < 0
                               for w in test_nonwords)
        nonwords_correct_percent = nonwords_correct / len(test_nonwords)
        unclassified = (
            sum(score_word(w, weights) == 0 for w in test_words) +
            sum(score_word(w, weights) == 0 for w in test_nonwords)
        )
        model_results['iteration'].append(i)
        model_results['words_correct'].append(words_correct_percent)
        model_results['nonwords_correct'].append(nonwords_correct_percent)
        model_results['unclassified'].append(unclassified)

    results_df = pd.DataFrame(model_results)
    return results_df


def run_model(output_filename):
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

    sim_results = full_model_run(train_words, train_nonwords, test_words,
                                 test_nonwords, n_iterations=2000)
    sim_results.set_index('iteration')
    sim_results.to_csv(output_filename)
    sim_results[['words_correct', 'nonwords_correct']].plot()
    plt.savefig('figures/bain_replication.png')

if __name__ == '__main__':
    doctest.testmod()
    run_model('bain_model_replication.csv')
