from __future__ import division
import collections
import doctest
import pdb
import string
import nltk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_bigram_counts(word_list):
    """
    Given a list of words, produce a dictionary mapping first
    letters to the counts of the letters that follow them.

    >>> ab = get_bigram_counts(['ababa'])
    >>> ab['a']['b'] == 2
    True
    >>> ab['b']['a'] == 2
    True
    """
    count_dict = {c: collections.defaultdict(int)
                  for c in string.ascii_lowercase}
    for word in word_list:
        for c1, c2 in zip(word[:-1], word[1:]):
            count_dict[c1][c2] += 1
    return count_dict


def make_bigram_matrix(bigram_counts):
    """
    Given a count_dict returned by get_bigram_counts, make
    a matrix of character1-character2 counts.
    """
    count_df = pd.DataFrame.from_dict(bigram_counts)
    count_df[pd.isnull(count_df)] = 0
    return count_df.as_matrix()


def normalize_bigram_matrix(bigram_counts):
    """
    Normalize the bigram counts within each c1 so they reflect proportions
    for each c1.
    """
    normed_counts = {c: dict() for c in string.ascii_lowercase}
    for c1, c2_counts in bigram_counts.items():
        total = sum(c2_counts.values())
        for c2, count in c2_counts.items():
            normed_counts[c1][c2] = count / total
    return normed_counts

moby_raw_words = nltk.corpus.gutenberg.words('melville-moby_dick.txt')
moby_words = []

for word in moby_raw_words:
    if word.isalpha():
        moby_words.append(word.lower())

moby_bigram_counts = get_bigram_counts(moby_words)
normed_counts = normalize_bigram_matrix(moby_bigram_counts)
bigram_matrix = make_bigram_matrix(normed_counts)

# To plot:
plt.pcolor(np.log(bigram_matrix + 1), cmap=plt.cm.Blues)
plt.xlim((0, 26))
plt.ylim((0, 26))
plt.xticks(np.arange(0.5, 26, 1), string.ascii_lowercase)
plt.yticks(np.arange(0.5, 26, 1), string.ascii_lowercase)

if __name__ == '__main__':
    print(doctest.testmod())

