from __future__ import division
import collections
import math
import pdb
import nltk

class InterpolatedModel:
    """
    A model based on a trigram Markov model, but which uses bigram
    and unigram probabilities to smooth the estimates and
    fill in values for unseen trigrams.
    """

    def __init__(self, trigram_weight, bigram_weight, unigram_weight):
        """
        The model requires separate weights for the trigram, bigram and
        unigram probabilities. These weights must be 0 <= w <= 1, and
        must sum to 1.
        """
        self.trigram_counts = collections.defaultdict(int)
        self.bigram_counts = collections.defaultdict(int)
        self.unigram_counts = collections.defaultdict(int)
        self.total_chars = 0
        self.trigram_weight = trigram_weight
        self.bigram_weight = bigram_weight
        self.unigram_weight = unigram_weight

    def train(self, text):
        """
        Given a sample of real text as a single string, splits
        the text into words (ignoring words containing puncuation or
        digits), pads each word with start and end symbols,
        and updates the model's unigram, bigram and trigram counts.
        """
        self.count_unigrams(text)
        self.count_bigrams(text)
        self.count_trigrams(text)

    def count_unigrams(self, text):
        words = (w.lower() for w in text.split() if w.isalpha())
        for word in words:
            for char in ('*' + word + '$'):
                self.unigram_counts[char] += 1
                self.total_chars += 1

    def count_bigrams(self, text):
        words = (w.lower() for w in text.split() if w.isalpha())
        for word in words:
            padded_word = '**' + word + '$'
            for ind in range(len(padded_word) - 1):
                self.bigram_counts[padded_word[ind:(ind + 2)]] += 1

    def count_trigrams(self, text):
        words = (w.lower() for w in text.split() if w.isalpha())
        for word in words:
            padded_word = '**' + word + '$'
            for ind in range(len(padded_word) - 2):
                self.trigram_counts[padded_word[ind:(ind + 3)]] += 1

    def char_probabilities(self, word):
        """
        For each character in `word`, calculate the interpolated probability
        for that character, returning a list of probabilities the same 
        length as `word`.
        """
        # Pad with the end-symbol '$'
        word = word + '$'
        probs = []
        for c_index in range(len(word)):
            char = word[c_index]
            if c_index == 0:
                trigram = '**' + char
                bigram = '*' + char
            elif c_index == 1:
                trigram = '*' + word[0] + char
            else:
                trigram = word[(c_index - 2):(c_index + 1)]
                bigram = word[(c_index -1):(c_index + 1)]
            trigram_prob = self.get_trigram_prob(trigram)
            bigram_prob = self.get_bigram_prob(bigram)
            unigram_prob = (self.unigram_counts[char] /
                            self.total_chars) * self.unigram_weight
            char_prob = trigram_prob + bigram_prob + unigram_prob
            probs.append((char, char_prob))
        return probs

    def get_trigram_prob(self, trigram):
        trigram_context = trigram[:-1]
        trigram_count = self.trigram_counts.get(trigram, 0)
        context_count = self.bigram_counts.get(trigram_context, None)
        if context_count is None:
            return 0
        return (trigram_count / context_count) * self.trigram_weight

    def get_bigram_prob(self, bigram):
        bigram_context = bigram[:-1]
        bigram_count = self.bigram_counts.get(bigram, 0)
        context_count = self.unigram_counts[bigram_context]
        return (bigram_count / context_count) * self.bigram_weight


    def word_probability(self, word):
        char_probs = self.char_probabilities(word)
        total_log_prob = sum(math.log(p) for c, p in char_probs)
        return total_log_prob

if __name__ == '__main__':
    m = InterpolatedModel(0.75, 0.2, 0.05)
    m.train(nltk.corpus.gutenberg.raw('melville-moby_dick.txt'))
    test_words = ['wasp', 'dime', 'hood', 'leek', 'open']
    test_nonwords = ['annk', 'ormp', 'palk', 'raln', 'scip']

    print("Words:")
    for w in test_words:
        p = m.word_probability(w)
        print(w + ': ' + str(p))

    print("Nonwords:")
    for w in test_nonwords:
        p = m.word_probability(w)
        print(w + ': ' + str(p))