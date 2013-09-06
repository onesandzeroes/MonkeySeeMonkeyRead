"""
Markov models of bigram character probabilities.
Throughout this script, I'm using '^' and '$'
as special start and end symbols.
"""
import doctest
import itertools
import nltk


def return_chars_from_words(word_list):
    """
    Given a raw word list (possibly containing things like
    punctuation, which will be stripped out), return all the
    characters, plus start and end symbols around each word.

    >>> res = list('^hello$^world$')
    >>> list(return_chars_from_words(['hello', 'world'])) == res
    True
    """
    words = (w.lower() for w in word_list if w.isalpha())
    char_gen = itertools.chain(c for w in words for c in ('^' + w + '$'))
    for c in char_gen:
        yield c


def generate_word(model, start_letter=None, max_length=10):
    if start_letter is None:
        word = '^'
    else:
        word = '^' + start_letter
    while (len(word) + 2) < max_length:
        new_char = model.choose_random_word(context=word[-1])
        word += new_char
        if new_char == '$':
            break
    # If we've reached max_length without hitting an end-symbol,
    # make sure it's added on
    else:
        word += '$'
    return word[1:-1]


def test_moby_model():
    moby_raw_words = nltk.corpus.gutenberg.words('melville-moby_dick.txt')
    moby_chars = list(return_chars_from_words(moby_raw_words))

    bigram_model = nltk.model.NgramModel(2, moby_chars, pad_left=False)

if __name__ == '__main__':
    print(doctest.testmod())
