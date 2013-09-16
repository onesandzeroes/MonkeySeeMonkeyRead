import random

with open('dan_learned_words.txt') as word_file:
    words = [line.strip() for line in word_file]

eg_words = random.sample(words, 10)

with open('dan_nonwords.txt') as non_file:
    nonwords = [line.strip() for line in non_file]

eg_nons = random.sample(nonwords, 10)

print(eg_words)
print(eg_nons)
