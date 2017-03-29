__author__ = 'wenjusun'

import re

compiled_patterns = re.compile('[0-9]{1,}')
class StopWordsFilter():
    stop_words = []

    def __init__(self):
        with open(r'stop_words.txt') as f:
            for line in f:
                self.stop_words.append(line.strip())

    def is_stopword(self, word):

        is_digits=lambda x: compiled_patterns.match(x)
        return len(word) <2 or word in self.stop_words or is_digits(word)

def test_stopwords():
    print stopWordsFilter.is_stopword('of')
    print stopWordsFilter.is_stopword('baby')
    print stopWordsFilter.is_stopword('1')
    print stopWordsFilter.is_stopword('12')

stopWordsFilter = StopWordsFilter()

test_stopwords()