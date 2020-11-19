import nltk
from nltk.stem import SnowballStemmer
import numpy as np
#nltk.download('punkt') # Requires only one run

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    stemmer = SnowballStemmer(language="english")
    return stemmer.stem(word.lower())

def bag_of_words(tok_sentence, all_words):
    tok_sentence = [stem(word) for word in tok_sentence]
    bag_of_words = np.zeros(len(all_words), dtype=np.float32)
    for index, word in enumerate(all_words):
        if word in tok_sentence:
            bag_of_words[index] = 1.0
    return bag_of_words

