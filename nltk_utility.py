import nltk
from nltk.stem import SnowballStemmer
import numpy as np
import re
#nltk.download('punkt') # Requires only one run - comment out after first run

# Suite of methods used for natural language processing of English sentences

# Tokenize sentence - break the sentence up into smaller units, in this case, individual words
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Stem words - reduce word in an attempt to obtain the root of the word
# Done in an attempt to simplify words to their most basic form and then derive meaning from (or find patterns using) this root of the word
def stem(word):
    # Utilize nltk's SnowballStemmer
    stemmer = SnowballStemmer(language="english")
    # The below is done due to English-speakers tendency to greet with words such as 'heyyy' and 'hiii'
    # SnowballStemmer does not reduce these words down to 'hey' and 'hi', and thus the below process is carried out
    if 'hey' in word.lower() or 'hi' in word.lower():
        return 'hi'
    return stemmer.stem(word.lower())

# Bag-of-words model - representation of text data used in extracting features from the text
# Describes the occurrence of words in the given text (in this case a tokenized sentence) relative to all of the words in the data set
def bag_of_words(tok_sentence, all_words):
    # Stem each word in the tokenized sentence
    tok_sentence = [stem(word) for word in tok_sentence]
    # Instantiate a numpy zeros array of shape len(all_words) and type numpy float32 (required by PyTorch model)
    bag_of_words = np.zeros(len(all_words), dtype=np.float32)
    # For each of the (stemmed) words appearing in the data set (all_words), check if the word appears in the tokenized and stemmed sentence (tok_sentence) passed into the method
    # If the word appears in tok_sentence, set the element in the numpy zeros array (bag_of_words) at the corresponding index equal to 1.0, indicating that the word is present in the sentence
    for index, word in enumerate(all_words):
        if word in tok_sentence:
            bag_of_words[index] = 1.0
    return bag_of_words
