'''
Created by: Kevin De Angeli
Email: Kevindeangeli@utk.edu
Date: 12/30/20
'''

import string
import re
from pickle import dump, load
from unicodedata import normalize
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# split a loaded document into sentences
def to_sentences(doc):
    return doc.strip().split('\n')

def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# shortest and longest sentence lengths
def sentence_lengths(sentences):
    lengths = [len(s.split()) for s in sentences]
    return min(lengths), max(lengths)

# clean a list of lines
def clean_lines(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        # normalize unicode characters
        line = normalize('NFD', line).encode('ascii', 'ignore')
        line = line.decode('UTF-8')
        # tokenize on white space
        line = line.split()
        # convert to lower case
        line = [word.lower() for word in line]
        # remove punctuation from each token
        line = [word.translate(table) for word in line]
        # remove non-printable chars form each token
        line = [re_print.sub('', w) for w in line]
        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]
        # store as string
        cleaned.append(' '.join(line))
    return cleaned

def save_clean_sentences(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)

# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))

# create a frequency table for all words
def to_vocab(lines):
	vocab = Counter()
	for line in lines:
		tokens = line.split()
		vocab.update(tokens)
	return vocab

def to_vocab(lines):
	vocab = Counter()
	for line in lines:
		tokens = line.split()
		vocab.update(tokens)
	return vocab

def trim_vocab(vocab, min_occurance):
	tokens = [k for k,c in vocab.items() if c >= min_occurance]
	return set(tokens)

# mark all OOV with "unk" for all lines
def update_dataset(lines, vocab):
	new_lines = list()
	for line in lines:
		new_tokens = list()
		for token in line.split():
			if token in vocab:
				new_tokens.append(token)
			else:
				new_tokens.append('unk')
		new_line = ' '.join(new_tokens)
		new_lines.append(new_line)
	return new_lines

def tokenize(x, name=""):
	"""
	Tokenize x
	:param x: List of sentences/strings to be tokenized
	:return: Tuple of (tokenized x data, tokenizer used to tokenize x)
	"""
	# TODO: Implement
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(x)

	with open(name+'tokenizer.pickle', 'wb') as handle:
		pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
	return tokenizer.texts_to_sequences(x), tokenizer

def loadTokenizer(name):
	with open(name+'tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)
	return tokenizer


def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    # TODO: Implement
    return pad_sequences(x, maxlen=length, padding='post')

def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])