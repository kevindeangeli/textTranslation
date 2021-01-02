'''
Created by: Kevin De Angeli
Email: Kevindeangeli@utk.edu
Date: 12/30/20
'''
import numpy as np
import matplotlib.pyplot as plt
from tools import load_clean_sentences, save_clean_sentences, loadTokenizer, pad, logits_to_text
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from rnn_models import simple_model

#load English data
#filename = 'data/eur_english.pkl' #Cleaned data pre tokenizaer
#english = load_clean_sentences(filename)
#filename = 'data/eur_spanish.pkl' #Cleaned data pre tokenizer
#spanish = load_clean_sentences(filename)
#text_tokenized, text_tokenizer = tokenize(spanish, "en" )
#save_clean_sentences(text_tokenized, "english_tokenized")
#NOTE: remember to change to numpy before saving
#en_tokenized_padded = pad(en_tokenized, 642)
#sp_tokenized_padded = pad(sp_tokenized, 642)
#save_clean_sentences(en_tokenized_padded, "en_tokenized_padded")
#save_clean_sentences(sp_tokenized_padded, "sp_tokenized_padded")





'''
Max English sentence: 642
Max Spanish sentence: 632
'''
max_spanish_sequence_length = 632
max_english_sequence_lengh = 642

en_tokenized = load_clean_sentences("english_tokenized")
sp_tokenized = load_clean_sentences("spanish_tokenized")
sp_tokenizer = loadTokenizer("sp")
en_tokenizer = loadTokenizer("en")

en_tokenized_padded = pad(en_tokenized, 642)
sp_tokenized_padded = pad(sp_tokenized, 632)
sp_tokenized_padded = sp_tokenized_padded.reshape((-1,632,1))

spanish_vocab_size = len(sp_tokenizer.word_index)
english_vocab_size = len(en_tokenizer.word_index)
print("English vocabulary size:", english_vocab_size)
print("Spanish vocabulary size:", spanish_vocab_size)
print("----")
print(sp_tokenized_padded.shape)
print(en_tokenized_padded.shape)
print("read This:::::::", sp_tokenized_padded.shape[-2])
# Reshaping the input to work with a basic RNN
tmp_x = pad(en_tokenized_padded, 632)
tmp_x = tmp_x.reshape((-1, sp_tokenized_padded.shape[-2], 1))

# Train the neural network
simple_rnn_model = simple_model(
    tmp_x.shape,
    max_spanish_sequence_length,
    english_vocab_size,
    spanish_vocab_size+1)

print(simple_rnn_model.summary())

simple_rnn_model.fit(tmp_x, sp_tokenized_padded, batch_size=32, epochs=10, validation_split=0.2)
path = "savedModel.ckpt"

simple_rnn_model.save()
print(logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], sp_tokenizer))
# print("\nCorrect Translation:")
# print(french_sentences[:1])
#
# print("\nOriginal text:")
# print(english_sentences[:1])
print(logits_to_text(simple_rnn_model.predict(tmp_x[:2])[0], sp_tokenizer))
print(logits_to_text(simple_rnn_model.predict(tmp_x[:3])[0], sp_tokenizer))
print(logits_to_text(simple_rnn_model.predict(tmp_x[:4])[0], sp_tokenizer))






