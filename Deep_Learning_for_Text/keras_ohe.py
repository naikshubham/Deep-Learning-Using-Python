# Using word-level one-hot encoding

from keras.preprocessing.text import Tokenizer

samples = ['Deep Learing for text and sequences', 'Generative Deeep Learing learns statistical patterns from the lanuage , thats what models are good at learning']

tokenizer = Tokenizer(num_words=15) # creates a tokenizer, configured to only take into account the 10 most common words
tokenizer.fit_on_texts(samples)     # builds the word index

sequences = tokenizer.texts_to_sequences(samples) # turns strings into list of integer indices
print('sequences->', sequences)

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

word_index = tokenizer.word_index  # recover the word index that was computed

print(tokenizer.word_index)
print(one_hot_results)