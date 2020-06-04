# word-level one-hot encoding

import numpy as np

samples = ["The cat sat on the mat", "The dog ate my homework"]

token_index = {}  # buils an index of all tokens in the data
for sample in samples:
    for word in sample.split():   # tokenize the samples via split method
        if word not in token_index:
            token_index[word] = len(token_index) + 1   # assigns a unique index to each unique word.we dont attribute index 0 to anything

max_length = 10  # vectorizes the samples. We only consider the first max_length words in each sample

print(max(token_index.values()) + 1)

results = np.zeros(shape=(len(samples),
                    max_length,
                    max(token_index.values())+1)) # here we store the results

sample = "The cat sat on the mat"
print(list(enumerate(sample.split()))[:max_length])

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1

print(results)