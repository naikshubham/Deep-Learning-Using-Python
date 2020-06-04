import numpy as np 

samples = ["the cat sat on the mat, why not on the flooe, it should have been sat on the floor", "AI in the new Game Changer", "Deep Learning at its best"]

token_index ={}    # dict
max_words = 15

for sample in samples:
	for word in sample.split():
		if word not in token_index:
			if len(token_index) == 15:
				break
			token_index[word] = len(token_index) + 1


print('token_index->', token_index)
max_length = 10

print(len(samples), max_length, max(token_index.values()) + 1)
results = np.zeros(shape=(len(samples),
							max_length,
							max(token_index.values()) +1))
print(results.shape)

for i, sample in enumerate(samples):
	for j, word in list(enumerate(sample.split()))[:max_length]:
		index = token_index.get(word)
		results[i, j, index] = 1

print(results)
