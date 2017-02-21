import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np
import random
import pickle
from collections import Counter


lemmatizer = WordNetLemmatizer()
hm_lines = int(1e8)


def create_lexicon(pos, neg):
	lexicon = list()
	for fi in [pos, neg]:
		with open(fi, 'r') as f:
			contents = f.readlines()
			for ln in contents[:hm_lines]:
				try:
					all_words = word_tokenize(ln.lower())
					# all_words = ln.lower().split()
					lexicon += list(all_words)
				except UnicodeDecodeError as e:
					print all_words
					continue

	lexicon2 = list()
	for w in lexicon:
		try:
			lexicon2.append(lemmatizer.lemmatize(w))
		except UnicodeDecodeError as e:
			print w
			continue
	w_counts = Counter(lexicon2)

	lexicon_final = list()
	for w in w_counts:
		if 1000 > w_counts[w] > 50:
			lexicon_final.append(w)
	print len(lexicon_final)
	return lexicon_final


def sample_handling(sample, lexicon, classification):
	featureset = list()
	with open(sample, 'r') as f:
		contents = f.readlines()
		for ln in contents[:hm_lines]:
			try:
				current_words = word_tokenize(ln.lower())
				current_words = [lemmatizer.lemmatize(w) for w in current_words]
				features = np.zeros(len(lexicon))
	 			for word in current_words:
					if word.lower() in lexicon:
						index_val = lexicon.index(word.lower())
						feature[index_val] += 1
				features = list(features)
				featureset.append([features, classification])
			except UnicodeDecodeError as e:
				print current_words
				continue

	return featureset


def create_featureset_and_labels(pos, neg, test_size= 0.1):
	lexicon = create_lexicon(pos, neg)
	features = list()
	features += sample_handling(pos, lexicon, [1,0])
	features += sample_handling(neg, lexicon, [0,1])
	random.shuffle(features)

	features = np.array(features)
	testing_size = int(test_size * len(features))

	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])

	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return train_x, train_y, test_x, test_y


if __name__ == '__main__':

	train_x, train_y, test_x, test_y = create_featureset_and_labels('./data/pos.txt', './data/neg.txt')

	with open('sentiment_set.pickle', 'wb') as f:
		pickle.dump([train_x, train_y, test_x, test_y], f)
