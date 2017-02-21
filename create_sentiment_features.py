import numpy as np
import random
import pickle
from sklearn.feature_extraction.text import CountVectorizer

def load_textfile_rows(filepath, sep = ','):
    with open(filepath, 'r') as f:
        rows = [row.strip() for row in f.readlines()]
        return rows

def write_vectors_to_file(writefile, vecs):
    with open(writefile, 'w') as f:
        for vec in vecs:
            f.write(','.join([str(x) for x in vec]) + '\n')
    print writefile


def create_featureset_and_labels(pos, neg, test_size= 0.1):

	cv = CountVectorizer(lowercase = True, ngram_range=(1,1), min_df = 50, max_df = .33)
	cv.fit(neg + pos)

	neg_trans = cv.transform(neg)
	pos_trans = cv.transform(pos)
	
	features = list()
	features += zip(pos_trans.toarray(), [[1, 0] for _ in xrange(len(pos_trans.todense()))])
	features += zip(neg_trans.toarray(), [[0, 1] for _ in xrange(len(neg_trans.todense()))])
	random.shuffle(features)

	features = np.array(features)
	testing_size = int(test_size * len(features))

	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])

	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return train_x, train_y, test_x, test_y



# neg_trans = cv.transform(neg)
# pos_trans = cv.transform(pos)

# write_vectors_to_file('./data/neg_vecs.txt', np.array(neg_trans.todense()))
# write_vectors_to_file('./data/pos_vecs.txt', np.array(pos_trans.todense()))

if __name__ == '__main__':
	pos = load_textfile_rows('./data/pos.txt')
	neg = load_textfile_rows('./data/neg.txt')
	train_x, train_y, test_x, test_y = create_featureset_and_labels(pos, neg)

	with open('sentiment_set.pickle', 'wb') as f:
		pickle.dump([train_x, train_y, test_x, test_y], f)