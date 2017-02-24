import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

import pickle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score

import logging
import common
logger = common.init_logging('rnn_notMNIST.log')

logger.info('Loading pickled vecs...')
data = pickle.load(open('notMNIST_data.pickle', 'rb'))
Xdata = np.array([img[1] for img in data])
ydata = np.array([img[2] for img in data])

logger.info('Train Test Split...')
train_x, test_x, train_y, test_y = train_test_split(Xdata, ydata)
train_x = np.array(train_x[:500])
train_y = np.array(train_y[:500])
test_x = np.array(test_x[:500])
test_y = np.array(test_y[:500])


ytrain_sing = np.argmax(train_y, 1)
ytest_sing = np.argmax(test_y, 1)

hm_epochs = 5
n_classes = 10
batch_size = 128

chunk_size = 28
n_chunks = 28
rnn_size = 128

logger.info('Init placeholder x,y.')
x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float', [None, 10])

def recurrent_neural_network_model(x):
	layer = dict(weights = tf.Variable(tf.random_normal([rnn_size, n_classes])), 
				 biases = tf.Variable(tf.random_normal([n_classes]))
				 )
	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, chunk_size])
	x = tf.split(0, n_chunks, x)

	lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
	outputs, states = rnn.rnn(lstm_cell, x, dtype = tf.float32)

	output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
	return output


def train_neural_network(x, y):
	prediction = recurrent_neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		# sess = tf.Session()
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			logger.info('Starting epoch %d ...' % (epoch+1))
			epoch_loss = 0
			ii = 0
			while ii < len(train_x):
				logger.info('Epoch %s, Batch %s.' % (epoch+1, ii+1))
				start = ii
				end = ii + batch_size

				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])
				# batch_x = batch_x.reshape((min(batch_size, len(batch_x)), n_chunks, chunk_size))

				_, c = sess.run([optimizer, cost], feed_dict = {x : batch_x, y : batch_y})
				epoch_loss += c
				ii += 1
			print 'Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		accuracy_eval = accuracy.eval({x : test_x.reshape((-1, n_chunks, chunk_size)), y : test_y}, session = sess)
		print 'Accuracy:', accuracy_eval
		logger.info('Accuracy: %s' % accuracy_eval)
		yhat_test = np.argmax(sess.run(prediction, feed_dict = {x : test_x.reshape((-1, n_chunks, chunk_size)), y : test_y}), axis = 1)
		# logger.info('AUC: %.4f' % roc_auc_score(ytest_sing, yhat_test))
		# logger.info('Precision: %.4f' % precision_score(ytest_sing, yhat_test))
		# logger.info('Recall: %.4f' % recall_score(ytest_sing, yhat_test))
		print confusion_matrix(ytest_sing, yhat_test), '\n'

def y_specific_roc_auc(n, y, yhat):
	pairs = list()
	for y_, yhat_ in zip(y, yhat):
		if (y_ == n) | (yhat_ == n):
			pairs.append([y_==n, yhat_==n])
	return pairs

for n in range(10):
	pairs = y_specific_roc_auc(n, ytest_sing, yhat_test)
	print "%s : %.2f" % (n, roc_auc_score([_[0] for _ in pairs], [_[1] for _ in pairs]))


if __name__ == '__main__':
	train_neural_network(x, y)
