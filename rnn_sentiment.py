import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score

import logging
import common
logger = common.init_logging('rnn_sentiment.log')

## Import X_vecs, y_vecs from fintech_XY.pickle
logger.info('Loading pickled vecs...')
# X_vecs, y_vecs = pickle.load(open('fintech_XY.pickle', 'rb'))
# train_x, test_x, train_y, test_y = train_test_split(X_vecs, y_vecs)
# train_x, test_x, train_y, test_y = train_test_split(X_vecs[:1000], y_vecs[:1000])

train_x, train_y, test_x, test_y = pickle.load(open('sentiment_set.pickle', 'rb'))
train_x = np.array(train_x[:5000])
train_y = np.array(train_y[:5000])
test_x = np.array(test_x[:5000])
test_y = np.array(test_y[:5000])


ytrain_sing = np.argmax(train_y, 1) == 0
ytest_sing = np.argmax(test_y, 1) == 0

hm_epochs = 3
n_classes = 2
batch_size = 129

# chunk_size = 25
# n_chunks = 495
chunk_size = 12
n_chunks = 36
rnn_size = 128

logger.info('Init placeholder x,y.')
x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float', [None, n_classes])

def recurrent_neural_network_model(x):
	layer = dict(weights = tf.Variable(tf.random_normal([rnn_size, n_classes])), 
				 biases = tf.Variable(tf.random_normal([n_classes]))
				 )
	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, chunk_size])
	x = tf.split(0, n_chunks, x)

	lstm_cell = rnn_cell.BasicLSTMCell(rnn_size, forget_bias = 0.5)
	outputs, states = rnn.rnn(lstm_cell, x, dtype = tf.float32)

	output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
	return output


def train_neural_network(x, y):
	prediction = recurrent_neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction, y ))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
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
				batch_x = batch_x.reshape((min(batch_size, len(batch_x)), n_chunks, chunk_size))

				_, c = sess.run([optimizer, cost], feed_dict = {x : batch_x, y : batch_y})
				epoch_loss += c
				ii += 1
			print 'Epoch', epoch+1, 'completed out of', hm_epochs, '. loss:', epoch_loss
			logger.info('Epoch %s completed out of %s. Loss: %s.' % (epoch+1, hm_epochs, epoch_loss))

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		accuracy_eval = accuracy.eval({x : test_x.reshape((-1, n_chunks, chunk_size)), y : test_y}, session = sess)
		print 'Accuracy:', accuracy_eval
		logger.info('Accuracy: %s' % accuracy_eval)
		yhat_test = np.argmax(sess.run(prediction, feed_dict = {x : test_x.reshape((-1, n_chunks, chunk_size)), y : test_y}), axis = 1)
		logger.info('AUC: %.4f' % roc_auc_score(ytest_sing, yhat_test))
		logger.info('Precision: %.4f' % precision_score(ytest_sing, yhat_test))
		logger.info('Recall: %.4f' % recall_score(ytest_sing, yhat_test))
		print confusion_matrix(ytest_sing, yhat_test), '\n'

if __name__ == '__main__':
	train_neural_network(x, y)
