import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

hm_epochs = 10
n_classes = 2
batch_size = 100

chunk_size = 5
n_chunks = 8
rnn_size = 64


x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float', [None, n_classes])

def load_biodeg_data():
	with open('biodeg.csv', 'r') as f:
		biodeg = [row.strip().split(';') for row in f.readlines()]
	X_bio = np.array([row[:-2] for row in biodeg])
	y_bio = np.array([[row[-1] == 'RB', row[-1] != 'RB'] for row in biodeg])
	return X_bio, y_bio

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


def train_neural_network(x, y, X_bio, y_bio):
	prediction = recurrent_neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction, y ))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_x = X_bio.reshape((batch_size, n_chunks, chunk_size))
			_, c = sess.run([optimizer, cost], feed_dict = {x : X_bio, y : y_bio})
			epoch_loss = c
			if (epoch % 10) == 0:
				print 'Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print 'Accuracy:', accuracy.eval({x : X_bio.reshape((batch_size, n_chunks, chunk_size)), y : y_bio})


if __name__ == '__main__':
	X_bio, y_bio = load_biodeg_data()
	train_neural_network(x, y, X_bio, y_bio)
