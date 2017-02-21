# import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data/', one_hot = True)

n_nodes_h1 = 500
n_nodes_h2 = 500
n_nodes_h3 = 500
n_nodes_h4 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])

def neural_network_model(data):
	hidden_1_layer = dict(weights = tf.Variable(tf.random_normal([784, n_nodes_h1])), 
						  biases = tf.Variable(tf.random_normal([n_nodes_h1]))
						  )
	hidden_2_layer = dict(weights = tf.Variable(tf.random_normal([n_nodes_h1, n_nodes_h2])),
						  biases = tf.Variable(tf.random_normal([n_nodes_h2]))
						  )
	hidden_3_layer = dict(weights = tf.Variable(tf.random_normal([n_nodes_h2, n_nodes_h3])),
						  biases = tf.Variable(tf.random_normal([n_nodes_h3]))
						  )
	hidden_4_layer = dict(weights = tf.Variable(tf.random_normal([n_nodes_h3, n_nodes_h4])),
						  biases = tf.Variable(tf.random_normal([n_nodes_h4]))
						  )
	output_layer = dict(weights = tf.Variable(tf.random_normal([n_nodes_h4, n_classes])),
						  biases = tf.Variable(tf.random_normal([n_classes]))
						  )

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
	l4 = tf.nn.relu(l4)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
	return output


def train_neural_network(x, y):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction, y ))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	hm_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x : epoch_x, y : epoch_y})
				epoch_loss += c
			print 'Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print 'Accuracy:', accuracy.eval({x : mnist.test.images, y : mnist.test.labels})


if __name__ == '__main__':
	train_neural_network(x, y)
