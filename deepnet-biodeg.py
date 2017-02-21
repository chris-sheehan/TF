import numpy as np
import tensorflow as tf


input_len = 41
n_nodes_h1 = 500
n_nodes_h2 = 500
n_nodes_h3 = 500
n_nodes_h4 = 500

n_classes = 2
batch_size = 100
hm_epochs = 500

x = tf.placeholder('float', [None, input_len])
y = tf.placeholder('float', [None, n_classes])


def load_biodeg_data():
	with open('biodeg.csv', 'r') as f:
		biodeg = [row.strip().split(';') for row in f.readlines()]
	X_bio = np.array([row[:-1] for row in biodeg])
	y_bio = np.array([[row[-1] == 'RB', row[-1] != 'RB'] for row in biodeg])
	return X_bio, y_bio

def neural_network_model(data):
	hidden_1_layer = dict(weights = tf.Variable(tf.random_normal([input_len, n_nodes_h1])), 
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


def train_neural_network(x, y, X_bio, y_bio):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction, y ))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			_, c = sess.run([optimizer, cost], feed_dict = {x : X_bio, y : y_bio})
			epoch_loss = c
			if (epoch % 10) == 0:
				print 'Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print 'Accuracy:', accuracy.eval({x : X_bio, y : y_bio})


if __name__ == '__main__':
	X_bio, y_bio = load_biodeg_data()
	train_neural_network(x, y, X_bio, y_bio)
