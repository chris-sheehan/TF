import numpy as np
import pickle
from sklearn.cross_validation import train_test_split
import tensorflow as tf

train_x, train_y, test_x, test_y = pickle.load(open('sentiment_set.pickle', 'rb'))

train_x = train_x[:10000]
train_y = train_y[:10000]
test_x = test_x[:10000]
test_y = test_y[:10000]

input_len = len(train_x[0])
n_nodes_h1 = 500
n_nodes_h2 = 500
n_nodes_h3 = 500
n_nodes_h4 = 500

n_classes = 2	
batch_size = 500
hm_epochs = 10

x = tf.placeholder('float', [None, input_len])
y = tf.placeholder('float', [None, n_classes])

# yhat = tf.placeholder('float', [None, len(test_x)])

print input_len

def load_file(filepath, sep = ','):
	with open(filepath, 'r') as f:
		rows = np.array([row.strip().split(sep) for row in f.readlines()]).astype(int)
	return rows

def neural_network_model(data):
	hidden_1_layer = dict(weights = tf.Variable(tf.random_normal([input_len, n_nodes_h1])), 
						  biases = tf.Variable(tf.random_normal([n_nodes_h1]))
						  )
	hidden_2_layer = dict(weights = tf.Variable(tf.random_normal([n_nodes_h1, n_nodes_h2])),
						  biases = tf.Variable(tf.random_normal([n_nodes_h2]))
						  )
	# hidden_3_layer = dict(weights = tf.Variable(tf.random_normal([n_nodes_h2, n_nodes_h3])),
	# 					  biases = tf.Variable(tf.random_normal([n_nodes_h3]))
	# 					  )
	# hidden_4_layer = dict(weights = tf.Variable(tf.random_normal([n_nodes_h3, n_nodes_h4])),
	# 					  biases = tf.Variable(tf.random_normal([n_nodes_h4]))
	# 					  )
	output_layer = dict(weights = tf.Variable(tf.random_normal([n_nodes_h2, n_classes])),
						  biases = tf.Variable(tf.random_normal([n_classes]))
						  )

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	# l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	# l3 = tf.nn.relu(l3)

	# l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
	# l4 = tf.nn.relu(l4)

	output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']
	return output

def train_neural_network(x):

	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction, y ))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			ii = 0
			while ii < len(train_x):
				start = ii
				end = ii + batch_size

				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])
				_, c = sess.run([optimizer, cost], feed_dict = {x : batch_x, y : batch_y})
				epoch_loss += c
				ii += 1
			print 'Epoch', epoch+1, 'completed out of', hm_epochs, '. loss:', epoch_loss

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print 'Accuracy:', accuracy.eval({x : test_x, y : test_y})
		yhat_test = np.argmax(sess.run(prediction, feed_dict = {x : test_x, y : test_y}), axis = 1)



if __name__ == '__main__':
	
	# print 'Loading vectors.'
	# neg_vecs = load_file('./data/neg_vecs.txt')
	# pos_vecs = load_file('./data/pos_vecs.txt')

	# X_input = np.append(neg_vecs, pos_vecs, axis = 0)
	# y_input = [[0,1] for _ in range(len(neg_vecs))] + [[1,0] for _ in range(len(pos_vecs))]
	# train_neural_network(x, y, X_input, y_input)
	train_neural_network(x)
