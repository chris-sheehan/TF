# import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import common
from common.db_utils import *
from sklearn.cross_validation import train_test_split

conn = init_redshift()
conn.query('select np, is_relevant from np_relevance_train_set;')
nps = map(list, conn.cursor.fetchall())
conn.cursor.close()
npdict = {row[0] : row[1] for row in nps}
vecs = pd.read_csv('/Users/csheehan/Documents/repos/projects/news_trends/extracted_nps_vecs.txt')

vecs['relevant'] = vecs.np.apply(lambda x: npdict.get(x))
vecs.relevant = vecs.relevant.fillna(False)


X_vals = vecs.iloc[:, 2:-1].values
y_vals = np.array([[1,0]  if row.relevant else [0,1] for index, row in vecs.iterrows()])

train_x, test_x, train_y, test_y = train_test_split(X_vals, y_vals, test_size = .15)

input_len = len(train_x[0])
n_nodes_h1 = 500
n_nodes_h2 = 500

n_classes = 2
batch_size = 100
hm_epochs = 20

x = tf.placeholder('float', [None, input_len])
y = tf.placeholder('float', [None, n_classes])

def neural_network_model(data):
	hidden_1_layer = dict(weights = tf.Variable(tf.random_normal([input_len, n_nodes_h1])), 
						  biases = tf.Variable(tf.random_normal([n_nodes_h1]))
						  )
	hidden_2_layer = dict(weights = tf.Variable(tf.random_normal([n_nodes_h1, n_nodes_h2])),
						  biases = tf.Variable(tf.random_normal([n_nodes_h2]))
						  )
	output_layer = dict(weights = tf.Variable(tf.random_normal([n_nodes_h2, n_classes])),
						  biases = tf.Variable(tf.random_normal([n_classes]))
						  )

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']
	return output


def train_neural_network(x, y):
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


if __name__ == '__main__':
	train_neural_network(x, y)
