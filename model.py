import tensorflow as tf
import numpy as np
import load_dataset
sess = tf.InteractiveSession()
data_sets = load_dataset.read_dataset(10)
print data_sets.validation.images.shape
print data_sets.validation.labels.shape


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

x = tf.placeholder("float", shape=[None,32,32,3])
y_ = tf.placeholder("float", shape=[None,2])
keep_prob = tf.placeholder("float")


x_reshape = tf.reshape(x, [-1,32,32,3])

W_conv1 = weight_variable([3,3,3,16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x_reshape, W_conv1) + b_conv1)

W_conv2 = weight_variable([3,3,16,16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

W_conv3 = weight_variable([3,3,16,16])
b_conv3 = bias_variable([16])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)


h_pool3 = max_pool_2x2(h_conv3)


W_conv4 = weight_variable([3,3,16,32])
b_conv4 = bias_variable([32])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

W_conv5 = weight_variable([3,3,32,32])
b_conv5 = bias_variable([32])
h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)

W_conv6 = weight_variable([3,3,32,32])
b_conv6 = bias_variable([32])
h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)


h_pool6 = max_pool_2x2(h_conv6)
h_pool6_flat = tf.reshape(h_pool6, [-1, 8*8*32])

W_fc1 = weight_variable([8*8*32, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool6_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024,1024])
b_fc2 = bias_variable([1024])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([1024,2])
b_fc3 = bias_variable([2])
y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

cross_entropy = -tf.reduce_sum(y_conv*tf.log(y_+1e-7))
train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

for iteration in xrange(100000):
	if iteration % 20 == 0:
		acc = accuracy.eval(feed_dict={x:data_sets.validation.images, 
				y_:data_sets.validation.labels, keep_prob:1.0})
		loss = sess.run(cross_entropy, feed_dict={x:data_sets.validation.images,
				y_:data_sets.validation.labels, keep_prob:1.0})
	
		print '%dth iteration... accuracy >> %lf, loss .. %lf' % (iteration, acc, loss/float(data_sets.validation.num_examples))

	batch_x, batch_y = data_sets.train.next_batch(50)
	sess.run([train_step], feed_dict={x:batch_x, y_:batch_y, keep_prob:0.5})














