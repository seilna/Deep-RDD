import glob
import cv2
import load_dataset
import tensorflow as tf
import numpy as np

IM_SIZE = 32
BATCH_SIZE = 100

sess = tf.InteractiveSession()

data_sets = load_dataset.read_dataset(20)
print "training dataset shape >> " + str(data_sets.train.images.shape)
print "validation dataset shape >> " + str(data_sets.validation.images.shape)

print data_sets.train.labels
print data_sets.validation.labels

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[None,IM_SIZE,IM_SIZE,3])
y_ = tf.placeholder(tf.float32, shape=[None,2])
keep_prob = tf.placeholder("float")


x_reshape = tf.reshape(x, [-1,IM_SIZE,IM_SIZE,3])


W_conv1 = weight_variable([5,5,3,16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x_reshape, W_conv1) + b_conv1)

W_conv2 = weight_variable([5,5,16,16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

W_conv3 = weight_variable([5,5,16,16])
b_conv3 = bias_variable([16])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)


"""
1 pooling
"""
h_pool3 = max_pool_2x2(h_conv3)


W_conv4 = weight_variable([5,5,16,32])
b_conv4 = bias_variable([32])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

W_conv5 = weight_variable([5,5,32,32])
b_conv5 = bias_variable([32])
h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)

W_conv6 = weight_variable([5,5,32,32])
b_conv6 = bias_variable([32])
h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)


"""
2 pooling
"""
h_pool6 = max_pool_2x2(h_conv6)

W_conv7 = weight_variable([5,5,32,64])
b_conv7 = bias_variable([64])
h_conv7 = tf.nn.relu(conv2d(h_pool6, W_conv7) + b_conv7)

W_conv8 = weight_variable([5,5,64,64])
b_conv8 = weight_variable([64])
h_conv8 = tf.nn.relu(conv2d(h_conv7, W_conv8) + b_conv8)

W_conv9 = weight_variable([5,5,64,64])
b_conv9 = bias_variable([64])
h_conv9 = tf.nn.relu(conv2d(h_conv8, W_conv9) + b_conv9)


"""
3 pooling
"""
h_pool9 = max_pool_2x2(h_conv9)
h_pool9_flat = tf.reshape(h_pool9, [-1, (IM_SIZE/8)*(IM_SIZE/8)*64])



W_fc1 = weight_variable([(IM_SIZE/8)*(IM_SIZE/8)*64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool9_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024,1024])
b_fc2 = bias_variable([1024])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([1024,2])
b_fc3 = bias_variable([2])

y_matmul = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

y_conv = tf.nn.softmax(y_matmul)

l2_loss = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv+1e-7), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-3).minimize(l2_loss)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.initialize_all_variables().run()

loss_curve = open('./loss_curve.csv', 'w')
saver = tf.train.Saver()

SAVE_PATH = "./checkpoint/"
for iteration in xrange(1000000):
    if iteration % 70 == 0:
        valid_batch_x, valid_batch_y = data_sets.validation.next_batch(BATCH_SIZE)
        acc = accuracy.eval(feed_dict={x: valid_batch_x,
                                       y_: valid_batch_y, keep_prob:1.0})
        loss = sess.run(l2_loss, feed_dict={x: valid_batch_x,
                                            y_: valid_batch_y, keep_prob:1.0})

        print '%dth iteration... accuracy >> %lf, loss .. %lf' % (iteration, acc, loss)
        contents = str(loss) + ',\n'
        loss_curve.write(contents)

    batch_x, batch_y = data_sets.train.next_batch(BATCH_SIZE)
    train_step.run(feed_dict={x:batch_x, y_:batch_y, keep_prob:0.5})

    if iteration % 4000 == 0  and iteration > 0 == 0 :
        checkpoint_file = SAVE_PATH + "ver1.0_iteration." + str(iteration) + ".ckpt"
        saver.save(sess,checkpoint_file)
        print "CNN models are saved in %s." % checkpoint_file



