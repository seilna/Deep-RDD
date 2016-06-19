import glob
import cv2
import load_dataset
import numpy as np
import tensorflow as tf
import sys
from getting_glasses_region import glasses_region
from getting_glasses_region import relative_region
# loading opencv cascade model for detecting face and eye regions.
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("./haarcascade_eye.xml")


"""
CNN models definition.
models consist of 9 convolution layers,
                  3 pooling layers,
                  2 fc layers.
"""

IM_SIZE = 32
BATCH_SIZE = 100
WINDOW_SIZE = 4

"""
There are 2 inference mode.
one is usaul state,
another is wearing glasses state.
you might change mode executing program with giving options.
"""
glasses = False


"""
Determines model's sensitivity. (3~5 recommended.)
"""
if len(sys.argv) == 2:
    WINDOW_SIZE = int(sys.argv[1])

if len(sys.argv) == 3:
	assert sys.argv[2] == "glasses", "invalid option used."
	glasses = True

assert len(sys.argv) < 4, "Too many options given."
print "model sensitivity >> %d" % WINDOW_SIZE

sess = tf.InteractiveSession()
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
keep_prob = tf.placeholder(tf.float32)

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

saver = tf.train.Saver()
SAVE_PATH = "./checkpoint/ver1.0_iteration.64000.ckpt"
saver.restore(sess, SAVE_PATH)


"""
getting realtime video of user,
detecting whether eye is closed or not using trained CNN models.
"""
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("/Users/naseil/Desktop/original.mov")
prev_face = [(0,0,30,30)]
prev_eyes = [(1,1,1,1), (1,1,1,1)]
drowsiness_check_list = [0] * WINDOW_SIZE
drowsiness_check_idx = 0


def rotate_check(face_size):
	face_size /= 10
	if not hasattr(rotate_check, "full_count"):
		rotate_check.full_count = 0
	if not hasattr(rotate_check, "size_distribution"):
		rotate_check.size_distribution = [0] * 5000
	rotate_check.full_count += 1
	rotate_check.size_distribution[face_size/100] += 1
	
	percentage = rotate_check.size_distribution[face_size/100] * 100 / rotate_check.full_count
	if percentage  < 10:
		return False
	rotate_check.size_distribution[face_size/100] += 1
	return True

relative_region_list = []
if glasses == True:
	relative_region_list = relative_region()
	print relative_region_list
continuous_error_count = 0
total_cnt = 0
error_classified_cnt = 0

error_path = "./errorounes_classified/"
error_file_num = 0
while True:
	ret, frame = cap.read()
	total_cnt  += 1
	frame = cv2.resize(frame,(250,250))
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	face = face_cascade.detectMultiScale(gray, 1.1, 3)

	error_check = False
	if len(face) != 1:
		face=prev_face
		error_check = True
		continuous_error_count += 1
	else:
		continuous_error_count = 0
		prev_face = face
	for a,b,w,h in face:
		face_size = w*h
		prev_face = face
        roi_gray = gray[b:b+h, a:a+w]
        roi_color = frame[b:b+h, a:a+w]
        eyes = []
        if glasses == False:
            eyes = eye_cascade.detectMultiScale(roi_gray)
        elif glasses == True and error_check == False:
			eyes = glasses_region(face, relative_region_list)
        if len(eyes) != 2:
            error_check = True
            eyes = prev_eyes
            continuous_error_count += 1
        else:
            continuous_error_count = 0
            prev_eyes = eyes

        cv2.rectangle(frame, (a,b), (a+w,b+h), (255,0,0), 1)
        eye_full_cnt = 0
        for f_ex,f_ey,f_ew,f_eh in eyes:
            ex, ey, ew, eh = int(f_ex), int(f_ey), int(f_ew), int(f_eh)
            eye_region_image = roi_color[ey:ey+eh, ex:ex+ew]
            prev_eyes = eyes
            p,q,r = eye_region_image.shape
            if p==0 or q==0 : break
            input_images = cv2.resize(eye_region_image, (32,32))
            input_images.resize((1,32,32,3))

            input_images = np.divide(input_images, 255.0)

            # Detecting drowsiness using CNN models.
            label = sess.run(tf.argmax(y_conv, 1), feed_dict={keep_prob:1.0, x:input_images})
            drowsiness_check_list[drowsiness_check_idx%WINDOW_SIZE] = label[0]
            drowsiness_check_idx += 1
            print drowsiness_check_list
            # if drowsiness if detected,
            # imaegs will be shown with red boxing.
            if rotate_check(face_size) == True and continuous_error_count < 5 and drowsiness_check_list == [1]*WINDOW_SIZE:
				cv2.rectangle(roi_color, (int(ex),int(ey)), (int(ex+ew), int(ey+eh)), (0,255,0), 1)
				eye_full_cnt += 1
				p,q,r = frame.shape
				error_classified_cnt += 1		
				error_file_num += 1
				error_file_name = error_path + str(error_file_num) + ".jpeg"
				#cv2.imwrite(error_file_name, frame)

				"""
				for i in xrange(p):
					for j in xrange(q):
					    frame[i,j,2] = 150 
				"""
            elif rotate_check(face_size) == True:
				cv2.rectangle(roi_color, (int(ex),int(ey)), (int(ex+ew), int(ey+eh)), (0,255,0), 1)
        if eye_full_cnt == 2:
			p,q,r = frame.shape
			for i in xrange(p):
				for j in xrange(q):
					frame[i,j,2] = 150

	cv2.imshow("Deep-CNN", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):break
cap.release()
cv2.destroyAllWindows()


