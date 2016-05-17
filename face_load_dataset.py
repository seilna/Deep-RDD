import numpy as np
import glob
import cv2
import subprocess

IM_SIZE = 64
class DataSet(object):
	def __init__(self, images, labels):
		self.images = images
		self.labels = labels
		self.epoch_completed = 0
		self.index_in_epoch = 0

		assert images.shape[0] == labels.shape[0], "assertion error : # of images and labels are not equated."
		self.num_examples = images.shape[0]

	def next_batch(self, batch_size):
		assert batch_size <= self.num_examples

		start = self.index_in_epoch
		self.index_in_epoch += batch_size

		if self.index_in_epoch > self.num_examples:
			"""
			if batch index touch the num_exmaples,
			shuffle the training dataset and start next new batch
			"""
			perm = np.arange(self.num_examples)
			np.random.shuffle(perm)
			self.images = self.images[perm]
			self.labels = self.labels[perm]

			#start the next batch
			start = 0
			self.index_in_epoch = batch_size

		end = self.index_in_epoch
		return self.images[start:end], self.labels[start:end]

def image_to_dataset():

    image, label = [], []
    images = None
    images = glob.glob('./face_region/usual/*')

    print "usual size >> " + str(len(images))
    for file in images:
        im = cv2.imread(file)
        im = cv2.resize(im, (IM_SIZE, IM_SIZE))
        image.append(im)
        label.append([1,0])

    images = glob.glob('./face_region/rotate/*')

    print "drowsiness size >> " + str(len(images))
    for file in images:
        im = cv2.imread(file)
        im = cv2.resize(im, (IM_SIZE, IM_SIZE))
        image.append(im)
        label.append([0,1])

    image = [x/float(255) for x in image]
    image = np.array(image)
    label = np.array(label)

    #print image
    perm = np.arange(image.shape[0])
    np.random.shuffle(perm)
    print perm
    image = image[perm]
    label = label[perm]

    return image, label

def read_dataset(validation_rate):

	assert validation_rate > 0 and validation_rate < 100

	class DataSets(object):
		pass
	data_sets = DataSets()

	image, label = image_to_dataset()

	VALIDATION_SIZE = validation_rate * image.shape[0] / 100

	train_image = image[VALIDATION_SIZE:]
	train_label = label[VALIDATION_SIZE:]

	validation_image = image[:VALIDATION_SIZE]
	validation_label = label[:VALIDATION_SIZE]

	data_sets.train = DataSet(train_image, train_label)
	data_sets.validation = DataSet(validation_image, validation_label)

	return data_sets

#data_sets = read_dataset(10)

