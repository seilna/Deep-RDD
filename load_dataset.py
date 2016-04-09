import numpy as np
import glob
import cv2


class Dataset(object):
	def __init__(self, images, labels):
		self.images = images
		self.labels = labels
		self.epoch_completed = 0
		self.index_in_epoch = 0
		
		self.num_exmaples = images.shape[0]



	def next_batch(self, batch_size):
		start = self.index_in_epoch
		self.index_in_epoch += batch_size
		

