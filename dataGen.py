import numpy as np
import cv2
import glob
import dlib
import os
import matplotlib.pyplot as plt
import random
import time
from skimage import transform
import scipy.misc as scm

class DataGenerator:
	# split data into Train, Validation, Test sets
	def __init__(self, img_dir=None, gt_dir = None,train_data_file = None):
		""" Initializer
		Args:
			joints_name			: List of joints condsidered
			img_dir				: Directory containing every images
			train_data_file		: Text file with training set data
			remove_joints		: Joints List to keep (See documentation)
		"""
		self.img_dir = img_dir
		self.gt_dir = gt_dir
		self.train_data_file = train_data_file
		self.direction = ['gazeR','gazeG','gazeB']
		# self.direction = ['close','upleft','up','upright','left','front','right','downleft','down','downright']
		# self.direction = ['gaze']
		if img_dir != None:
			self.images = sorted(glob.glob(img_dir + '/*.png'))

	# Accessor
	def get_train(self):
		return self.train_set

	def get_valid(self):
		return self.valid_set

	def get_test(self):
		return self.test_set

	def get_batch(self, batch_size = 16, set = 'train'):
		list_file = []
		for i in range(batch_size):
			if set == 'train':
				list_file.append(random.choice(self.train_set))
			elif set == 'valid':
				list_file.append(random.choice(self.valid_set))
			else:
				print('Set must be : train/valid')
				break
		return list_file

	def gaussian(self, img, height,width, c1_x, c1_y, c2_x, c2_y, sigma = 1):
		""" Draw a gaussian map
		Args:
				width	 : input img width
				height	 : input img height
				c1_x	   : center point 1 x 
				c1_y	   : center point 1 y
				c2_x	   : center point 2 x 
				c2_y	   : center point 2 y
				sigma	 : 
		"""
		# height = img.shape[0]
		# width = img.shape[1]
		#centerp #.34 pts [128.,177.]
		centerp_x = 128. # ->64
		centerp_y = 150. # ->64

		deltx = centerp_x-64.
		delty = centerp_y-64.

		gaussian_map = np.zeros((height, width), dtype = np.float32)
		print(c2_x)
		print(c2_y)
		# Generate gaussian
		for x in range(width):
			for y in range(height):
				origin_x = x + deltx
				origin_y = y + delty
				if(x<width/2):
			  		g = np.exp(- ((origin_x - c1_x) ** 2 + (origin_y - c1_y) ** 2) / (2 * sigma ** 2))
				else:
					g = np.exp(- ((origin_x - c2_x) ** 2 + (origin_y - c2_y) ** 2) / (2 * sigma ** 2))
				gaussian_map[y, x] = g
		 
		return gaussian_map

	def generate_hm(self, img_name = None, img_dir = None, direction = 0, pts=None, out_name = None, out_dir = None):
		# # print(self.images)
		# single img
		if img_name != None:
			img = self.open_img(img_name,0)
			if direction == 0:
				im = self.gaussian(img,128,128, 0, 0, 0, 0, 0) #0
			elif pts != None:
				im = self.gaussian(img, 128, 128, pts[0][0], pts[0][1], pts[1][0], pts[1][1], 6)
			# im = self.gaussian(img, 64, 128, 192, 128, 10) #5
			if out_name != None:
				plt.imsave(out_name, im)
		# all img in a dir
		elif img_dir != None:
			count = 0
			path_imgs = img_dir
			files = sorted(glob.glob(path_imgs + '/*.png'))
			for name in files:
				# print(name)
				count += 1
				img = dlib.load_rgb_image(name)
				im = self.gaussian(img, pts[0][0], pts[0][1], pts[1][0], pts[1][1], 6) #3
				if not os.path.exists(out_dir):
					os.makedirs(out_dir)
				filename = out_dir+'/'+self.images[count]
				plt.imsave(filename, im)
		
		# test 2 point in large 9 
		# im = gaussian(img, 0, 0, 0, 0, 1) #0
		# im = gaussian(img, 21, 64, 149, 64, 1) #1
		# im = gaussian(img, 64, 64, 192, 64, 1) #2
		# im = gaussian(img, 106, 64, 234, 64, 1) #3
		# im = gaussian(img, 21, 128, 149, 128, 1) #4
		# im = gaussian(img, 64, 128, 192, 128, 1) #5
		# im = gaussian(img, 106, 128, 234, 128, 1) #6
		# im = gaussian(img, 21, 192, 149, 192, 1) #7
		# im = gaussian(img, 64, 192, 192, 192, 1) #8
		# im = gaussian(img, 95, 130, 190, 130, 1) #9

		print('heatmap generation done\n')

	def _create_train_table(self, train_dir = None):
		""" Create Table of samples #TODO just filenames array?
		"""
		self.train_table = []
		self.data_dict = {}
		input_file = open(self.train_data_file, 'r')
		for line in input_file:
			line = line.strip()
			line = line.split(' ')
			name = line[0]
			gtMap = line[1]
			direction = line[2] #Note: str type
			eyes = list(map(int,line[3:]))
			w = [1] * len(direction)
			if eyes != [-1] * len(eyes):
				eyes = np.reshape(eyes, (-1,2))
				# w = [1] * eyes.shape[0]
				for i in range(eyes.shape[0]):
					if np.array_equal(eyes[i], [-1,-1]):
						w[0] = 0 # w[1] TODO w len 1/2?
				self.data_dict[name] = {'gtMap' : gtMap, 'direction' : direction, 'eyes' : eyes, 'weights' : w}
				self.train_table.append(name)
		# else:
		# 	files = sorted(glob.glob(train_dir + '/*.png'))
		# 	self.train_table = files
		# 	for name in files:
		# 		w = [1] * len(self.direction)
		# for i in range(len(self.direction)):
		# 	if np.array_equal(joints[i], [-1,-1]):
		# 		w[i] = 0
				# self.data_dict[name] = {'gtMap' : ,'weights' : w}
		input_file.close()
		print('READING TRAIN Directory finished\n')
		

	def _randomize(self):
		""" shuffle the set
		"""
		random.shuffle(self.train_table)

	def _create_sets(self, validation_rate = 0.1):
		""" Select Elements to feed training and validation set 
		Args:
			validation_rate		: Percentage of validation data (in ]0,1[, don't waste time use 0.1)
		"""
		sample = len(self.train_table)
		valid_sample = int(sample * validation_rate)
		self.train_set = self.train_table[:sample - valid_sample]
		self.valid_set = self.train_table[sample - valid_sample:]
		# self.valid_set = []
		# preset = self.train_table[sample - valid_sample:]
		# print('START SET CREATION')
		# for elem in preset:
		# 	if self._complete_sample(elem):
		# 		self.valid_set.append(elem)
		# 	else:
		# 		self.train_set.append(elem)
		print('SET CREATED')
		# np.save('Dissertation/local_Dataset/video_frame/Dataset-Validation-Set', self.valid_set)
		# np.save('Dissertation/local_Dataset/video_frame/Dataset-Training-Set', self.train_set)
		print('--Training set :', len(self.train_set), ' samples.')
		print('--Validation set :', len(self.valid_set), ' samples.')
	
	def add_to_train():
		print('add to train set done\n')
	
	def generate_set(self, rand = False, file_dir = None):
		""" Generate the training and validation set
		Args:
			rand : (bool) True to shuffle the set
		"""
		self._create_train_table(file_dir)
		if rand:
			self._randomize()
		self._create_sets()

	def _aux_generator(self, batch_size = 16, stacks = 4, normalize = True, sample_set = 'train'):
		""" Auxiliary Generator
		Args:
			See Args section in self._generator
		"""
		while True:
			train_img = np.zeros((batch_size, 256,256,3), dtype = np.float32)
			# TODO now=[4,4,256,256,3]change size 
			train_gtmap = np.zeros((batch_size, stacks, 256, 256, len(self.direction)), np.float32)
			train_weights = np.zeros((batch_size, len(self.direction)), np.float32)
			i = 0
			while i < batch_size:
				try:
					if sample_set == 'train':
						name = random.choice(self.train_set)
					elif sample_set == 'valid':
						name = random.choice(self.valid_set)
					eyes = self.data_dict[name]['eyes']
					gtMap = self.data_dict[name]['gtMap']
					direction = self.data_dict[name]['direction']
					weight = np.asarray(self.data_dict[name]['weights'])
					train_weights[i] = weight 
					img = self.open_img(name,0)
					gtMap = self.open_img(gtMap,1)
					# # hm = self.generate_hm(img_name = name, direction = direction, pts=eyes)
					# # hm = scm.imresize(hm, (256,256))
					# # img, hm = self._augment(img, hm) # TODO augmentation??
					gtMap = np.expand_dims(gtMap, axis = 0)
					gtMap = np.repeat(gtMap, stacks, axis = 0)
					print(gtMap.shape)
					if normalize:
						train_img[i] = img.astype(np.float32) / 255
						train_gtmap[i] = gtMap.astype(np.float32) / 255
					else :
						train_img[i] = img.astype(np.float32)
						train_gtmap[i] = gtMap.astype(np.float32)
					i = i + 1
				except :
					print('error file: ', name)
			yield train_img, train_gtmap, train_weights


	def open_img(self, name, flag, color = 'RGB'):
		""" Open an image 
		Args:
			name	: Name of the sample
			color	: Color Mode (RGB/BGR/GRAY)
		"""
		# if name[-1] in self.letter: #TODO what it does?
		# 	name = name[:-1]
		if flag == 0: # img
			filename = os.path.join(self.img_dir, name)
		elif flag == 1:
			filename = os.path.join(self.gt_dir, name)
		# print(filename)
		img = cv2.imread(os.path.join(self.gt_dir, name))
		if color == 'RGB':
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			return img
		elif color == 'BGR':
			return img
		elif color == 'GRAY':
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			return img
		else:
			print('Color mode supported: RGB/BGR. If you need another mode do it yourself :p')





