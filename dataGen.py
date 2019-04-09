from skimage import transform
from PIL import ImageEnhance
from PIL import Image
import numpy as np
import cv2
import glob
import dlib
import os
import random
import time

class DataGenerator:
	# split data into Train, Validation, Test sets
	def __init__(self, img_dir=None, gt_dir = None,train_data_file = None,outDim = 1,path_to_predictor='shape_predictor_68_face_landmarks.dat'):
		""" Initializer
		Args:
			joints_name			: List of joints condsidered
			img_dir				: Directory containing every images
			train_data_file		: Text file with training set data
			remove_joints		: Joints List to keep (See documentation)
		"""
		self.img_dir = img_dir
		self.gt_dir_l = gt_dir[0]
		self.gt_dir_r = gt_dir[1]
		self.train_data_file = train_data_file
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(path_to_predictor)
		if outDim == 3: 
			self.direction = ['gazeR','gazeG','gazeB']
		elif outDim == 2:
			self.direction=['left','right']
		elif outDim ==1:
			self.direction=['gaze']
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

	# ======================= Gaussian heatmap =======================

	def _gaussian(self, height,width, c1_x, c1_y, sigma = 1):
		""" Draw a gaussian map
		Args:
				width	 : input img width
				height	 : input img height
				c1_x	   : center point x 64*64 scale
				c1_y	   : center point y
				sigma	 : 
		"""
		x = np.arange(0, width, 1, float)
		y = np.arange(0, height, 1, float)[:, np.newaxis]
		return np.exp(-4*np.log(2) * ((x-c1_x)**2 + (y-c1_y)**2) / sigma**2)

	def generate_hm(self, img_name = None, img_dir = None, direction = 0, size = 64, pts=None, out_name = None, out_dir = None):
		# single img
		if img_name != None:
			if pts is None:
				if direction != -1:
					pts=self.get_center_coord(direction,version=0) #TODO version
				else:
					print('Please specify direction / center point coords to generate heatmap.')
					return
			im = np.zeros((size, size, len(self.direction)), dtype = np.float32)
			im[:,:,0] = self._gaussian(size, size, pts[0][0],pts[0][1],sigma= 3)
		# all img in a dir
		elif img_dir != None:
			count = 0
			path_imgs = img_dir
			# files = sorted(glob.glob(path_imgs + '/*.png'))
			for name in self.images:
				count += 1
				# img = dlib.load_rgb_image(name)
				im = np.zeros(shape=(size,size,3))
				im[:,:,0] = self._gaussian(size, size, pts[0][0], pts[0][1], 3) # left
				im[:,:,1] = self._gaussian(size, size, pts[1][0], pts[1][1], 3) # right
				# im = self.gaussian(img, size, size, pts[0][0], pts[0][1], pts[1][0], pts[1][1], 5) #0
				if not os.path.exists(out_dir):
					os.makedirs(out_dir)
				hm_name = out_dir+'/'+str(count)+'.png'
				cv2.imwrite('hm_name',cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
		return im

	# ======================= Image Processing =======================

	def getPts(self,image): # get facial points
		if isinstance(image, str):
			image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)   
		dets = self.detector(image)
		shape = self.predictor(image,dets[0])
		coords = np.zeros((68, 2), dtype='float')
		for i in range(0,68):
			coords[i] = (float(shape.part(i).x),float(shape.part(i).y))
		return coords

	def transformation_from_points(self,points1):
		# make points1 -> point2

		# make sure points are of same type
		points1 = points1.astype(np.float32)
		points2 = np.array([[62., 135.],[ 203., 128.],[ 84., 130.],[127., 124.],[173., 126.],[104., 199.]], np.float32)
 
		c1 = np.mean(points1, axis=0)
		c2 = np.mean(points2, axis=0)
		points1 -= c1
		points2 -= c2

		s1 = np.std(points1)
		s2 = np.std(points2)
		points1 /= s1
		points2 /= s2

		# ||RA-B||; M=BA^T
		A = points1.T # 2xN
		B = points2.T # 2xN
		M = np.dot(B, A.T)
		U, S, Vt = np.linalg.svd(M)
		R = np.dot(U, Vt)

		s = s2/s1
		sR = s*R
		c1 = c1.reshape(2,1)
		c2 = c2.reshape(2,1)
		T = c2 - np.dot(sR,c1) # 

		trans_mat = np.hstack([sR,T])   # 2x3

		return trans_mat

	def warp_im(self,in_image, trans_mat, dst_size):
		output_image = cv2.warpAffine(in_image,trans_mat,dst_size)
		return output_image

	def crop_im(self,image,points=None,size=(64,64)):
		if points is None:
			points = np.array([127., 124.], np.int64) # TODO find outNo.28 points middle eye
		XC = 32
		YC = 64
		leftx = int(round(max(XC-32,0)))
		rightx = int(round(min(XC+32,image.shape[1])))

		miny = int(round(max(YC-32,0)))
		maxy = int(round(min(YC+32,image.shape[0])))
		# print(leftx,rightx,miny,maxy)
		image_l = image[miny:maxy,leftx:rightx,: ]

		XC = 96
		YC = 64
		leftx = int(round(max(XC-32,0)))
		rightx = int(round(min(XC+32,image.shape[1])))

		miny = int(round(max(YC-32,0)))
		maxy = int(round(min(YC+32,image.shape[0])))
		image_r = image[miny:maxy,leftx:rightx,: ]
		return (image_l,image_r)

	def _transform(self,image,coords, filename = None): 
		# single img transformation
		row_idx1 = [0,16,36,27,45,48]
		points1 = coords[row_idx1,:]
		M = self.transformation_from_points(points1)
		frame = self.warp_im(image, M, (256,256))
		if (filename != None):
			cv2.imwrite(filename,cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
		return frame

	def _transform_all(self, img_dir = None, out_dir = None):
		# transform all imgs in dir
		if img_dir != None:
			files = sorted(glob.glob(img_dir + '/*.png'))
		else: 
			print('img_dir for transform err.\n')
			return
		count = 0
		for names in files:
			print(names)
			img = dlib.load_rgb_image(names)
			coords = self.getPts(img)
			count+=1
			filename = out_dir+'/frame'+str(count)+'.png'
			output = self._transform(img,coords,filename)
		return output # TODO need return or not?

	# ======================= Data Augmentation =======================

	def _rotate(self,image, hm_l, hm_r,max_rotation = 30, rand=True):
		r = 1
		if rand:
			r= random.choice([0,1])
		if r: 
			r_angle = np.random.randint(-1*max_rotation, max_rotation)
			image = transform.rotate(image, r_angle, preserve_range = True)
			hm_l = transform.rotate(hm_l, r_angle, preserve_range = True)
			hm_r = transform.rotate(hm_r, r_angle, preserve_range = True)
		return (image, hm_l,hm_r)

	def _flip(self,image,hm_l,hm_r, rand=True):
		r = 1
		if rand:
			r= random.choice([0,1])
		if r:
			f = random.choice([0,1])
			image = cv2.flip(image, f)
			hm_l = cv2.flip(hm_l, f)
			hm_r = cv2.flip(hm_r, f)
		return (image, hm_l,hm_r,r)

	def _color_augment(self,image,hm_l,hm_r,rand=True):
		r = 1
		if rand:
			r= random.choice([0,1])
		if r:
			image = Image.fromarray(image.astype(np.uint8)) 
			hm_l = Image.fromarray(hm_l.astype(np.uint8))
			hm_r = Image.fromarray(hm_r.astype(np.uint8))

			random_factor = np.random.randint(0, 31) / 10.
			color_image = ImageEnhance.Color(image).enhance(random_factor) # Saturation
			color_hm_l = ImageEnhance.Color(hm_l).enhance(random_factor)
			color_hm_r = ImageEnhance.Color(hm_r).enhance(random_factor)

			random_factor = np.random.randint(10, 21) / 10. 
			brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor) # Brightness
			brightness_hm_l = ImageEnhance.Brightness(color_hm_l).enhance(random_factor)
			brightness_hm_r = ImageEnhance.Brightness(color_hm_r).enhance(random_factor)
			
			random_factor = np.random.randint(10, 21) / 10.
			contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor) # Contrastness
			contrast_hm_l = ImageEnhance.Contrast(brightness_hm_l).enhance(random_factor)
			contrast_hm_r = ImageEnhance.Contrast(brightness_hm_r).enhance(random_factor)
			
			random_factor = np.random.randint(0, 31) / 10.
			enhance_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)
			enhance_hm_l = ImageEnhance.Sharpness(contrast_hm_l).enhance(random_factor)
			enhance_hm_r = ImageEnhance.Sharpness(contrast_hm_r).enhance(random_factor)

			image = np.array(enhance_image)
			hm_l = np.array(enhance_hm_l)
			hm_r = np.array(enhance_hm_r)
		return (image,hm_l,hm_r) #Sharpness

	# ======================= Prepare Dataset =======================

	def _create_train_table(self, train_dir = None):
		""" Create Table of samples #TODO just filenames array?
		"""
		self.train_table = []
		self.data_dict = {}
		input_file = open(self.train_data_file, 'r')
		for line in input_file:
			if line in ['\n', '\r\n']:
				print('READING end of file')
				break
			line = line.strip()
			line = line.split(' ')
			name = line[0]
			gtMap = line[1]
			direction = int(line[2]) #Note: str type
			eyes = list(map(int,line[3:]))
			w = [1] * len(self.direction)
			if eyes != [-1] * len(eyes):
				eyes = np.reshape(eyes, (-1,2))
				# w = [1] * eyes.shape[0]
				for i in range(eyes.shape[0]):
					if np.array_equal(eyes[i], [-1,-1]):
						w[0] = 0 # w[1] TODO w len 1/2?
				self.data_dict[name] = {'gtMap' : gtMap, 'direction' : direction, 'eyes' : eyes, 'weights' : w}
				self.train_table.append(name)
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
		print('SET CREATED')
		print('--Training set :', len(self.train_set), ' samples.')
		print('--Validation set :', len(self.valid_set), ' samples.')
	
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
			train_img = np.zeros((batch_size, 64,64,3), dtype = np.float32) #TODO just test with 128
			train_gtmap = np.zeros((batch_size, stacks, 64, 64, len(self.direction)), np.float32)
			train_weights = np.zeros((batch_size, len(self.direction)), np.float32)
			train_direction = np.zeros(batch_size, np.float32)
			i = 0
			while i < batch_size:
				try:
					if sample_set == 'train':
						name = random.choice(self.train_set)
					elif sample_set == 'valid':
						name = random.choice(self.valid_set)
					eyes = self.data_dict[name]['eyes']
					gtname = self.data_dict[name]['gtMap']
					train_direction[i] = self.data_dict[name]['direction']
					weight = np.asarray(self.data_dict[name]['weights'])
					train_weights[i] = weight 
					img = self.open_img(name,0)
					gtMap = self.generate_hm('test.txt', direction = train_direction[i], size = 64, pts=eyes)
					gtMap = np.expand_dims(gtMap, axis = 0)
					gtMap = np.repeat(gtMap, stacks, axis = 0) # 4*64*64*2
					if normalize:
						train_img[i] = img.astype(np.float32) / 255
						train_gtmap[i] = gtMap.astype(np.float32) / 255
					else :
						train_img[i] = img.astype(np.float32)
						train_gtmap[i] = gtMap.astype(np.float32)
					i = i + 1
				except Exception as e: 
					print('error file: ',name , gtname,' i = ',i)
					print(e)
					exit(0)
			yield train_img, train_gtmap, train_weights, train_direction

	# ======================= Update Dataset =======================

	def add_train_table(self,name,gtMap,dir_num,pts):
		name = name
		gtMap = gtMap
		w = [1] * len(self.direction)
		self.data_dict[name] = {'gtMap' : gtMap, 'direction' : dir_num, 'eyes' : pts, 'weights' : w}
		self.train_table.append(name)

	def write_txt(self, img_path, gt_path, direction, pts=None,filename=None):
		if img_path is None:
			num=0
			for file in os.listdir(self.img_dir):
				base=os.path.basename(file)
				tmp=img_path.split('.')
				n=str(tmp[0])
				if n >num:
					num=n
			num+=1
			img_path=str(num)+'.png'
		if gt_path is None:
			gt_path=img_path
		if pts is None:
			pts=self.get_center_coord(direction,version=2)
		if filename is None:
			filename=self.train_data_file
			
		with open(filename, 'a') as file:
			# print(pts[0][0])
			line=img_path+' '+gt_path+' '+str(direction)+' '+str(pts[0][0])+' '+str(pts[0][1])+' '+str(pts[1][0])+' '+str(pts[1][1])+'\n'
			print('writing: '+line)
			file.write(line)

	def replace(self,file_path, pattern, subst):
		#Create temp file
		fh, abs_path = mkstemp()
		with fdopen(fh,'w') as new_file:
			with open(file_path) as old_file:
				for line in old_file:
					new_file.write(line.replace(pattern, subst))
		#Remove original file
		remove(file_path)
		#Move new file
		move(abs_path, file_path)
		print('Replacing single line in txt done')


	# ======================= Helper Function =======================

	def open_img(self, name, flag, color = 'RGB'):
		""" Open an image 
		Args:
			name	: Name of the sample
			color	: Color Mode (RGB/BGR/GRAY)
		"""
		if flag == 0: # img
			filename = os.path.join(self.img_dir, name)
		elif flag == 1: # gtMap left
			filename = os.path.join(self.gt_dir_l, name)
		elif flag == 2: # gtMap right
			filename = os.path.join(self.gt_dir_r, name)
		
		img = cv2.imread(filename)
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

	def get_center_coord(self,direction,version=0, deltx=-1, delty=-1):
		if version==0: # eye center point based on true pts coord
			coord = [[[0,0],[0,0]],[[25,26],[25,26]],[[35, 24],[35, 24]],[[42,29],[42,29]],[[25,30],[25,30]],[[32,31],[32,31]],[[43, 33],[43, 33]],[[25, 38],[25, 38]],[[33, 42],[33, 42]],[[39, 38],[39, 38]]]
			coord = coord[direction]
		elif version==1: # eye center on 64*64 gtMap slicing into 3x6 grid
			coord = [[[0,0],[0,0]],[[5, 10],[37, 10]],[[16, 10],[48, 10]],[[27, 10],[59, 10]],[[5, 32],[37, 32]],[[16, 32],[48, 32]],[[27, 32],[59, 32]],[[5, 53],[37, 53]],[[16, 53],[48, 53]],[[27, 53],[59, 53]]]
			coord = coord[direction]
		elif version==2: # eye center pts on 64x64 gtMap slicing into 3*3 grid
			coord = [[[0,0],[0,0]],[[11,11],[11,11]],[[32,11],[32,11]],[[53,11],[53,11]],[[11,32],[11,32]],[[32,32],[32,32]],[[53,32],[53,32]],[[11,53],[11,53]],[[32,53],[32,53]],[[53,53],[53,53]]]
			coord = coord[direction]
		return coord




