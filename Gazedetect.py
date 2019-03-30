# transform img
import glob
import dlib
import numpy as np
import cv2
from time import time
import os

class Gazedetector:
	def __init__(self, path_to_predictor='Dissertation/Action-Units-Heatmaps/shape_predictor_68_face_landmarks.dat', enable_cuda=True):
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(path_to_predictor)
		# Initialise AU detector
		# self.Gazedetector = fn(1)
		# self.enable_cuda = enable_cuda
		# if not os.path.exists('model'):
			# os.mkdir('model')

		# if not os.path.isfile('model/AUdetector.pth.tar'):
		# 	request_file.urlretrieve(
		# 			"https://esanchezlozano.github.io/files/AUdetector.pth.tar",
		# 			'model/AUdetector.pth.tar')

		# net_weigths = torch.load('model/AUdetector.pth.tar', map_location=lambda storage, loc: storage)

		# net_dict = {k.replace('module.',''): v for k, v in net_weigths['state_dict'].items()}
		
		# self.Gazedetector.load_state_dict(net_dict)
		# if self.enable_cuda:
			# self.Gazedetector = self.Gazedetector.cuda()
		# self.Gazedetector.eval()

	def getPts(self,image):
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
 
		'''1 - 消除平移的影响 '''
		c1 = np.mean(points1, axis=0)
		c2 = np.mean(points2, axis=0)
		points1 -= c1
		points2 -= c2

		'''2 - 消除缩放的影响 '''
		s1 = np.std(points1)
		s2 = np.std(points2)
		points1 /= s1
		points2 /= s2

		'''3 - 计算矩阵M=BA^T；对矩阵M进行SVD分解；计算得到R '''
		# ||RA-B||; M=BA^T
		A = points1.T # 2xN
		B = points2.T # 2xN
		M = np.dot(B, A.T)
		U, S, Vt = np.linalg.svd(M)
		R = np.dot(U, Vt)

		'''4 - 构建仿射变换矩阵 '''
		s = s2/s1
		sR = s*R
		c1 = c1.reshape(2,1)
		c2 = c2.reshape(2,1)
		T = c2 - np.dot(sR,c1) # 模板人脸的中心位置减去 需要对齐的中心位置（经过旋转和缩放之后）

		trans_mat = np.hstack([sR,T])   # 2x3

		return trans_mat

	def warp_im(self,in_image, trans_mat, dst_size):
		output_image = cv2.warpAffine(in_image,trans_mat,dst_size)
		return output_image

	def crop_im(self,image,points=None,size=(256,256)):
		if points is None:
			points = np.array([127., 124.], np.int64) # TODO find outNo.28 points middle eye
		XC = points[27][0]
		YC = points[27][1]
		leftx = int(round(max(XC-128,1)))
		rightx = int(round(min(XC+128,image.shape[1])))

		miny = int(round(max(YC-128,1)))
		maxy = int(round(min(YC+128,image.shape[0])))
		# print(leftx,rightx,miny,maxy)
		image = image[miny:maxy,leftx:rightx,: ]
		return image
		# print('crop img done\n')

	# To test video haddling
	def process_video(self, src = None, save = False, dir = None, show = False):
	# def process_video(self, src = None, outName = None, thresh = 0.2, nms = 0.5 , codec = 'DIVX', pltJ = True, pltL = True, pltB = True, show = False):
		""" Process Video with face point detection
		Args:
			src				: Source (video path) ## TODO or 0 for webcam
			save		: outName (set name of output file, set to None if you don't want to save)
			dir			: Codec to use for video compression (see OpenCV documentation)
			show			: (bool) True to show the direction
			plt_j			: (bool) Plot Gaze Direction as circles
			plt_b			: (bool) Plot Bounding Boxes
		"""
		cam = cv2.VideoCapture(src)
		success,image = cam.read()
		count = 0
		while success:
			# if count == 5:
			# 	break
			dets = self.detector(image)
			# if (dets.size() is 0):
			# 	print('None dets\n')
			shape = self.predictor(image,dets[0])
			coords = np.zeros((68, 2), dtype='float')
			for i in range(0,68):
				coords[i] = (float(shape.part(i).x),float(shape.part(i).y))
			
			# cut directly #TODO or transform points first?
			frame = self.crop_im(image,coords)
			if save:
				if dir is None:
					dir = r'Dissertation/local_Dataset/video_frame/' 
					filename = 'Dissertation/local_Dataset/video_frame/frame'+str(count)+'.png'  
				else:
					filename = dir + '/frame'+str(count)+'.png'
				if not os.path.exists(dir):
					os.makedirs(dir)
				cv2.imwrite(filename,frame)
				# cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) # save frame as png file      
			
			if show:
				print('gaze direction is: ')
			success,image = cam.read()
			print('Read a new frame: ', success,' count = ',count)
			count += 1
		cam.release()
		print('processing video done\n')

	def _transform(self,image,filename = None):
		# filename : (bool)save img or not
		# if isinstance(image, str):
		# 	image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)   
		# dets = self.detector(image)
		# shape = self.predictor(image,dets[0])
		# coords = np.zeros((68, 2), dtype='float')
		# for i in range(0,68):
		# 	coords[i] = (float(shape.part(i).x),float(shape.part(i).y))
		coords = getPts(image)

		row_idx1 = [0,16,36,27,45,48]
		points1 = coords[row_idx1,:]
		M = self.transformation_from_points(points1)
		frame = self.warp_im(image, M, (256,256))
		if (filename != None):
			cv2.imwrite(filename,cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
		return frame

	def transform(self, img_dir = None, out_dir = None):
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
			count+=1
			filename = out_dir+'/frame'+str(count)+'.png'
			output = self._transform(img,filename)
		return output # TODO need return or not?

	def detectGaze(self,image,filename = None):
		print(filename)
		frame = self._transform(image)

		# image = frame.swapaxes(2,1).swapaxes(1,0)/255.0 # TODO not sure keep it or not
		# input = torch.from_numpy(image).float().unsqueeze(0) #TODO find out waht it does
		# if self.enable_cuda:
		# 	image = image.cuda()

		# input_var = torch.autograd.Variable(image)
		# outputs = self.Gazedetector(input_var) # feed into network
		# pred = np.zeros(5) # TODO out channel 
		# out_tmp = outputs[-1][0,:,:,:]
		# for k in range(0,5):
		# 	tmp = out_tmp[k,:,:].data.max()
		# 	if tmp < 0:
		# 		tmp = 0
		# 	elif tmp > 5:
		# 		tmp = 5
		# 	pred[k] = tmp

		# if self.enable_cuda:
		# 	maps = out_tmp.cpu()
		# else:
		# 	maps = out_tmp
		# return pred, maps, frame
		return frame





