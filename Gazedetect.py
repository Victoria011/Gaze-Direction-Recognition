from hourglass import HourglassModel
from time import time, clock, sleep
from train_launcher import process_config
from dataGen import DataGenerator
import numpy as np
import tensorflow as tf
import glob
import dlib
import numpy as np
import cv2
import os
import sys

class Gazedetector:
	def __init__(self, path_to_predictor='Dissertation/Action-Units-Heatmaps/shape_predictor_68_face_landmarks.dat', config_dict = 'config.cfg', enable_cuda=True):
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(path_to_predictor)
		self.params = config_dict
		self.dataset = DataGenerator(self.params['img_dir'],self.params['gt_dir'],self.params['training_txt_file'])
		self.model = HourglassModel(nFeat=self.params['nfeats'], nStack=self.params['nstacks'], nLow=self.params['nlow'], outputDim=self.params['output_dim'], batch_size=self.params['batch_size'], attention = self.params['mcam'],training=True, drop_rate= self.params['dropout_rate'], lear_rate=self.params['learning_rate'], decay=self.params['learning_rate_decay'], decay_step=self.params['decay_step'], dataset=self.dataset, logdir_train=self.params['log_dir_train'], logdir_test=self.params['log_dir_test'], model_dir=self.params['model_dir'], tiny= self.params['tiny'], w_loss=self.params['weighted_loss'], modif=False, name=self.params['name'])
		self.sess=tf.Session()


	# ============================ Hourglass Model ==============================
	def model_init(self):
		""" Initialize the Hourglass Model
		"""
		t = time()
		self.model.generate_model()
		self.sess.run(self.model.init)
		# with self.graph.as_default():
		# 	self.model.generate_model()
		print('Graph Generated in ', int(time() - t), ' sec.')

	def load_model(self, load = None):
		""" Load pretrained weights (See README)
		Args:
			load : File to load
		"""
		# with self.graph.as_default():
		# 	self.model.restore(load)
		self.model.restore(load)

	# ============================ IMAGE PROCESS ==============================

	def process_video(self, src = None, save = False, dir = None, show = False, direction = True,debug = False):
		cam = cv2.VideoCapture(src)
		success,image = cam.read()
		count = 0
		while success:
			# print(image.shape)
			dets = self.detector(image)
			shape = self.predictor(image,dets[0])
			coords = np.zeros((68, 2), dtype='float')
			for i in range(0,68):
				coords[i] = (float(shape.part(i).x),float(shape.part(i).y))
			
			# print(coords[36][0],coords[36][1])
			# cut directly 
			frame = self.dataset._transform(image,coords)
			frame = self.dataset.crop_im(image,coords)
			# frame = self._transform(frame, coords)
			if save:
				if dir is None:
					dir = r'video/frame/' 
					filename = 'video/frame/frame'+str(count)+'.png'  
				else:
					filename = dir + '/frame'+str(count)+'.png'
				if not os.path.exists(dir):
					os.makedirs(dir)
				cv2.imwrite(filename,frame)	 
			
			if direction:
				pred_img = self.single_img_pred(image = frame,debug=True)
				pred_dir, l_confidence, r_confidence = self._direction(pred_img[0], debug=True)
				print('For frame count: [',str(count),'] Prediction: DIRECTION = ',pred_dir,' LEFT_CONFIDENCE = ', l_confidence,' RIGHT_CONFIDENCE = ', r_confidence)
			if show:
				print('Draw direction. ')
			success,image = cam.read()
			print('Read a new frame: ', success,' count = ',count)
			count += 1
		cam.release()
		# if debug:
		# 	print('Video Detection Done: ', time() - t, ' sec.')
		print('Video Done.')

	def save_output(self,output,filename=None,out_dir=None):
		if filename is None:
			if out_dir is None:
				print('Please specify output path or filename.')
			else:
				# for i in range(len(output.shape[0])):
				filename = out_dir+'/0000.png'
				if self.params['output_dim'] == 1:
					output=output[:,:,:,0]
				cv2.imwrite(filename,output[i])
		else:
			# for i in range(len(filename)):
			tmp=output[0,:,:,0]*255
			cv2.imwrite(filename,tmp.astype(np.uint8))

	def single_img_pred(self,image=None,filename = None, transform=False, debug = False, sess = None):
		""" predicting single image
		"""
		if debug:
			t = time()
		if filename is not None:
			image = dlib.load_rgb_image(filename)
		if transform:
			coords = self.dataset.getPts(image)
			image = self.dataset._transform(image, coords)
		if image.shape == (64,64,3):
			if sess is None:
				out = self.sess.run(self.model.output, feed_dict={self.model.x : np.expand_dims(image, axis = 0)})
			else:
				out = sess.run(self.model.output, feed_dict={self.model.x : np.expand_dims(image, axis = 0)})
		else: 
			print('Image Size does not match placeholder shape')
			raise Exception
		if debug:
			print('Single Img Pred: ', time() - t, ' sec.')
		return out

	def _accuracy(self,output,gt_filename=None,color='GREY',gtMaps=None,pts=None,debug=False, sess = None):
		''' Given a batch prediction and a Ground Truth batch (gtMaps) / path
		'''
		output = tf.convert_to_tensor(output,np.float32)
		if debug:
			t = time()
		if pts is None:
			if gtMaps is None:
				if color == 'RGB':
					gtMap = cv2.imread(gt_filename)
				gtMap = cv2.imread(gt_filename, cv2.IMREAD_GRAYSCALE)
				gtMap = cv2.resize(gtMap, dsize=(64, 64), interpolation=cv2.INTER_AREA)

			# if gtMap.shape[:2] == (64,64):
				gtMap = np.expand_dims(gtMap, axis = 0)
				gtMap = np.repeat(gtMap, params['nstacks'], axis = 0)
				if gtMap.shape != (64,64,3):
					gtMap = np.expand_dims(gtMap, 3)
				gtMap = tf.expand_dims(gtMap, axis = 0)
		
			if sess is None:
				gtMap = gtMap.eval(session=self.sess)
			else:
				gtMap = gtMap.eval(session=sess)		
			self.model.set_label(gtMap)

			if(gtMap.shape[1:]!=(4,64,64,1)):
				accuracy = self.model._accur(output[:, self.params['nstacks'] - 1, :, :,0], gtMap[:, params['nstacks'] - 1, :, :,0], 1)
				if sess is None:
					accuracy = self.sess.run(accuracy)
				else:
					accuracy = sess.run(accuracy)
					print(accuracy)
			else:
				print('Ground Truth Image Size does not match placeholder shape')
				raise Exception
		else:
			err = tf.to_float(0)
			# right_err = tf.to_float(0)
			pts = np.float32(pts)
			err = tf.add(err, self.model._compute_err(output[0,self.params['nstacks'] - 1,:,:,0], pts=pts[0]))
			u_y,u_x = self.model._argmax(output[0, self.params['nstacks'] - 1, :, :,0])
			# err_r = tf.add(right_err, self.model._compute_err(output[0,self.params['nstacks'] - 1,:,:,1], pts=pts[1]))
			# err_l,err_r = self.sess.run([err_l,err_r])
			err,u_y,u_x = self.sess.run([err,u_y,u_x])
			accuracy=err
			print(accuracy)
			print('Max coords: ',u_y,u_x)

		if debug:
			print('Accuracy Calculation: ', time() - t, ' sec.')
		return accuracy

	# ============================ Benchmark Methods ============================

	def mse_print(self,mse):
		# Print MSE
		print('=================== MSE Evaluation: =================== ')
		print('Direction : close(0)| l_up(1)|  up(2)| r_up(3)| left(4)| front(5)| right(6)| l_down(7)| down(8)| r_down(9)| Average')
		print('Left MSE  :   ','{0:.2f}'.format(mse[0][0]),'|  ','{0:.2f}'.format(mse[0][1]),'| ','{0:.2f}'.format(mse[0][2]),'|  ','{0:.2f}'.format(mse[0][3]),'|  ','{0:.2f}'.format(mse[0][4]),'|   ','{0:.2f}'.format(mse[0][5]),'|   ','{0:.2f}'.format(mse[0][6]),'|    ','{0:.2f}'.format(mse[0][7]),'|  ','{0:.2f}'.format(mse[0][8]),'|    ','{0:.2f}'.format(mse[0][9]),'|  ','{0:.2f}'.format(mse[0][10]))
		print('Right MSE :   ','{0:.2f}'.format(mse[1][0]),'|  ','{0:.2f}'.format(mse[1][1]),'| ','{0:.2f}'.format(mse[1][2]),'|  ','{0:.2f}'.format(mse[1][3]),'|  ','{0:.2f}'.format(mse[1][4]),'|   ','{0:.2f}'.format(mse[1][5]),'|   ','{0:.2f}'.format(mse[1][6]),'|    ','{0:.2f}'.format(mse[1][7]),'|  ','{0:.2f}'.format(mse[1][8]),'|    ','{0:.2f}'.format(mse[1][9]),'|  ','{0:.2f}'.format(mse[1][10]))
		print('Total MSE :   ','{0:.2f}'.format(mse[2][0]),'|  ','{0:.2f}'.format(mse[2][1]),'| ','{0:.2f}'.format(mse[2][2]),'|  ','{0:.2f}'.format(mse[2][3]),'|  ','{0:.2f}'.format(mse[2][4]),'|   ','{0:.2f}'.format(mse[2][5]),'|   ','{0:.2f}'.format(mse[2][6]),'|    ','{0:.2f}'.format(mse[2][7]),'|  ','{0:.2f}'.format(mse[2][8]),'|    ','{0:.2f}'.format(mse[2][9]),'|  ','{0:.2f}'.format(mse[2][10]))

	def demo(self):
		print('Demo')


if __name__ == '__main__':
	t = time()


	print('Number of arguments:', len(sys.argv), 'arguments.') 
	print('Argument List:', str(sys.argv)) 
	print('Argument 1: ',sys.argv[1])

	params = process_config('config.cfg')
	gazedetector = Gazedetector('shape_predictor_68_face_landmarks.dat',params,enable_cuda=False)
	# predict.color_palette()
	# predict.LINKS_JOINTS()
	gazedetector.model_init()
	gazedetector.load_model(tf.train.latest_checkpoint('result/result_01/model/'))

	if sys.argv[1] == '1':
	# for predicting a single image
		image_path = 'train_x/frame0.png'
		pred_img = gazedetector.single_img_pred(filename = image_path,debug=True)
		if len(sys.argv) == 3:
			gazedetector.save_output(pred_img[:, params['nstacks'] - 1, :, :,:],'train_x/frame100.png')
	elif sys.argv[1] == '2':
		# calculate single accuracy
		image_path = 'train_x/frame0.png'
		pred_img = gazedetector.single_img_pred(filename = image_path,debug=True)
		gazedetector.save_output(pred_img[:, params['nstacks'] - 1, :, :,:],'train_x/frame100.png')
		pts=[[25,26],[25,26]]
		accuracy = gazedetector._accuracy(pred_img,pts=pts,debug=True)
	elif sys.argv[1] == '3':
		# calculate single confidence
		image_path = 'train_1/test/399.png'
		pred_img = gazedetector.single_img_pred(filename = image_path,debug=True)
		print(pred_img.shape[0])
		direction,x,y = gazedetector._direction(pred_img, debug=True)
		l_confidence, r_confidence = gazedetector._confidence(pred_img,debug=True)
		print(direction,x,y)
		print(l_confidence, r_confidence)
	elif sys.argv[1] == '4':
		# for predicting a batch of unlabelled imgs
		batch_path = 'train_1/test/batch/'
		gazedetector.prediction(batch_path,show=True,debug=True)
	elif sys.argv[1] == '5':
	# for testing imgs in a directory with label
		batch_path = 'train_1/test/batch/'
		gt_dir=[1,2,3,4,5,6,7,8,9]
		dir_accuracy = gazedetector.batch_test(batch_path,gt_dir,debug=True)
		print(dir_accuracy)
	elif sys.argv[1] == '6':
		# for video process - direction prediction
		video_path = 'video/UserFrontal_C_End.avi'
		save_path = 'video/frame'
		gazedetector.process_video(video_path, save = True, dir = save_path, show = False, direction = True,debug = True)
	elif sys.argv[1] == '7':
	# for test set MSE calculation
		gazedetector.test('test.txt' ,save = False, print = True, debug = True)
		# mse=[[2.345]*11 for _ in range(3)]
		# gazedetector.mse_print(mse)

	# For drawing point on img
	# cv2.circle()
	# cv2.putText(

	print('Done: ', time() - t, ' sec.')
