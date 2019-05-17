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
		self.model = HourglassModel(nFeat=self.params['nfeats'], nStack=self.params['nstacks'], nLow=self.params['nlow'], outputDim=self.params['output_dim'], batch_size=self.params['batch_size'], training=True, drop_rate= self.params['dropout_rate'], lear_rate=self.params['learning_rate'], decay=self.params['learning_rate_decay'], decay_step=self.params['decay_step'], dataset=self.dataset, logdir_train=self.params['log_dir_train'], logdir_test=self.params['log_dir_test'], model_dir=self.params['model_dir'], w_loss=self.params['weighted_loss'], name=self.params['name'])
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

	def process_video(self, src = None, outName = None, codec = 'DIVX',save = False, dir = None, show = False, direction = True,debug = False):
		cam = cv2.VideoCapture(src)
		success,image = cam.read()
		shape = np.asarray((cam.get(cv2.CAP_PROP_FRAME_HEIGHT),cam.get(cv2.CAP_PROP_FRAME_WIDTH))).astype(np.int)
		frames = cam.get(cv2.CAP_PROP_FRAME_COUNT)
		fps = cam.get(cv2.CAP_PROP_FPS)
		if outName != None:
			fourcc = cv2.VideoWriter_fourcc(*codec)
			outVid = cv2.VideoWriter( outName, fourcc, fps, tuple(shape.astype(np.int))[::-1], 1)
		cur_frame = 0
		startT = time()
		count = 0
		while success and (cur_frame < frames or frames == -1) and outVid.isOpened():
			dets = self.detector(image)
			if len(dets) > 0:
				shape = self.predictor(image,dets[0])
				coords = np.zeros((68, 2), dtype='float')
				for i in range(0,68):
					coords[i] = (float(shape.part(i).x),float(shape.part(i).y))
				
				frame = self.dataset._transform(image,coords)
				frame_l,frame_r = self.dataset.crop_im(image,coords,size=(64,64))

				if save:
					if dir is None:
						dir = r'video/frame/' 
						filename = 'video/frame/frame'+str(count)+'.png'  
					else:
						filename1 = dir + '/left_'+str(count)+'.png'
						filename2 = dir + '/right_'+str(count)+'.png'
					if not os.path.exists(dir):
						os.makedirs(dir)
					cv2.imwrite(filename1,frame_l)	 
					cv2.imwrite(filename2,frame_r)
			
				if direction:
					pred_img_l = self.single_img_pred(image = frame_l)
					pred_img_r = self.single_img_pred(image = frame_r)

					dir_l,confidence_l,max_pts_l = self.model.pred_direction(pred_img_l[0,3,:,:,0],version=2)
					dir_r,confidence_r,max_pts_r = self.model.pred_direction(pred_img_r[0,3,:,:,0],version=2)

					print('For frame count: [',str(count),'] Prediction: DIRECTION (l,r): ',dir_l,', ',dir_r,' LEFT_CONFIDENCE = ', confidence_l,' RIGHT_CONFIDENCE = ', confidence_r)
					print('maxpoints left = ', max_pts_l,' right = ',max_pts_r)
					delta_l_x = coords[27][0]-64 # change coordinates back to original img scale
					delta_l_y = coords[27][1]-32

					delta_r_x = coords[27][0]-0
					delta_r_y = coords[27][1]-32
					cv2.circle(image, (int(max_pts_l[0]+delta_l_x),int(max_pts_l[1]+delta_l_y)), 5, (0,0,255), thickness=1)
					cv2.circle(image, (int(max_pts_r[0]+delta_r_x),int(max_pts_r[1]+delta_r_y)), 5, (0,255,0), thickness=1)
					
					txt='Predicted direction for frame '+str(count)+' left: '+str(dir_l)+' confidence: '+str(confidence_l)
					cv2.putText(image, txt, (int(coords[27][0]-60), int(coords[27][1]-20)), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1, cv2.LINE_AA)
					
					txt='Predicted direction for frame '+str(count)+' right: '+str(dir_l)+' confidence: '+str(confidence_r)
					cv2.putText(image, txt, (int(coords[27][0]-60), int(coords[27][1]+20)), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1, cv2.LINE_AA)	

				if show:
					cv2.imshow('stream', image)
				outVid.write(np.uint8(image))
			success,image = cam.read()
			print('Read a new frame: ', success,' count = ',count)
			count += 1
		cv2.destroyAllWindows()
		cam.release()
		if outName != None:
			print(outVid.isOpened())
			outVid.release()
			print(outVid.isOpened())
		print(time() - startT)	
		print('Video Done.')

	def save_output(self,output,filename=None,out_dir=None,pts=None):
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
			if pts is None:
				tmp=output[0,:,:,0]*255
			else:
				tmp= output[0,:,:,0]
				tmp[pts[1]][pts[0]]=tmp[pts[1]][pts[0]]*255
			cv2.imwrite(filename,tmp.astype(np.uint8))

	def single_img_pred(self,image=None,filename = None, transform=False, debug = False, sess = None):
		""" predicting single image 64x64x3
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

	def _test_set(self,batch_path):
		test_img,test_gt_pts,test_gt_direction = self.dataset._test_set_generator(batch_path)
		return test_img,test_gt_pts,test_gt_direction 

	def batch_test(self,batch_path,test_img,test_gt_pts,test_gt_direction,gt_txt=None,debug=False,sess=None):
		'''Batch test set
		'''
		if debug:
			t = time()
		if sess is None:
			out = self.sess.run(self.model.output, feed_dict={self.model.x : test_img})
		else:
			out = sess.run(self.model.output, feed_dict={self.model.x : test_img})
		
		num_image = out.shape[0]
		dir_accur,direction,confidence,max_pts = self.model.dir_accuracy(out[:,self.params['nstacks'] - 1,:,:,:],test_gt_direction,num_image,test_gt_pts)
		for i in range(num_image):
			print('i: ',i,' pred dir: ',direction[i],' confidence: ', confidence[i],' gt_dir: ', test_gt_direction[i])
		print('Total direction accuracy: ',dir_accur)
		if debug:
			print('Batch Test: ', time() - t, ' sec.')
		return test_img,dir_accur,direction,confidence,max_pts 

	def _accuracy(self,output,gt_filename=None,color='GREY',gtMaps=None,pts=None,debug=False, sess = None):
		''' Given a batch prediction and a Ground Truth batch (gtMaps) / pts
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
			pts = np.float32(pts)
			err = tf.add(err, self.model._compute_err(output[0,self.params['nstacks'] - 1,:,:,0], pts=pts[0]))
			u_y,u_x = self.model._argmax(output[0, self.params['nstacks'] - 1, :, :,0])
			err,u_y,u_x = self.sess.run([err,u_y,u_x])
			accuracy=err
			print(accuracy)
			print('Max coords: ',u_x,u_y)

		if debug:
			print('Accuracy Calculation: ', time() - t, ' sec.')
		return (accuracy,u_x,u_y)

	# ============================ Add dataset ============================

	def _add_train(self, prefix, image, confidence, direction,pts,train_path=None,txt_file=None):
		'''Adding confident data to training set
		'''
		name_list = []
		idx=[]
		for i in range(len(confidence)):
			if confidence[i] > 0.9:
				name=prefix+str(i)+'.png'
				name_list.append(name)
				idx.append(i)
				filename = train_path+'/'+name
				img = image[i,:,:,:]
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				cv2.imwrite(filename,img)
			else:
				name=prefix+str(i)+'.png'
				filename = 'uncertain/'+name
				img = image[i,:,:,:]
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				cv2.imwrite(filename,img)
		self.dataset.add_train_table(name_list,name_list,direction[idx],pts[idx])
		self.dataset.write_txt(name_list, name_list, direction[idx], pts=pts[idx],filename=txt_file)

	def _cml(self,train_txt,train_path,unlabel_path,uncertain_path,sess=None):
		'''CML prepare output with predictions
		'''
		files=sorted(glob.glob(unlabel_path + '/*.png'))
		num_image = len(files)
		unlabelled = np.zeros((num_image, 64,64,3), dtype = np.float32)
		direction = np.zeros(num_image, np.int32)
		confidence = np.zeros(num_image, np.float32)
		max_pts = np.zeros((num_image, 2), np.int32)
		i=0
		for name in files:
			im=cv2.imread(name)
			unlabelled[i]=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
			i+=1

		# test
		if sess is None:
			out = self.sess.run(self.model.output, feed_dict={self.model.x : unlabelled})
		else:
			out = sess.run(self.model.output, feed_dict={self.model.x : unlabelled})

		for i in range(num_image):
			direction[i],confidence[i],max_pts[i] = self.model.pred_direction(out[i,self.params['nstacks'] - 1,:,:,0])
		return unlabelled, direction, confidence, max_pts

	def _retrain(self):
		self.model.training_init(nEpochs=self.params['nepochs'], epochiter=self.params['epoch_size'], saveStep=self.params['saver_step'], dataset = self.dataset)
	# ============================ Plot Functions ===============================
	def _write_video(self,image,direction,confidence,max_pts):
		print('writing video')
		for i in range(len(direction)):
			im = image[i,:,:,:]

		cv2.circle(img1, (32,32), 5, (0,255,0), thickness=1)
		cv2.putText(img1, str(1), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1, cv2.LINE_AA)

		cv2.circle(img2, (40,40), 5, (0,255,0), thickness=1)
		cv2.putText(img2, str(2), (32, 32), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1, cv2.LINE_AA)

		height , width , layers =  img1.shape

		video = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc(*"XVID"),3,(width,height))

		video.write(img1)
		video.write(img2)
		video.write(img3)

		cv2.destroyAllWindows()
		video.release()

	# ============================ Benchmark Methods ============================

	def _mse(self,test_gt_direction,confidence):
		mse=[[0]*11 for _ in range(3)] # left,right,total (0-9 direction, avg)
		dataCount=[0,0,0,0,0,0,0,0,0,0] # img num for each direction

	def _mse_print(self,mse):
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
	gazedetector.model_init()

	if sys.argv[1] == '1':
	# for predicting a single image
		gazedetector.load_model(tf.train.latest_checkpoint('result/model/'))
		image_path = sys.argv[2]
		pred_img = gazedetector.single_img_pred(filename = image_path,debug=True)
		if len(sys.argv) == 4:
			gazedetector.save_output(pred_img[:, params['nstacks'] - 1, :, :,:],sys.argv[3])
	elif sys.argv[1] == '2':
		# calculate single accuracy
		gazedetector.load_model(tf.train.latest_checkpoint('result/model/'))
		image_path = sys.argv[2]
		pred_img = gazedetector.single_img_pred(filename = image_path,debug=True)
		pts=[[25],[25]] # providing ground truth label for calculation
		accuracy,x,y = gazedetector._accuracy(pred_img,pts=pts,debug=True)
		if len(sys.argv) == 5:
			gazedetector.save_output(pred_img[:, params['nstacks'] - 1, :, :,:],'train_x/frame100.png',pts=[x,y])
	elif sys.argv[1] == '3':
		# calculate single confidence
		gazedetector.load_model(tf.train.latest_checkpoint('result/model/'))
		image_path = sys.argv[2]
		pred_img = gazedetector.single_img_pred(filename = image_path,debug=True)
		direction,x,y = gazedetector._direction(pred_img, debug=True)
		l_confidence, r_confidence = gazedetector._confidence(pred_img,debug=True)
		print(direction,x,y)
		print(l_confidence, r_confidence)
	elif sys.argv[1] == '4':
		gazedetector.load_model(tf.train.latest_checkpoint('result/model/'))
		# for predicting a batch of unlabelled imgs
		batch_path = sys.argv[2]
		gazedetector.prediction(batch_path,show=True,debug=True)
	elif sys.argv[1] == '5':
		# for testing imgs in a directory with label
		gazedetector.load_model(tf.train.latest_checkpoint('result/model/'))
		batch_path = sys.argv[2]
		gt_txt = sys.argv[3]
		gazedetector.dataset._create_test_table(gt_txt)
		test_img,test_gt_pts,test_gt_direction = gazedetector._test_set(batch_path) # get test set img&label
		conf_img,dir_accur,direction,confidence,max_pts  = gazedetector.batch_test(batch_path,test_img,test_gt_pts,test_gt_direction,gt_txt,debug=True)
		# gazedetector._add_train('prefix', conf_img, confidence, direction,max_pts,'ttt.txt') #only use this if train_table exist
	elif sys.argv[1] == '6':
		# for video process - direction prediction
		gazedetector.load_model(tf.train.latest_checkpoint('result/result/model/'))

		video_path = sys.argv[2]
		outName = sys.argv[3]
		if len(sys.argv) == 5:
			save_path = sys.argv[4]
			s = True
		else:
			save_path=None
			s = False

		gazedetector.process_video(video_path, outName = outName, codec = 'DIVX', save=s, dir =save_path,direction = True)
	elif sys.argv[1] == '7':
	# for CML training
		gazedetector.dataset.generate_set(rand = True)
		gazedetector.model.training_init(nEpochs=2, epochiter=2, saveStep=params['saver_step'], dataset = gazedetector.dataset)

		train_txt=sys.argv[2]  # for adding new image path to training txt file 
		train_path=sys.argv[3] # for saving image to training set directory
		unlabel_path='unlabelled'
		uncertain_path='uncertain'
		
		gazedetector.load_model(tf.train.latest_checkpoint('result/result/model/'))

		unlabelled, direction, confidence, max_pts = gazedetector._cml(train_txt,train_path,unlabel_path,uncertain_path)
		gazedetector._add_train('prefix', unlabelled, confidence, direction,max_pts,train_path, train_txt) # dicide whether add to train
		gazedetector._retrain()

	print('Done: ', time() - t, ' sec.')
