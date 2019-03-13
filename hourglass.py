import time
import tensorflow as tf
import numpy as np
import sys
import datetime
import os

# ------------- process dataset
import matplotlib.pyplot as plt
import glob
from PIL import Image
from random import randint

# Load data
x_train = []
y_train = []
x_test = []
y_test = []

x_path_imgs = 'Dissertation/Dataset/train_x'
y_path_imgs = 'Dissertation/Dataset/train_y'
x_files = sorted(glob.glob(x_path_imgs + '/*.png'))
y_files = sorted(glob.glob(y_path_imgs + '/*.png'))
for filename in x_files:
    img = Image.open(filename)
    data = np.array(img,dtype="float32") / 255.0  
    x_train.append(data)
    im64 = img.resize((64, 64), Image.ANTIALIAS)
    data = np.array(im64,dtype="float32") / 255.0  
    y_train.append(data)

x_train = np.array(x_train)
y_train = np.array(y_train)
# x_train = tf.convert_to_tensor(x_train, np.float32)
# y_train = tf.convert_to_tensor(y_train, np.float32)
print(x_train.shape, y_train.shape)

imgNum = x_train.shape[0]
print(imgNum)

# ------------- 
class HourglassModel():
	def __init__(self, nFeat = 256, nStack = 1, nModules = 1, nLow = 1, outputDim = 3, batch_size = 32, lear_rate = 2.5e-4, decay = 0.96, decay_step = 2000, dataset = None, training = True, w_summary = True, w_loss = False, name = 'hourglass'):
		""" Initializer
		Args:
			nStack				: number of stacks (stage/Hourglass modules)
			nFeat				: number of feature channels on conv layers
			nLow				: number of downsampling (pooling) per module
			outputDim			: number of output Dimension (16 for MPII)
			batch_size			: size of training/testing Batch
			##dro_rate			: Rate of neurons disabling for Dropout Layers
			lear_rate			: Learning Rate starting value
			decay				: Learning Rate Exponential Decay (decay in ]0,1], 1 for constant learning rate)
			decay_step			: Step to apply decay
			dataset			: Dataset (class DataGenerator)
			training			: (bool) True for training / False for prediction
			w_summary			: (bool) True/False for summary of weight (to visualize in Tensorboard)
			??tiny				: (bool) Activate Tiny Hourglass
			??attention			: (bool) Activate Multi Context Attention Mechanism (MCAM)
			## logdir_train/test ??
			##modif				: (bool) Boolean to test some network modification # DO NOT USE IT ! USED TO TEST THE NETWORK
			name				: name of the model
		"""
		self.nStack = nStack
		self.nFeat = nFeat
		self.nModules = nModules
		self.outDim = outputDim
		self.batchSize = batch_size
		self.training = training
		self.w_summary = w_summary
		# self.tiny = tiny
		# self.dropout_rate = drop_rate
		self.learning_rate = lear_rate
		self.decay = decay
		self.name = name
		self.decay_step = decay_step
		self.nLow = nLow
		self.dataset = dataset
		# self.cpu = '/cpu:0'
		# self.gpu = '/gpu:0'
		# self.logdir_train = logdir_train
		self.logdir_train = 'Dissertation/log/'
		self.logdir_test = 'Dissertation/log/'
		self.w_loss = w_loss
		self.joints = 3 #outputDim
		self.imgAccur = [0,0,0]
		self.train_weights = [1] * self.batchSize
		self.w_summary = True
		print("Init model done.")

	def generate_model(self):
		print('Creating Model')
		t_start = time.time()
		with tf.name_scope('inputs'):
			self.x = tf.placeholder("float", [None, 256,256,3]) # 'None' cuz define at run time
			self.y = tf.placeholder("float", [None, 64,64,3])
			print('--Inputs : Done')
		with tf.name_scope('model'):
			self.output = self.fullNetwork(self.x, 256, 1, 3)
			print('--Model : Done')
		with tf.name_scope('loss'):
			self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels= self.y), name = 'cross_entropy_loss') 
			print('--Loss : Done')
		with tf.name_scope('steps'):
			self.train_step = tf.Variable(0, trainable=False)
		# with tf.name_scope('lr'):
			# self.lr = tf.train.exponential_decay(self.learning_rate, self.train_step, self.decay_step, self.decay, staircase= True, name= 'learning_rate')
		with tf.name_scope('rmsprop_optimizer'):
			self.rmsprop = tf.train.RMSPropOptimizer(learning_rate= self.learning_rate)
			print('--Optim : Done')
		with tf.name_scope('minimize'):
			self.train_rmsprop = self.rmsprop.minimize(self.loss, self.train_step)
			print('--Minimizer : Done')
		self.init = tf.global_variables_initializer()
		print('--Init : Done')
		with tf.name_scope('training'):
			tf.summary.scalar('loss', self.loss, collections = ['train'])
		with tf.name_scope('summary'):
			for i in range(self.joints):
				tf.summary.scalar('imgAccuracy', self.imgAccur[i], collections = ['train', 'test'])
		self.train_op = tf.summary.merge_all('train')
		# self.test_op = tf.summary.merge_all('test')
		self.weight_op = tf.summary.merge_all('weight')
		print('Model generation: ' + str(time.time()- t_start))
		del t_start

	def conv2d(self, inputs, filters, kernel_size = 1, strides = 1, pad = 'VALID', name = None):
		with tf.name_scope(name):
		# kernel = tf.get_variable('weight', shape=[kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters], initializer=tf.random_normal_initializer())
			kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name= 'kernel')
			conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding=pad, data_format='NHWC')
			if self.w_summary:
				with tf.device('/cpu:0'):
					tf.summary.histogram('weights_summary', kernel, collections = ['weight'])
		# with tf.device('/cpu:0'):
			# tf.summary.histogram('weights_summary', kernel, collections = ['train'])
		return conv

	def convBlock(self, inputs, numOut, name = 'convBlock'):
		# DIMENSION CONSERVED
		with tf.name_scope(name):
			expansion = 2
			outplanes = int(numOut/expansion)
			conv_1 = self.conv2d(inputs, outplanes, kernel_size=1, strides=1, pad = 'VALID')
			norm_1 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu)
			pad = tf.pad(norm_1, np.array([[0,0],[1,1],[1,1],[0,0]]))
			conv_2 = self.conv2d(pad, outplanes, kernel_size=3, strides=1)
			norm_2 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu)
			conv_3 = self.conv2d(norm_2, numOut, kernel_size=1, strides=1, pad = 'VALID')
			norm_3 = tf.contrib.layers.batch_norm(conv_3, 0.9, epsilon=1e-5) 
		return norm_3

	def downsample(self, inputs, numOut, name = 'downsample'):
		with tf.name_scope(name):
			conv = self.conv2d(inputs, numOut, kernel_size=1, strides=1, pad = 'VALID') #(64,128)
			norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5) #(128)
		return norm

	def bottleneck(self, inputs, numOut, ds = 0, dsOut = 0, name = 'bottleneck'):
		# DIMENSION CONSERVED
		with tf.name_scope(name):
			convb = self.convBlock(inputs, numOut)
			if ds == 1:
				ds = self.downsample(inputs,dsOut) 
			else:
				ds = inputs
			out = tf.add_n([convb,ds])
			out = tf.nn.relu(out,name='relu')
			return out

	def hourglass(self, inputs, n, numOut, name = 'hourglass'):
		with tf.name_scope(name):
			up_1 = self.bottleneck(inputs, numOut, name = 'up1')
			low_ = tf.contrib.layers.max_pool2d(inputs, [2,2],[2,2], 'VALID')
			low_1 = self.bottleneck(low_, numOut, name = 'low1')
        
			low_2 = low_1
			if n > 1:
				low_2 = self.hourglass(low_1, n-1, numOut, name='low2') ##### ? change low_1?
			else:
				low_2 = self.bottleneck(low_1, numOut, name='low2') #####到底是low1/2?
			low_3 = self.bottleneck(low_2, numOut, name = 'low3')     
			up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3]*2, name= 'upsampling')
        
			return tf.add_n([up_2,up_1])

	def fullNetwork(self, inputs, nFeat = 256, nModules = 1, outDim = 3):
		with tf.name_scope('preprocessing'):
			pad_1 = tf.pad(inputs, np.array([[0,0],[2,2],[2,2],[0,0]]))
			conv_1 = self.conv2d(pad_1, 64, kernel_size=6, strides=2) 
			print("conv_1 shape = " + str(conv_1.shape))
			#64
			norm_1 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu)
			#64 -》128 1
			conv_2 = self.bottleneck(norm_1, 128,1, 128)
			print("conv_2 shape = " + str(conv_2.shape))

			pol_1 = tf.contrib.layers.max_pool2d(conv_2, [2,2], [2,2], padding= 'VALID')

			conv_3 = self.bottleneck(pol_1, 128,0,0)
			print("conv_3 shape = " + str(conv_3.shape))
			conv_4 = self.bottleneck(conv_3, 256, 1, 256)
			print("conv_4 shape = " + str(conv_4.shape))

			hg = self.hourglass(conv_4, 1, 256)

			ll = hg
			ll = self.bottleneck(ll, 256, 0, 0)
			ll = self.conv2d(ll, 256, kernel_size=1, strides=1, pad='VALID')
			print("full newtwork: 1. ll shape = " + str(ll.shape))
			ll = tf.contrib.layers.batch_norm(ll, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu)
            
			# ll = tf.image.resize_nearest_neighbor(ll, tf.shape(ll)[1:3]*2, name= 'upsampling')
			# predict heatmaps
			tmp_out = self.conv2d(ll, outDim, kernel_size=1, strides=1, pad='VALID')
			print("full newtwork: 2. tmp_out shape = " + str(tmp_out.shape))
		return tmp_out

	def _argmax(self, tensor):
		""" ArgMax
		Args:
			tensor	: 2D - Tensor (Height x Width : 64x64 )
		Returns:
			arg		: Tuple of max position
		"""
		resh = tf.reshape(tensor, [-1])
		argmax = tf.arg_max(resh, 0)
		return (argmax // tensor.get_shape().as_list()[0], argmax % tensor.get_shape().as_list()[0])

	def _compute_err(self, u, v):
		""" Given 2 tensors compute the euclidean distance (L2) between maxima locations
		Args:
			u		: 2D - Tensor (Height x Width : 64x64 )
			v		: 2D - Tensor (Height x Width : 64x64 )
		Returns:
			(float) : Distance (in [0,1])
		"""
		u_x,u_y = self._argmax(u)
		v_x,v_y = self._argmax(v)
		return tf.divide(tf.sqrt(tf.square(tf.to_float(u_x - v_x)) + tf.square(tf.to_float(u_y - v_y))), tf.to_float(91))

	def _accur(self, pred, gtMap, num_image):
		""" Given a Prediction batch (pred) and a Ground Truth batch (gtMaps),
		returns one minus the mean distance.
		Args:
			pred		: Prediction Batch (shape = num_image x 64 x 64)
			gtMaps		: Ground Truth Batch (shape = num_image x 64 x 64)
			num_image 	: (int) Number of images in batch
		Returns:
			(float)
		"""
		err = tf.to_float(0)
		for i in range(num_image):
			err = tf.add(err, self._compute_err(pred[i], gtMap[i]))
		return tf.subtract(tf.to_float(1), err/num_image)

	def _accuracy_computation(self):
		""" Computes accuracy tensor
		"""
		self.imgAccur = [0,0,0]
		for i in range(self.joints): ## TODO joints=???
			self.imgAccur[i]=self._accur(self.output[:, self.nStack - 1, :, :,i], self.y[:, self.nStack - 1, :, :, i], self.batchSize)

	def _define_saver_summary(self, summary = True):
		""" Create Summary and Saver
		Args:
			logdir_train		: Path to train summary directory
			logdir_test		: Path to test summary directory
		"""
		if (self.logdir_train == None) or (self.logdir_test == None):
			raise ValueError('Train/Test directory not assigned')
		else:
			# with tf.device(self.cpu): TODO
			self.saver = tf.train.Saver()
			if summary:
				# with tf.device(self.cpu):## TODO
				self.train_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())
				self.test_summary = tf.summary.FileWriter(self.logdir_test)
					#self.weight_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())

	def _init_weight(self):
		w = [1] * 3 ## TODO outDim?
		# for i in range(3):
		# 	if np.array_equal(joints[i], [-1,-1]):
		# 		w[i] = 0
		# self.data_dict[name] = {'box' : box, 'joints' : joints, 'weights' : w}
		weight = np.asarray(w)
		for i in range(self.batchSize):
			self.train_weights[i] = weight
		# train_weights = np.zeros((batchSize, 3), np.float32)
		print('Session initialization')
		self.Session = tf.Session()
		t_start = time.time()
		self.Session.run(self.init)
		print('Sess initialized in ' + str(int(time.time() - t_start)) + ' sec.')

	def _train(self, nEpochs=10, epochiter=4, saveStep=2, validIter=10):
		# sess = tf.Session()
		# sess.run(init)
		# print('Session initilized')
		# saver = tf.train.Saver()
		# self.summary_train = tf.summary.FileWriter(train_dir , tf.get_default_graph())
		t_train_start = time.time()
		print('Start training')
		self.resume = {}
		self.resume['accur'] = []
		self.resume['loss'] = []
		self.resume['err'] = []
		with tf.name_scope('training'):
			for epoch in range(nEpochs):
				t_epoch_start = time.time()
				avg_cost = 0.
				cost=0.
				print('========Training Epoch: ', (epoch + 1))
				with tf.name_scope('epoch_' + str(epoch)):
					for i in range(epochiter):
						x_batch = np.zeros((self.batchSize,256,256,3))
						y_batch = np.zeros((self.batchSize, 64, 64,3))
						if i % saveStep == 0:
							# with tf.name_scope('batch_train'):
							x_batch[:,:,:,:] = x_train[i:(i+self.batchSize),:,:,:] # nparray type
							y_batch[:,:,:,:] = y_train[i:(i+self.batchSize),:,:,:]
							print('x_batch = ' + str(x_batch.shape) + 'y_batch = ' + str(y_batch.shape))
							_,c,summary = self.Session.run([self.train_rmsprop, self.loss, self.train_op], feed_dict={self.x: x_batch, self.y: y_batch})
						else:
							_,c, = self.Session.run([self.train_rmsprop, self.loss], feed_dict={self.x: x_batch, self.y: y_batch})
						cost += c
						avg_cost += c / (epochiter * self.batchSize)
					t_epoch_finish = time.time()
					print("Epoch:", (epoch + 1), '  avg_cost= ', "{:.9f}".format(avg_cost),' time_epoch=', str(t_epoch_finish-t_epoch_start))
					# save weight
					weight_summary = self.Session.run(self.weight_op, {self.x : x_batch, self.y: y_batch})
					self.train_summary.add_summary(weight_summary, epoch)
					self.train_summary.flush()
					print('Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done in ' + str(int(t_epoch_finish-t_epoch_start)) + ' sec.' + ' -avg_time/batch: ' + str(((t_epoch_finish-t_epoch_start)/epochiter))[:4] + ' sec.')
					with tf.name_scope('save'):
						self.saver.save(self.Session, os.path.join(os.getcwd(),str(self.name + '_' + str(epoch + 1))))
					self.resume['loss'].append(cost)

					# Validation Set
					# accuracy_array = np.array([0.0]*len(self.joint_accur))
					# for i in range(validIter):
					# 	img_valid, gt_valid, w_valid = next(self.generator)
					# 	accuracy_pred = self.Session.run(self.joint_accur, feed_dict = {self.img : img_valid, self.gtMaps: gt_valid})
					# 	accuracy_array += np.array(accuracy_pred, dtype = np.float32) / validIter
					# print('--Avg. Accuracy =', str((np.sum(accuracy_array) / len(accuracy_array)) * 100)[:6], '%' )
					# self.resume['accur'].append(accuracy_pred)
					# self.resume['err'].append(np.sum(accuracy_array) / len(accuracy_array))
					# valid_summary = self.Session.run(self.test_op, feed_dict={self.img : img_valid, self.gtMaps: gt_valid})
					# self.test_summary.add_summary(valid_summary, epoch)
					# self.test_summary.flush()
			# saver.save(sess, 'result/model.ckpt') #模型存储的文件夹
			print('Training Done')
			print('Resume:' + '\n' + '  Epochs: ' + str(nEpochs) + '\n' + '  n. Images: ' + str(nEpochs * epochSize * self.batchSize) )
			print('  Final Loss: ' + str(cost) + '\n' + '  Relative Loss: ' + str(100*self.resume['loss'][-1]/(self.resume['loss'][0] + 0.1)) + '%' )
			# print('  Relative Improvement: ' + str((self.resume['err'][-1] - self.resume['err'][0]) * 100) +'%')
			print('  Training Time: ' + str( datetime.timedelta(seconds=time.time() - t_start)))
		# t_end = time()
		# print('Training Done : ' + str(t_end - t_start)) 

	def training_init(self, nEpochs=10, epochiter=4, saveStep=2, dataset = None, load=None):
		with tf.name_scope('Session'):
			# with tf.device(self.cpu): ##TODO
			self._init_weight()
			self._define_saver_summary()
			if load is not None:
				self.saver.restore(self.Session, load)
					#try:
						#	self.saver.restore(self.Session, load)
					#except Exception:
						#	print('Loading Failed! (Check README file for further information)')
			self._train(nEpochs, epochiter, saveStep, validIter=10)



