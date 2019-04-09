import time
import tensorflow as tf
import numpy as np
import datetime
import os

# ------------- 
class HourglassModel():
	def __init__(self, nFeat = 256, nStack = 4, nLow = 4, outputDim = 2, batch_size = 16, attention = False, training = True, drop_rate = 0.2, lear_rate = 2.5e-4, decay = 0.96, decay_step = 2000, dataset = None, w_summary = True, logdir_train = None, logdir_test = None,model_dir = None, tiny = True, w_loss = False, modif = True, name = 'tiny_hourglass', gpu=False):
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
		self.outDim = outputDim
		self.batchSize = batch_size
		self.attention = attention
		self.training = training
		self.w_summary = w_summary
		self.tiny = tiny
		self.dropout_rate = drop_rate
		self.learning_rate = lear_rate
		self.decay = decay
		self.decay_step = decay_step
		self.nLow = nLow
		self.modif = modif
		self.dataset = dataset
		self.cpu = '/cpu:0'
		if gpu:
			self.gpu = '/gpu:0'
		else:
			self.gpu = '/cpu:0'
		self.logdir_train = logdir_train
		self.logdir_test = logdir_test
		self.model_dir = model_dir
		self.w_loss = w_loss
		if outputDim == 3:
			self.direction = ['gazeR','gazeG','gazeB']
			self.imgAccur = [0,0,0]
		elif outputDim == 2:
			self.direction = ['left','right']
			self.imgAccur = [0,0]
		self.avg_cost = 0
		self.dir_accur=[0.0]
		self.name = name
		print("Init model done.")

	def set_label(self,y): 
		""" Returns ground truth (Placeholder) Tensor
			Shape: (None, nbStacks, 64, 64, outputDim)
			Type : tf.float32
			Warning: Be sure to build the model first
		"""
		self.y = y

	def get_input(self):
		""" Returns Input (Placeholder) Tensor
			Shape: (None,256,256,3)
			Type : tf.float32
			Warning: Be sure to build the model first
		"""
		return self.x

	def get_output(self):
		""" Returns Output Tensor
			Shape: (None, nbStacks, 64, 64, outputDim)
			Type : tf.float32
			Warning: Be sure to build the model first
		"""
		return self.output

	def get_label(self): 
		""" Returns ground truth (Placeholder) Tensor
			Shape: (None, nbStacks, 64, 64, outputDim)
			Type : tf.float32
			Warning: Be sure to build the model first
		"""
		return self.y

	def get_loss(self):
		""" Returns Loss Tensor
			Shape: (1,)
			Type : tf.float32
			Warning: Be sure to build the model first
		"""
		return self.loss

	def get_saver(self):
		""" Returns Saver
		/!\ USE ONLY IF YOU KNOW WHAT YOU ARE DOING
		Warning:
			Be sure to build the model first
		"""
		return self.saver

	def get_accuracy(self):
		return self.imgAccur

	# ==================== Hourglass Network ========================

	def generate_model(self):
		# create complete graph
		print('Creating Model')
		t_start = time.time()
		with tf.device(self.gpu):
			with tf.name_scope('inputs'):
				self.x = tf.placeholder(tf.float32, [None, 128, 128, 3]) # 'None' cuz define at run time
				# self.x = tf.placeholder(tf.float32, [None, 256, 256, 3]) # 'None' cuz define at run time
				if self.w_loss: #TODO what is does?
					self.weights = tf.placeholder(dtype = tf.float32, shape = (None, self.outDim))
				# Shape Ground Truth Map: batchSize x nStack x 64 x 64 x outDim
				self.y = tf.placeholder(dtype = tf.float32, shape = (None, self.nStack, 64, 64, self.outDim))
				print('--Inputs : Done')
			with tf.name_scope('model'):
				self.output = self.fullNetwork(self.x, self.nFeat, self.nStack, self.outDim)
				print('--Graph : Done')
			with tf.name_scope('loss'):
				self.loss = tf.reduce_mean(tf.losses.huber_loss(labels=self.y, predictions=self.output), name = 'huber_loss') 
				# self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels= self.y), name = 'cross_entropy_loss') 
				print('--Loss : Done')
		with tf.device(self.cpu):
			with tf.name_scope('accuracy'):
				self._accuracy_computation()
				accurTime = time.time()
				print('--Accuracy : Done')
			with tf.name_scope('steps'):
				self.train_step = tf.Variable(0, name = 'global_step', trainable= False)
			with tf.name_scope('lr'):
				self.lr = tf.train.exponential_decay(self.learning_rate, self.train_step, self.decay_step, self.decay, staircase= True, name= 'learning_rate')
				print('--LearngingRate : Done')
		with tf.device(self.gpu):
			with tf.name_scope('rmsprop_optimizer'):
				self.rmsprop = tf.train.RMSPropOptimizer(learning_rate= self.learning_rate)
				print('--Optim : Done')
			with tf.name_scope('minimize'):
				self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
				with tf.control_dependencies(self.update_ops):
					self.train_rmsprop = self.rmsprop.minimize(self.loss, self.train_step)
				print('--Minimizer : Done')
		self.init = tf.global_variables_initializer()
		print('--Init : Done')
		with tf.device(self.cpu):
			with tf.name_scope('training'):
				tf.summary.scalar('loss', self.loss, collections = ['train'])
				tf.summary.scalar('learning_rate', self.learning_rate, collections = ['train'])
			with tf.name_scope('summary'):
				for i in range(len(self.direction)):
					tf.summary.scalar(self.direction[i], self.imgAccur[i], collections = ['train', 'test'])
			# tf.summary.scalar('imgAccuracy', self.imgAccur[], collections = ['train', 'test'])
		self.train_op = tf.summary.merge_all('train')
		self.test_op = tf.summary.merge_all('test')
		self.weight_op = tf.summary.merge_all('weight')
		print('Model generation: ' + str(time.time()- t_start))
		del t_start

	def restore(self, load = None):
		with tf.name_scope('Session'):
			with tf.device(self.gpu):
				# with tf.device(self.cpu):
				self.Session = tf.Session()
				self._define_saver_summary(summary = False)
				if load is not None:
					print('Loading Trained Model')
					t = time.time()
					self.saver.restore(self.Session, load)
					print('Model Loaded (', time.time() - t,' sec.)')
				else:
					print('Please give a Model in args')

	def conv2d(self, inputs, filters, kernel_size = 1, strides = 1, pad = 'VALID', name = None):
		with tf.name_scope(name):
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
		
			if n > 1:
				low_2 = self.hourglass(low_1, n-1, numOut, name='low2') ##### change low_1?
			else:
				low_2 = self.bottleneck(low_1, numOut, name='low2') #####low1/2?

			low_3 = low_2
			low_3 = self.bottleneck(low_2, numOut, name = 'low3')	 
			up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3]*2, name= 'upsampling')
		
			return tf.add_n([up_2,up_1], name='out_hg')

	def fullNetwork(self, inputs, nFeat = 256, nStack = 4, outDim = 2):
		with tf.name_scope('model'):
			with tf.name_scope('preprocessing'):
				pad_1 = tf.pad(inputs, np.array([[0,0],[3,3],[3,3],[0,0]]), name='pad_1')
				# pad_1 = inputs
				conv_1 = self.conv2d(pad_1, 64, kernel_size=6, strides=1) 
				# print("conv_1 shape = " + str(conv_1.shape))
				#
				norm_1 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu)
				#
				conv_2 = self.bottleneck(norm_1, int(nFeat/2), 1, int(nFeat/2), name = 'conv_2')

				pol_1 = tf.contrib.layers.max_pool2d(conv_2, [2,2], [2,2], padding= 'VALID')

				conv_3 = self.bottleneck(pol_1, int(nFeat/2),0,0, name = 'conv_3')
				# print("conv_3 shape = " + str(conv_3.shape))
				conv_4 = self.bottleneck(conv_3, nFeat, 1, nFeat, name = 'conv_4')
				# print("conv_4 shape = " + str(conv_4.shape))
				previous = conv_4

			out = [None] * nStack
			
			with tf.name_scope('stacks'):
				for i in range(nStack):
					# print('nStack = '+str(i))
					hg = self.hourglass(conv_4, self.nLow, nFeat)
					ll = hg
					ll = self.bottleneck(ll, nFeat, 0, 0)
					ll = self.conv2d(ll, nFeat, kernel_size=1, strides=1, pad='VALID')
					ll = tf.contrib.layers.batch_norm(ll, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu)
					# predict heatmaps
					tmp_out = self.conv2d(ll, outDim, kernel_size=1, strides=1, pad='VALID')

					out[i] = tmp_out

					if i < nStack - 1:
						ll = self.conv2d(ll, nFeat, kernel_size=1, strides=1, pad='VALID')
						tmp_out_ = self.conv2d(tmp_out, nFeat, kernel_size=1, strides=1, pad='VALID')
						previous = previous + ll + tmp_out_

			return tf.stack(out, axis= 1 , name = 'final_output')

	# ==================== Accuracy Calculation ========================

	def _argmax(self, tensor):
		""" ArgMax
		Args:
			tensor	: 2D - Tensor (Height x Width : 64x64 )
		Returns:
			arg		: Tuple of max position NOTE y,x
		"""
		resh = tf.reshape(tensor, [-1]) # flatten into 1-D
		argmax = tf.argmax(resh,0)
		return (argmax // tensor.get_shape().as_list()[0], argmax % tensor.get_shape().as_list()[0])

	def _compute_err(self, u, v=None, pts=None):
		""" Given 2 tensors compute the euclidean distance (L2) between maxima locations
		Args:
			u		: 2D - Tensor (Height x Width : 64x64 )
			v		: 2D - Tensor (Height x Width : 64x64 )
		Returns:
			(float) : Distance (in [0,1])
		"""
		# 91=np.sqrt(64*64+64*64) max distance
		u_y,u_x = self._argmax(u)
		if pts is None:
			v_y,v_x = self._argmax(v)
		else:
			v_y = pts[1]
			v_x = pts[0]
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
		# left_err = tf.to_float(0)
		# right_err = tf.to_float(0)
		# for i in range(num_image):
		# 	left_tmp,right_tmp = self._compute_err(pred[i], gtMap[i])
		# 	left_err = tf.add(left_err, left_tmp)
		# 	right_err = tf.add(right_err, right_tmp)
		# return (tf.subtract(tf.to_float(1), left_err/num_image),tf.subtract(tf.to_float(1), right_err/num_image))

	def _accuracy_computation(self):
		""" Computes accuracy tensor
		"""
		for i in range(len(self.direction)):
			self.imgAccur[i]=self._accur(self.output[:, self.nStack - 1, :, :,i], self.y[:, self.nStack - 1, :, :, i], self.batchSize)
	
	def confidence(self, img, pts, eye):
		# print(type(left_x))
		'''Single img confidence 64*64
			pts : [x,y] center point coordinate
			eye : 0 - left 1 - right
		'''
		u_y,u_x = self._argmax(img)
		# 15.56 = np.sqrt(11*11+11*11) max distance from 1 class direction 
		err = tf.divide(tf.sqrt(tf.square(tf.to_float(u_x - pts[0][0])) + tf.square(tf.to_float(u_y - pts[0][1]))), tf.to_float(15.56))
		confidence = tf.subtract(tf.to_float(1), err)

	def pred_direction(self, img, version=2):
		''' 64*64*2 img
		'''
		tmp = [[0,0],[0,0]] # store max confidence [0][] left max [1][] right max
		direction = [0,0]
		total_l = total_r = 0
		for i in range(10):
			gt_pts = self.dataset.get_center_coord(i,version)
			l_confidence = self.confidence(img[:,:,0],gt_pts[0],0)
			r_confidence = self.confidence(img[:,:,1],gt_pts[1],1)
			l_confidence, r_confidence = tf.Session().run([l_confidence, r_confidence])
			if l_confidence > tmp[0][0]:
				tmp[0][0] = l_confidence
				tmp[0][1] = r_confidence
				direction[0] = i
			if r_confidence > tmp[1][1]:
				tmp[1][0] = l_confidence
				tmp[1][1] = r_confidence
				direction[1] = i
		if tmp[0][0] > tmp[1][1]: # left max confidence > right max confidence
			d = direction[0]
			l_confidence = tmp[0][0]
			r_confidence = tmp[0][1]
		else:
			d = direction[1]
			l_confidence = tmp[1][0]
			r_confidence = tmp[1][1]
		return (d,l_confidence,r_confidence)
	
	def dir_accuracy(self,pred,gt_dir,num_image):
		err_count = 0
		for i in range(num_image):
			direction,_,_ = self.pred_direction(pred[i])
			if direction != gt_dir[i]:
				err_count +=1
		self.dir_accur = tf.subtract(tf.to_float(1), err_count/num_image)
		return self.dir_accur

	def _define_saver_summary(self, summary = True):
		""" Create Summary and Saver
		Args:
			logdir_train		: Path to train summary directory
			logdir_test		: Path to test summary directory
		"""
		if (self.logdir_train == None) or (self.logdir_test == None):
			raise ValueError('Train/Test directory not assigned')
		else:
			with tf.device(self.cpu): 
				self.saver = tf.train.Saver()
			if summary:
				with tf.device(self.cpu):## TODO
					self.train_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())
					self.test_summary = tf.summary.FileWriter(self.logdir_test)
				# self.weight_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())

	def _init_weight(self):
		print('Session initialization')
		self.Session = tf.Session()
		t_start = time.time()
		self.Session.run(self.init)
		print('Sess initialized in ' + str(int(time.time() - t_start)) + ' sec.')

	def _train(self, nEpochs=3, epochiter=2, saveStep=2, validIter=10):
		with tf.name_scope('Train'):
			self.generator = self.dataset._aux_generator(self.batchSize, self.nStack, normalize = True, sample_set = 'train')
			self.valid_gen = self.dataset._aux_generator(self.batchSize, self.nStack, normalize = True, sample_set = 'valid')
			t_train_start = time.time()
			print('Start training')
			self.resume = {}
			self.resume['accur'] = []
			self.resume['loss'] = []
			self.resume['err'] = []
			
			for epoch in range(nEpochs):
				t_epoch_start = time.time()
				avg_cost = 0.
				cost=0.
				print('========Training Epoch: ', str(epoch),'/',(nEpochs))
				# Training Set
				with tf.name_scope('epoch_' + str(epoch)):
					for i in range(epochiter):
						x_batch, y_batch, w_batch, _ = next(self.generator)

						if i % saveStep == 0:
							if self.w_loss:
								_, c, summary = self.Session.run([self.train_rmsprop, self.loss, self.train_op], feed_dict = {self.x : x_batch, self.y: y_batch, self.weights: w_batch})
							else:
								_, c, summary = self.Session.run([self.train_rmsprop, self.loss, self.train_op], feed_dict = {self.x : x_batch, self.y: y_batch})
							# Save summary (Loss + Accuracy)
							self.train_summary.add_summary(summary, epoch*epochiter + i)
							self.train_summary.flush()
						else:
							if self.w_loss:
								_, c, = self.Session.run([self.train_rmsprop, self.loss], feed_dict = {self.x : x_batch, self.y: y_batch, self.weights: w_batch})
							else:	
								_, c, = self.Session.run([self.train_rmsprop, self.loss], feed_dict = {self.x : x_batch, self.y: y_batch})
						cost += c
						avg_cost += c/epochiter

					t_epoch_finish = time.time()
					#Save Weight (axis = epoch)
					if self.w_loss:
						weight_summary = self.Session.run(self.weight_op, {self.x : x_batch, self.y: y_batch, self.weights: w_batch})
					else :
						weight_summary = self.Session.run(self.weight_op, {self.x : x_batch, self.y: y_batch})
					self.train_summary.add_summary(weight_summary, epoch)
					self.train_summary.flush()

					print("Epoch:", (epoch + 1), '  avg_cost= ', "{:.9f}".format(avg_cost),' time_epoch=', str(t_epoch_finish-t_epoch_start))
				
				with tf.name_scope('save'):
					# TODO specify model_dir in config
					# model_dir = os.path.join(os.getcwd(),str(self.name + '_' + str(epoch + 1)))
					# model_dir = 'result/model'
					model_name = os.path.join(self.model_dir,str(self.name + '_' + str(epoch + 1)))
					self.saver.save(self.Session, model_name)
					# self.saver.save(self.Session, os.path.join(os.getcwd(),str(self.name + '_' + str(epoch + 1))))
				self.resume['loss'].append(cost)

				# Validation Set
				accuracy_array = np.array([0.0]*len(self.imgAccur))
				# dir_accur_array = np.array([0.0])
				for i in range(validIter):
					img_valid, gt_valid, w_valid, d_valid = next(self.generator)
					accuracy_pred = self.Session.run(self.imgAccur, feed_dict = {self.x : img_valid, self.y: gt_valid})
					accuracy_array += np.array(accuracy_pred, dtype = np.float32) / validIter
				print('--Avg. Accuracy =', str((np.sum(accuracy_array) / len(accuracy_array)) * 100)[:6], '%' )
				self.resume['accur'].append(accuracy_pred)
				self.resume['err'].append(np.sum(accuracy_array) / len(accuracy_array)) # TODO why
				valid_summary = self.Session.run(self.test_op, feed_dict={self.x : img_valid, self.y: gt_valid})
				self.test_summary.add_summary(valid_summary, epoch)
				self.test_summary.flush()

			print('Training Done')
			print('Resume:' + '\n' + '  Epochs: ' + str(nEpochs) + '\n' + '  n. Images: ' + str(nEpochs * epochiter * self.batchSize) )
			print('  Final Loss: ' + str(cost) + '\n' + '  Relative Loss: ' + str(100*self.resume['loss'][-1]/(self.resume['loss'][0] + 0.1)) + '%' )
			print('  Relative Improvement: ' + str((self.resume['err'][-1] - self.resume['err'][0]) * 100) +'%')
			print('  Training Time: ' + str( datetime.timedelta(seconds=time.time() - t_train_start)))

	def training_init(self, nEpochs=10, epochiter=1000, saveStep=500, dataset = None, load=None):
		with tf.name_scope('Session'):
			with tf.device(self.gpu):
				self._init_weight()
				self._define_saver_summary()
				if load is not None:
					self.saver.restore(self.Session, load)
				self._train(nEpochs, epochiter, saveStep, validIter=10)

	def add_train(self,img,img_path=None,direction=-1):
		print ('Adding img to train set')
		self.dataset.write_txt(img_path,direction=direction)

		# dataset. # generate gt hm according to direction
		# save train hm to img_dir gt_dir
		# update train table
		# print train number (showing increasing)

	def decide_add(self,img,l_confidence,r_confidence,img_path=None,direction=0):
		if l_confidence>0.5 and r_confidence > 0.5:
			self.add_train(img,img_path,direction)