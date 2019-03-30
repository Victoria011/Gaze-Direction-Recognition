import configparser
import Gazedetect
from dataGen import DataGenerator
from hourglass import HourglassModel
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import dlib

def process_config(conf_file):
	"""
	"""
	params = {}
	config = configparser.ConfigParser()
	config.read(conf_file)
	for section in config.sections():
		if section == 'DataSetHG':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Network':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Train':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Validation':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Saver':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
	return params

# process config
print('--Parsing Config File')
params = process_config('config.cfg')

# load GazeDetector
gazedetector = Gazedetect.Gazedetector('Dissertation/Action-Units-Heatmaps/shape_predictor_68_face_landmarks.dat',enable_cuda=False)

print('--Creating Dataset')
# train data generate
# img_dir = 'Dissertation/local_Dataset/video_frame/front'
dataset = DataGenerator(params['img_dir'],params['gt_dir'],params['training_txt_file'])
dataset.generate_set()
# train_set = DataGenerator.get_train()
# valid_set = DataGenerator.get_valid()
# batch_set = DataGenerator.get_batch(4,'train')

# training 
model = HourglassModel(nFeat=params['nfeats'], nStack=params['nstacks'], nLow=params['nlow'], outputDim=params['output_dim'], batch_size=params['batch_size'], attention = params['mcam'],training=True, drop_rate= params['dropout_rate'], lear_rate=params['learning_rate'], decay=params['learning_rate_decay'], decay_step=params['decay_step'], dataset=dataset, logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'], model_dir=params['model_dir'], tiny= params['tiny'], w_loss=params['weighted_loss'], modif=False, name=params['name'])
model.generate_model()
# model.training_init(nEpochs=params['nepochs'], epochiter=params['epoch_size'], saveStep=params['saver_step'], dataset = None)

sess=tf.Session()
sess.run(model.init)
model.restore(tf.train.latest_checkpoint('./result/model/'))

image_path = 'Dissertation/local_Dataset/video_frame/front/frame1.png'
# img = tf.image.decode_png(image_path, channels=3)
# transform img
img = dlib.load_rgb_image(image_path)
# pts = gazedetector.getPts(img)
# img = gazedetector._transform(img, pts)
img = tf.expand_dims(img, 0)
img = img.eval(session=sess)
# out, loss, accuracy = sess.run([model.output, model.loss, model.imgAccur], feed_dict={model.x : img})
out = sess.run(model.output, feed_dict={model.x : img})
# out, accuracy, loss = sess.run([model.output, model.imgAccur, model.loss], feed_dict={model.x : img})
# loss = model.get_loss()
# accuracy = model.get_accuracy()

# print(loss)
# print(accuracy)
num = out.shape[0]
for i in range(num):
	filename = 'Dissertation/local_Dataset/video_frame/front/output/frame'+str(i)+'.png'
	plt.imsave(filename, out[i][params['nstacks']-1].astype(np.uint8))

# # calculate confidence if > threshold add to train
# confidence = accuracy[0]
# if confidence > 0.5:
# 	print('Confident')
# 	# transformation
# 	# img = gazedetector._transform()
# 	# generate ground truth heatmap

# 	# write to txt file

# 	# add the train
# 	dataset.add_to_train(img)
# else:
# 	print('Not confident')
# calculate max point x y coord

# decide gaze direction

# calculate direction accuracy

