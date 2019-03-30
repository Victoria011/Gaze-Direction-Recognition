import numpy as np
import tensorflow as tf
from hourglass import HourglassModel
from dataGen import DataGenerator
import configparser

# train launcher

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

print('--Creating Dataset')
# train data generate
# img_dir = 'Dissertation/local_Dataset/video_frame/front'
dataset = DataGenerator(params['img_dir'],params['gt_dir'],params['training_txt_file'])
dataset.generate_set()
# train_set = DataGenerator.get_train()
# valid_set = DataGenerator.get_valid()
# batch_set = DataGenerator.get_batch(4,'train')

# training 
model = HourglassModel(nFeat=params['nfeats'], nStack=params['nstacks'], nLow=params['nlow'], outputDim=params['output_dim'], batch_size=params['batch_size'], attention = params['mcam'],training=True, drop_rate= params['dropout_rate'], lear_rate=params['learning_rate'], decay=params['learning_rate_decay'], decay_step=params['decay_step'], dataset=dataset, logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'], tiny= params['tiny'], w_loss=params['weighted_loss'], modif=False, name=params['name'])
# model = HourglassModel(nFeat=256, nStack=4, nModules=1, nLow=1, outputDim=3, batch_size=32, lear_rate = 2.5e-4, decay = 0.96, decay_step = 2000, dataset=None, training=True, w_loss=False, name="hourglass")
model.generate_model()
# model.training_init(nEpochs=params['nepochs'], epochiter=params['epoch_size'], saveStep=params['saver_step'], dataset = None)
# model.restore(tf.train.latest_checkpoint('./'))

# image_value = tf.read_file('Dataset/test_x/frame1.png')
# img = tf.image.decode_png(image_value, channels=3)
# img = tf.expand_dims(img, 0)
# img = img.eval(session=sess)
# out = sess.run(model.output, feed_dict={model.img : img})
# calculate max point x y coord

# decide gaze direction

# calculate direction accuracy

