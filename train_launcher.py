import configparser
from dataGen import DataGenerator
from hourglass import HourglassModel
import tensorflow as tf
import numpy as np
import dlib
import cv2
# from test_hg import HourglassModel

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

if __name__ == '__main__':
	# process config
	print('--Parsing Config File')
	params = process_config('config.cfg')
	print('nfeats=',params['nfeats'],' nstacks=',params['nstacks'],' output_dim=',params['output_dim'],' batch_size=',params['batch_size'])
	print('dropout_rate=',params['dropout_rate'], ' learning_rate=',params['learning_rate'], ' learning_rate_decay=',params['learning_rate_decay'], ' decay_step=',params['decay_step'])
	print('nepochs=',params['nepochs'], ' epoch_size=',params['epoch_size'], ' saver_step=',params['saver_step'])

	# load GazeDetector
	# gazedetector = Gazedetect.Gazedetector('shape_predictor_68_face_landmarks.dat',enable_cuda=True)

	print('--Creating Dataset')
	dataset = DataGenerator(params['img_dir'],params['gt_dir'],params['training_txt_file'])
	dataset.generate_set(rand = True)

	# training 
	model = HourglassModel(nFeat=params['nfeats'], nStack=params['nstacks'], nLow=params['nlow'], outputDim=params['output_dim'], batch_size=params['batch_size'], attention = params['mcam'],training=True, drop_rate= params['dropout_rate'], lear_rate=params['learning_rate'], decay=params['learning_rate_decay'], decay_step=params['decay_step'], dataset=dataset, logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'], model_dir=params['model_dir'], tiny= params['tiny'], w_loss=params['weighted_loss'], modif=False, name=params['name'], gpu=params['gpu'])
	model.generate_model()
	model.training_init(nEpochs=params['nepochs'], epochiter=params['epoch_size'], saveStep=params['saver_step'], dataset = None)

