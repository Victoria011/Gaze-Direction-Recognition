import numpy as np
import tensorflow as tf
from hourglass import HourglassModel

# training 
model = HourglassModel(nFeat=256, nStack=1, nModules=1, nLow=1, outputDim=3, batch_size=32, lear_rate = 2.5e-4, decay = 0.96, decay_step = 2000, dataset=None, training=True, w_loss=False, name="hourglass")
model.generate_model()
model.training_init(nEpochs=10, epochiter=4, saveStep=2, dataset = None)