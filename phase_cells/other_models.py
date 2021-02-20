import numpy as np 
import os

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K

K.set_image_data_format("channels_last")

def down_conv2(x, filters, n_convs = 2, bn = False):
		if bn:
				x = LeakyReLU(alpha=0.3)(BatchNormalization(axis = -1)(Conv2D(filters, 3, padding = 'same')(x)))
				x = LeakyReLU(alpha=0.3)(BatchNormalization(axis = -1)(Conv2D(filters, 3, padding = 'same')(x)))
		else:
				x = LeakyReLU(alpha=0.3)(Conv2D(filters, 3, padding = 'same')(x))
				x = LeakyReLU(alpha=0.3)(Conv2D(filters, 3, padding = 'same')(x))
		pool = MaxPooling2D(pool_size=(2, 2))(x)
		return pool, x
		
def up_conv2(x1, x2, filters, n_convs = 2, bn = False):
		x = concatenate([UpSampling2D(size = (2,2))(x1), x2], axis = -1)
		if bn:
				x = LeakyReLU(alpha=0.3)(BatchNormalization(axis = -1)(Conv2D(filters, 3, padding = 'same')(x)))
				x = LeakyReLU(alpha=0.3)(BatchNormalization(axis = -1)(Conv2D(filters, 3, padding = 'same')(x)))
		else:
				x = LeakyReLU(alpha=0.3)(Conv2D(filters, 3, padding = 'same')(x))
				x = LeakyReLU(alpha=0.3)(Conv2D(filters, 3, padding = 'same')(x))
		return x		

def unet1(input_size = (None,None,3), nb_classes = 1, base_filters = 16, bn = False, act_name = 'softmax'):
		inputs = Input(input_size)

		down1, add1 = down_conv2(inputs, base_filters, n_convs = 2, bn = bn)
		down2, add2 = down_conv2(down1, base_filters*2, n_convs = 2, bn = bn)
		down3, add3 = down_conv2(down2, base_filters*4, n_convs = 2, bn = bn)
		down4, add4 = down_conv2(down3, base_filters*8, n_convs = 2, bn = bn)
		down5, add5 = down_conv2(down4, base_filters*16, n_convs = 2, bn = bn)
		down6, add6 = down_conv2(down5, base_filters*32, n_convs = 2, bn = bn)
		if bn:
				down7 = LeakyReLU(alpha=0.3)(BatchNormalization(axis = -1)(Conv2D(base_filters*64, 3, padding = 'same')(down6)))
				down7 = LeakyReLU(alpha=0.3)(BatchNormalization(axis = -1)(Conv2D(base_filters*64, 3, padding = 'same')(down7)))
		else:
				down7 = LeakyReLU(alpha=0.3)(BatchNormalization(axis = -1)(Conv2D(base_filters*64, 3, padding = 'same')(down6)))
				down7 = LeakyReLU(alpha=0.3)(BatchNormalization(axis = -1)(Conv2D(base_filters*64, 3, padding = 'same')(down7)))				
		up8 = up_conv2(down7, add6, base_filters*32, n_convs = 2, bn = bn)
		up9 = up_conv2(up8, add5, base_filters*16, n_convs = 2, bn = bn)
		up10 = up_conv2(up9, add4,base_filters*8, n_convs = 2, bn = bn)
		up11 = up_conv2(up10, add3, base_filters*4, n_convs = 2, bn = bn)
		up12 = up_conv2(up11, add2,  base_filters*2,n_convs = 2, bn = bn)
		up13 = up_conv2(up12, add1, base_filters, n_convs = 2, bn = bn)

		output_layer = Conv2D(nb_classes, 1, activation = act_name)(up13)

		model = Model(inputs = inputs, outputs = output_layer)

		return model 

# model = unet1()