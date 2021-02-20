import os
import sys
sys.path.append('../')
import segmentation_models as sm
from segmentation_models import Unet

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

backbone = 'efficientnetb0'
model = Unet(backbone, input_shape = (512,512,3))
network_layers = model.layers
feature_layers = ['block6a_expand_activation', 'block4a_expand_activation','block3a_expand_activation', 'block2a_expand_activation']
with open('network_{}.txt'.format(backbone), 'w+') as f:
	for layer in network_layers:
		f.write('{}: {}\n'.format(layer.name, layer.output.get_shape()))
		if layer.name in feature_layers:
			f.write('\nFeature extansion ---{}\n'.format(layer.name))
