import os
import cv2
from skimage import io
from tifffile import imread, imsave
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.append('../')
import segmentation_models as sm
from segmentation_models_v1 import Unet, Linknet, PSPNet, FPN, AtUnet

from helper_function import precision, recall, f1_score, iou_calculate, generate_folder
from sklearn.metrics import confusion_matrix

sm.set_framework('tf.keras')
import glob
from natsort import natsorted

def str2bool(value):
    return value.lower() == 'true'

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default = '0')
parser.add_argument("--model_file", type=str, default = 'model_list.txt')
parser.add_argument("--model_index", type=int, default = 0)
parser.add_argument("--save", type=str2bool, default = False)
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
model_names, epoch_list = [], []
with open(args.model_file, 'r+') as f:
		lines = f.readlines()
		for i, line in enumerate(lines):
				model_names.append(line.strip().split(',')[0])
				epoch_list.append(int(line.strip().split(',')[1]))


# model_names = ['single-net-Unet-bone-efficientnetb5-pre-True-epoch-2400-batch-4-lr-0.0001-dim-800-train-1100-rot-0-set-cell_cycle_1984_v2-ext-True-loss-focal+dice-up-upsampling-filters-256-red_factor-1.0-pyr_agg-sum-bk-1.0-fl_weight-4.0-fv-1']
index = args.model_index
model_name = model_names[index]
epoch = epoch_list[index]

print(model_name)
print('Epoch:{}'.format(epoch))
model_root_folder = '/data/viability2_models/'
nb_train_test = 200
model_folder = model_root_folder+model_name

## parse model name
splits = model_name.split('-')
if splits[0] == 'cell_cycle':
	dataset = 'cell_cycle2'
	val_dim = 992
else:
	dataset = 'live_dead'
	val_dim = 832

nb_filters = 256
upsample = 'upsampling'
ext_flag = False
feature_version = None
data_version = 0
for v in range(len(splits)):
	if splits[v]=='set':
		if splits[v+1] == 'cycle_736x752':
			dataset = splits[v+1]
			val_dim1, val_dim2 = 736, 768
			test_dim1, test_dim2 = 736, 768
			dim1, dim2 = 736, 752
		if 'viability' in splits[v+1]:
			dataset = splits[v+1]
			val_dim1, val_dim2 = 832, 832
			test_dim1, test_dim2 = 832, 832
			dim1, dim2 = 832, 832
	elif splits[v] == 'net':
		net_arch = splits[v+1]
	elif splits[v] == 'bone':
		backbone = splits[v+1]
	elif splits[v] == 'dv':
		data_version = int(splits[v+1])
	elif splits[v] == 'up':
		upsample = splits[v+1]
	elif splits[v] == 'filters':
		nb_filters = int(splits[v+1])
	elif splits[v] == 'ext':
		ext_flag = True if splits[v+1] =='True' else False
	elif splits[v] == 'fv':
		if not splits[v+1] == 'None':
			feature_version = int(splits[v+1]); print('feature version:{}'.format(feature_version))	

def read_txt(txt_dir):
    lines = []
    with open(txt_dir, 'r+') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines
			
DATA_DIR = '/data/datasets/{}'.format(dataset)

image_dir = DATA_DIR+'/{}'.format('images')
masks_dir = DATA_DIR+'/{}'.format('masks')

if data_version <1:
	train_fns = read_txt(DATA_DIR+'/train_list.txt')
	valid_fns = read_txt(DATA_DIR+'/valid_list.txt')
	test_fns = read_txt(DATA_DIR+'/test_list.txt')
else:
	train_fns = read_txt(DATA_DIR+'/train{}_list.txt'.format(data_version))
	valid_fns = read_txt(DATA_DIR+'/valid{}_list.txt'.format(data_version))
	test_fns = read_txt(DATA_DIR+'/test{}_list.txt'.format(data_version))
# classes for data loading and preprocessing
class Dataset:
	"""CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

	Args:
		images_dir (str): path to images folder
		masks_dir (str): path to segmentation masks folder
		class_values (list): values of classes to extract from segmentation mask
		augmentation (albumentations.Compose): data transfromation pipeline 
			(e.g. flip, scale, etc.)
		preprocessing (albumentations.Compose): data preprocessing 
			(e.g. noralization, shape manipulation, etc.)

	"""

	CLASSES = ['bk', 'live', 'inter', 'dead']

	def __init__(
			self, 
			images_dir, 
			masks_dir,
			file_names,
			classes=None, 
			augmentation=None, 
			preprocessing=None,
	):
		self.ids = file_names
		self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
		self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
	
		# convert str names to class values on masks
		self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
	
		self.augmentation = augmentation
		self.preprocessing = preprocessing

	def __getitem__(self, i):
	
		# read data
# 		image = cv2.imread(self.images_fps[i])
# 		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 		mask = cv2.imread(self.masks_fps[i], 0)
		image = io.imread(self.images_fps[i])
		mask = io.imread(self.masks_fps[i])
		# image to RGB
		image = np.uint8((image-image.min())*255/(image.max()-image.min()))
		image = np.stack([image,image,image],axis =-1)
#         print(np.unique(mask))
		# extract certain classes from mask (e.g. cars)
		masks = [(mask == v) for v in self.class_values]
#         print(self.class_values)
		mask = np.stack(masks, axis=-1).astype('float')
	
		# add background if mask is not binary
		if mask.shape[-1] != 1:
			background = 1 - mask.sum(axis=-1, keepdims=True)
			mask = np.concatenate((mask, background), axis=-1)
	
		# apply augmentations
		if self.augmentation:
			sample = self.augmentation(image=image, mask=mask)
			image, mask = sample['image'], sample['mask']
	
		# apply preprocessing
		if self.preprocessing:
			sample = self.preprocessing(image=image, mask=mask)
			image, mask = sample['image'], sample['mask']
		
		return image, mask
	
	def __len__(self):
		return len(self.ids)


class Dataloder(tf.keras.utils.Sequence):
	"""Load data from dataset and form batches

	Args:
		dataset: instance of Dataset class for image loading and preprocessing.
		batch_size: Integet number of images in batch.
		shuffle: Boolean, if `True` shuffle image indexes each epoch.
	"""

	def __init__(self, dataset, batch_size=1, shuffle=False):
		self.dataset = dataset
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.indexes = np.arange(len(dataset))

		self.on_epoch_end()

	def __getitem__(self, i):
	
		# collect batch data
		start = i * self.batch_size
		stop = (i + 1) * self.batch_size
		data = []
		for j in range(start, stop):
			data.append(self.dataset[j])
	
		# transpose list of lists
		batch = [np.stack(samples, axis=0) for samples in zip(*data)]
	
		return (batch[0], batch[1])

	def __len__(self):
		"""Denotes the number of batches per epoch"""
		return len(self.indexes) // self.batch_size

	def on_epoch_end(self):
		"""Callback function to shuffle indexes each epoch"""
		if self.shuffle:
			self.indexes = np.random.permutation(self.indexes)

import albumentations as A

def round_clip_0_1(x, **kwargs):
	return x.round().clip(0, 1)

def get_validation_augmentation(dim1 = 736, dim2 = 768):
	"""Add paddings to make image shape divisible by 32"""
	test_transform = [
		A.PadIfNeeded(dim1, dim2)
#         A.PadIfNeeded(384, 480)
	]
	return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
	"""Construct preprocessing transform

	Args:
		preprocessing_fn (callbale): data normalization function 
			(can be specific for each pretrained neural network)
	Return:
		transform: albumentations.Compose

	"""

	_transform = [
		A.Lambda(image=preprocessing_fn),
	]
	return A.Compose(_transform)

# network
best_weight = model_folder+'/best_model-{:03d}.h5'.format(epoch)
CLASSES = []
preprocess_input = sm.get_preprocessing(backbone)

#create model
# CLASSES = ['live', 'inter', 'dead']
CLASSES = ['live', 'dead']
n_classes = len(CLASSES) + 1
activation = 'softmax'
net_func = globals()[net_arch]
decoder_filters=(int(nb_filters),int(nb_filters/2), int(nb_filters/4), int(nb_filters/8), int(nb_filters/16))
model = net_func(backbone, classes=n_classes, encoder_weights = None, activation=activation,\
		decoder_block_type = upsample, feature_version = feature_version, decoder_filters = decoder_filters)
#model = net_func(backbone, classes=n_classes, encoder_weights = None, activation=activation,\
#		decoder_block_type = upsample, decoder_filters = decoder_filters)
# model = net_func(backbone, classes=n_classes, activation=activation)
model.summary()
#load best weights
model.load_weights(best_weight)
## save model
model.save(model_folder+'/ready_model.h5')

result_dir = os.path.join(model_folder,'eval_train_val_test')
generate_folder(result_dir)

# evaluate model
# subsets = ['val', 'train', 'test']
subsets = ['test', 'val']
# subset = subsets[2]
for subset in subsets:
# 	subset = subsets[0]
	print('processing subset :{}'.format(subset))
	if subset == 'test':
		img_fns = test_fns
	if subset == 'val':
		img_fns = valid_fns
	elif subset == 'train':
		img_fns = train_fns

	test_dataset = Dataset(
		image_dir, 
		masks_dir,
		file_names = img_fns,
		classes=CLASSES, 
		augmentation=get_validation_augmentation(test_dim1,test_dim2),
		preprocessing=get_preprocessing(preprocess_input),
	)

	test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)
	print(test_dataloader[0][0].shape)
	nb_test = nb_train_test if subset == 'train' else len(test_dataloader)

	random.seed(0)
	test_indices = random.sample(range(len(test_dataloader)),nb_test)

	# calculate the pixel-level classification performance
	pr_masks = model.predict(test_dataloader)
	pr_masks = pr_masks[test_indices,:]
	pr_maps = np.argmax(pr_masks,axis=-1)

	gt_masks = []
	for i in test_indices:
		_, gt_mask = test_dataset[i];gt_masks.append(gt_mask)
	gt_masks = np.stack(gt_masks);gt_maps = np.argmax(gt_masks,axis=-1)

	## reshape
# 	offset = 8
# 	pr_masks = pr_masks[:,:,offset:-offset]
# 	gt_masks = gt_masks[:,:,offset:-offset,:]
# 	print(pr_masks.shape); print(gt_masks.shape)

	## IoU and dice coefficient
	pr_save_mask = np.concatenate([pr_masks[:,:,:,-1:],pr_masks[:,:,:,:-1]], axis = -1)
	gt_save_mask = np.concatenate([gt_masks[:,:,:,-1:],gt_masks[:,:,:,:-1]], axis = -1)
	test_ids = [test_dataset.ids[index] for index in test_indices]
	## IoU and dice coefficient
	iou_classes, mIoU, dice_classes, mDice = iou_calculate(gt_save_mask, pr_save_mask)
	print('iou_classes: {:.4f},{:.4f},{:.4f}; mIoU: {:.4f}'.format(iou_classes[0],iou_classes[1],iou_classes[2], mIoU))
	print('dice_classes: {:.4f},{:.4f},{:.4f}; mDice: {:.4f}'.format(dice_classes[0],dice_classes[1],dice_classes[2], mDice))
	# save the prediction
	pr_save_map = np.argmax(pr_save_mask,axis=-1)
	gt_save_map = np.argmax(gt_save_mask,axis=-1)
# 	save = True
	save = args.save
	if save:
		pred_dir = os.path.join(model_folder, 'pred_{}'.format(subset))
		generate_folder(pred_dir)
		for pi, fn in enumerate(test_ids):
			imsave(pred_dir+'/{}'.format(fn), pr_save_map[pi,:,:])
		io.imsave(model_folder+'/pr_{}.png'.format(fn.split('.')[0]), pr_save_map[pi,:,:])
	
	y_true=gt_save_map.flatten(); y_pred = pr_save_map.flatten()
	cf_mat = confusion_matrix(y_true, y_pred)
	prec_scores = []; recall_scores = []; f1_scores = []; iou_scores=[]
	for i in range(cf_mat.shape[0]):
		prec_scores.append(precision(i,cf_mat))
		recall_scores.append(recall(i,cf_mat))
		f1_scores.append(f1_score(i,cf_mat))
	print('Precision:{:.4f},{:,.4f},{:.4f}'.format(prec_scores[0], prec_scores[1], prec_scores[2]))
	print('mean precision: {:.4f}\n'.format(np.mean(prec_scores)))
	print('Recall:{:.4f},{:,.4f},{:.4f}'.format(recall_scores[0], recall_scores[1], recall_scores[2]))
	print('mean recall:{:.4f}\n'.format(np.mean(recall_scores)))
	# f1 score
	print('f1-score (pixel):{:.4f},{:,.4f},{:.4f}'.format(f1_scores[0],f1_scores[1],f1_scores[2]))
	print('mean f1-score (pixel):{:.4f}\n'.format(np.mean(f1_scores)))

	with open(result_dir+'/{}_summary_{}.txt'.format(subset, epoch), 'w+') as f:
		# save iou and dice
		f.write('iou_classes: {:.4f},{:.4f},{:.4f}; mIoU: {:.4f}\n'.format(iou_classes[0],iou_classes[1],iou_classes[2], mIoU))
		f.write('dice_classes: {:.4f},{:.4f},{:.4f}; mDice: {:.4f}\n'.format(dice_classes[0],dice_classes[1],dice_classes[2], mDice))
		# save confusion matrix
		f.write('confusion matrix:\n')
		np.savetxt(f, cf_mat, fmt='%-7d')
		# save precision
		f.write('precision:{:.4f},{:,.4f},{:.4f}\n'.format(prec_scores[0], prec_scores[1], prec_scores[2]))
		f.write('mean precision: {:.4f}\n'.format(np.mean(prec_scores)))
		# save recall
		f.write('recall:{:.4f},{:,.4f},{:.4f}\n'.format(recall_scores[0], recall_scores[1], recall_scores[2]))
		f.write('mean recall:{:.4f}\n'.format(np.mean(recall_scores)))
		# save f1-score
		f.write('f1-score (pixel):{:.4f},{:,.4f},{:.4f}\n'.format(f1_scores[0],f1_scores[1],f1_scores[2]))
		f.write('mean f1-score (pixel):{:.4f}\n'.format(np.mean(f1_scores)))

	# confusion matrix 2
	print('Confusion matrix2:')
	cf_mat = cf_mat[1:,1:]
	print(cf_mat)
	prec_scores = []; recall_scores = []; f1_scores = []; iou_scores=[]
	for i in range(cf_mat.shape[0]):
		prec_scores.append(precision(i,cf_mat))
		recall_scores.append(recall(i,cf_mat))
		f1_scores.append(f1_score(i,cf_mat))
	print('Precision:{:.4f},{:,.4f}'.format(prec_scores[0], prec_scores[1]))
	print('mean precision: {:.4f}\n'.format(np.mean(prec_scores)))
	print('Recall:{:.4f},{:,.4f}'.format(recall_scores[0], recall_scores[1]))
	print('mean recall:{:.4f}\n'.format(np.mean(recall_scores)))
	# f1 score
	print('f1-score (pixel):{:.4f},{:,.4f}'.format(f1_scores[0],f1_scores[1]))
	print('mean f1-score (pixel):{:.4f}\n'.format(np.mean(f1_scores)))

	with open(result_dir+'/{}_summary_cell_{}.txt'.format(subset, epoch), 'w+') as f:
		# save iou and dice
# 		f.write('iou_classes: {:.4f},{:.4f},{:.4f},{:.4f}; mIoU: {:.4f}\n'.format(iou_classes[-1],iou_classes[0],iou_classes[1],iou_classes[2], mIoU))
# 		f.write('dice_classes: {:.4f},{:.4f},{:.4f},{:.4f}; mDice: {:.4f}\n'.format(dice_classes[-1],dice_classes[0],dice_classes[1],dice_classes[2], mDice))
		# save confusion matrix
		f.write('confusion matrix:\n')
		np.savetxt(f, cf_mat, fmt='%-7d')
		# save precision
		f.write('precision:{:.4f},{:,.4f}\n'.format(prec_scores[0], prec_scores[1]))
		f.write('mean precision: {:.4f}\n'.format(np.mean(prec_scores)))
		# save recall
		f.write('recall:{:.4f},{:,.4f}\n'.format(recall_scores[0], recall_scores[1]))
		f.write('mean recall:{:.4f}\n'.format(np.mean(recall_scores)))
		# save f1-score
		f.write('f1-score (pixel):{:.4f},{:,.4f}\n'.format(f1_scores[0],f1_scores[1]))
		f.write('mean f1-score (pixel):{:.4f}\n'.format(np.mean(f1_scores)))