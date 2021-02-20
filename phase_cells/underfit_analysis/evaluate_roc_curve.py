import os
import cv2
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.append('../')
import segmentation_models as sm
from segmentation_models import Unet, Linknet, PSPNet, FPN

from helper_function import precision, recall, f1_score, iou_calculate, generate_folder
from sklearn.metrics import confusion_matrix

sm.set_framework('tf.keras')
import glob
from natsort import natsorted

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_root_folder = '/data/models/report_results/'
nb_train_test = 200

model_names = ['cellcycle-net-Unet-bone-efficientnetb2-pre-True-epoch-120-batch-3-lr-0.0005-down-True-dim-1024-train-1100-bk-0.5-rot-0-set-1984_v2-ext-True-fact-1-loss-focal+dice']
index = 0
model_name = model_names[index]
print(model_name)
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
for v in range(len(splits)):
	if splits[v]=='set':
		if splits[v+1] == 'dead':
			dataset = 'live_'+splits[v+1]
		if splits[v+1] == '1664':
			dataset = 'live_dead_'+splits[v+1]
			val_dim = 1664
		if splits[v+1] == '1984':
			dataset = 'cell_cycle_'+splits[v+1]
			val_dim = 1984
		if splits[v+1] == '1984_v2':
			dataset = 'cell_cycle_'+splits[v+1]
			val_dim = 1984
		if splits[v+1] == 'cell_cycle_1984_v2':
			dataset = splits[v+1]
			val_dim = 1984
	elif splits[v] == 'net':
		net_arch = splits[v+1]
	elif splits[v] == 'bone':
		backbone = splits[v+1]
	elif splits[v] == 'up':
		upsample = splits[v+1]
	elif splits[v] == 'filters':
		nb_filters = int(splits[v+1])
	elif splits[v] == 'ext':
		ext_flag = True if splits[v+1] =='True' else False
			
DATA_DIR = '/data/datasets/{}'.format(dataset)
x_train_dir = os.path.join(DATA_DIR, 'train_images') if not ext_flag else os.path.join(DATA_DIR, 'ext_train_images')
y_train_dir = os.path.join(DATA_DIR, 'train_masks')	 if not ext_flag else os.path.join(DATA_DIR, 'ext_train_masks')

x_valid_dir = os.path.join(DATA_DIR, 'val_images')
y_valid_dir = os.path.join(DATA_DIR, 'val_masks')

x_test_dir = os.path.join(DATA_DIR, 'test_images')
y_test_dir = os.path.join(DATA_DIR, 'test_masks')

if dataset == 'live_dead':
	x_train_dir +='2'; x_valid_dir+= '2'; x_test_dir+='2'

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
			classes=None, 
			augmentation=None, 
			preprocessing=None,
	):
		self.ids = natsorted(os.listdir(images_dir))
		self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
		self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
	
		# convert str names to class values on masks
		self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
	
		self.augmentation = augmentation
		self.preprocessing = preprocessing

	def __getitem__(self, i):
	
		# read data
		image = cv2.imread(self.images_fps[i])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(self.masks_fps[i], 0)
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

def get_validation_augmentation(dim = 832):
	"""Add paddings to make image shape divisible by 32"""
	test_transform = [
		A.PadIfNeeded(dim, dim)
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
best_weight = model_folder+'/best_model.h5'
CLASSES = []
preprocess_input = sm.get_preprocessing(backbone)

#create model
CLASSES = ['live', 'inter', 'dead']
n_classes = len(CLASSES) + 1
activation = 'softmax'
net_func = globals()[net_arch]
decoder_filters=(int(nb_filters),int(nb_filters/2), int(nb_filters/4), int(nb_filters/8), int(nb_filters/16))
model = net_func(backbone, classes=n_classes, activation=activation,\
		decoder_block_type = upsample, decoder_filters = decoder_filters)
# model = net_func(backbone, classes=n_classes, activation=activation)
model.summary()
#load best weights
model.load_weights(best_weight)
## save model
model.save(model_folder+'/ready_model.h5')

# define optomizer
optim = tf.keras.optimizers.Adam(0.001)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 1, 1, 0.5]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)

# result_dir 
result_dir = os.path.join(model_folder,'eval_train_val_test')
generate_folder(result_dir)

# evaluate model
# subsets = ['val', 'train', 'test']
# subsets = ['test','train']
# subset = subsets[2]
subset = 'test'
# for subset in subsets:
print('processing subset :{}'.format(subset))
if subset == 'val':
	x_test_dir = x_valid_dir; y_test_dir = y_valid_dir
elif subset == 'train':
	x_test_dir = x_train_dir; y_test_dir = y_train_dir

test_dataset = Dataset(
	x_test_dir, 
	y_test_dir, 
	classes=CLASSES, 
	augmentation=get_validation_augmentation(val_dim),
	preprocessing=get_preprocessing(preprocess_input),
)

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

nb_test = nb_train_test if subset == 'train' else len(test_dataloader)

random.seed(0)
test_indices = random.sample(range(len(test_dataloader)),nb_test)

# calculate the pixel-level classification performance
pr_masks = model.predict(test_dataloader)
# 	pr_masks = []
# 	for i in test_indices:
# 		pr_masks.append(model.predict(test_dataset[i]))
# 	pr_masks = np.stack(pr_masks)
pr_masks = pr_masks[test_indices,:]
pr_maps = np.argmax(pr_masks,axis=-1)

gt_masks = []
for i in test_indices:
	_, gt_mask = test_dataset[i];gt_masks.append(gt_mask)
gt_masks = np.stack(gt_masks);gt_maps = np.argmax(gt_masks,axis=-1)

## calcualte tpr and fpr, fpr, and roc_auc
from sklearn.metrics import roc_curve, auc
classes = ['G1', 'S', 'G2', 'BK']
tpr = dict(); fpr = dict(); thrs = dict(); roc_auc = dict();
for i in range(len(classes)):
	print('Class {}\n'.format(classes[i]))
	y_pr = pr_masks[:,:,:,i].flatten()
	y_true = gt_masks[:,:,:,i].flatten()
	fpr[classes[i]], tpr[classes[i]], thrs[classes[i]] = roc_curve(y_true, y_pr)
	roc_auc[classes[i]] = auc(fpr[classes[i]], tpr[classes[i]])

## decide the threshold by min(tpr - fpr)
tpr_cls = dict(); fpr_cls = dict(); thres_cls = dict()
for i in range(len(classes)):
	thr_index = np.argmax(tpr[classes[i]]-fpr[classes[i]])
	thres_cls[classes[i]] = thrs[classes[i]][thr_index]
	tpr_cls[classes[i]] = tpr[classes[i]][thr_index]
	fpr_cls[classes[i]] = fpr[classes[i]][thr_index]
print(thres_cls); print(tpr_cls); print(fpr_cls)

pr_maps_roc = []
for i in range(pr_masks.shape[0]):
	pr_map_cls = [(pr_masks[i,:,:,k]>thres_cls[key]) for k, key in enumerate(thres_cls.keys())]
	pr_map = pr_map_cls[0]*1.0
	for k in range(1,len(classes)):
		top = pr_map_cls[k].copy()*(k+1)
		top[np.where(np.logical_and(top>0,pr_map>0))]= pr_map[np.where(np.logical_and(top>0,pr_map>0))]
		pr_map[np.where(top>0)] = top[np.where(top>0)]
	pr_maps_roc.append(pr_map)
pr_maps_roc = np.stack(pr_maps_roc)
pr_masks_roc = np.stack([pr_maps_roc == k for k in range(1,len(classes)+1)],axis =-1)
## IoU and dice coefficient
iou_classes, mIoU, dice_classes, mDice = iou_calculate(gt_masks, pr_masks_roc)
print('iou_classes: {:.4f},{:.4f},{:.4f},{:.4f}; mIoU: {:.4f}'.format(iou_classes[-1],iou_classes[0],iou_classes[1],iou_classes[2], mIoU))
print('dice_classes: {:.4f},{:.4f},{:.4f},{:.4f}; mDice: {:.4f}'.format(dice_classes[-1],dice_classes[0],dice_classes[1],dice_classes[2], mDice))

def plot(file_name, pr_masks, pr_masks_roc, gt_masks, index, classes = ['G1', 'S', 'G2', 'BK']):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	rows, cols, size = 3,4,4.5
	fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(rows,cols)
	font_size = 15
	ax[0,0].imshow(gt_masks[index,:,:,0]); ax[0,1].imshow(gt_masks[index,:,:,1])
	ax[0,2].imshow(gt_masks[index,:,:,2]); ax[0,3].imshow(gt_masks[index,:,:,3])
	ax[1,0].imshow(pr_masks[index,:,:,0]); ax[1,1].imshow(pr_masks[index,:,:,1])
	ax[1,2].imshow(pr_masks[index,:,:,2]); ax[1,3].imshow(pr_masks[index,:,:,3])
	ax[2,0].imshow(pr_masks_roc[index,:,:,0]); ax[2,1].imshow(pr_masks_roc[index,:,:,1])
	ax[2,2].imshow(pr_masks_roc[index,:,:,2]); ax[2,3].imshow(pr_masks_roc[index,:,:,3])
	ax[0,0].set_title(classes[0]); ax[0,1].set_title(classes[1]); ax[0,2].set_title(classes[2])
	ax[0,3].set_title(classes[3])
	ax[0,0].set_ylabel('Ground Truth')	
	ax[1,0].set_ylabel('Probability')
	ax[2,0].set_ylabel('Prediction')
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=80)

pred_dir = '/data/eval_train_val_test/'
for i in range(10):
	file_name = pred_dir+'{}.png'.format(i)
	plot(file_name, pr_masks, pr_masks_roc, gt_masks, i)

## plot ROC curves for each classes
def plot_roc_curves(file_name, fpr, tpr, roc_auc, classes = ['G1', 'S', 'G2', 'BK'], bk = True):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	rows, cols, size = 1,1,5
	fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(rows,cols)
	font_size = 15
	ax.plot(fpr[classes[0]], tpr[classes[0]], '--', linewidth = 2); ax.plot(fpr[classes[1]], tpr[classes[1]], '-', linewidth = 2); ax.plot(fpr[classes[2]], tpr[classes[2]], '-', linewidth = 2);
	if bk:
		ax.plot(fpr[classes[3]], tpr[classes[3]], '--', linewidth = 2)
		ax.legend([classes[0]+'(AUC:{:.2f})'.format(roc_auc[classes[0]]),
				   classes[1]+'(AUC:{:.2f})'.format(roc_auc[classes[1]]),
				   classes[2]+'(AUC:{:.2f})'.format(roc_auc[classes[2]]),
				   classes[3]+'(AUC:{:.2f})'.format(roc_auc[classes[3]])], fontsize = font_size)
	else:
		ax.legend([classes[0]+'(AUC:{:.2f})'.format(roc_auc[classes[0]]),
				   classes[1]+'(AUC:{:.2f})'.format(roc_auc[classes[1]]),
				   classes[2]+'(AUC:{:.2f})'.format(roc_auc[classes[2]])], fontsize = font_size)
	ax.set_ylabel('True postive fraction', fontsize = font_size)
	ax.set_xlabel('False postive fraction', fontsize = font_size)
	ax.set_xlim([0.0,1.0])
	ax.set_ylim([0.0,1.0])
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=100)
file_name = result_dir + '/roc_curve_bk.png'
plot_roc_curves(file_name, fpr, tpr, roc_auc, classes = ['G1', 'S', 'G2', 'BK'], bk = True)
file_name = result_dir + '/roc_curve.png'
plot_roc_curves(file_name, fpr, tpr, roc_auc, classes = ['G1', 'S', 'G2', 'BK'], bk = False)