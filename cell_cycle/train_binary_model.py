import os
import cv2
from skimage import io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import segmentation_models_v1 as sm
from segmentation_models_v1 import Unet, Linknet, PSPNet, FPN, DUNet, BiFPN, Nestnet, ResUnet, AtUnet
sm.set_framework('tf.keras')

from helper_function import plot_deeply_history, plot_history, save_history
from helper_function import precision, recall, f1_score
from sklearn.metrics import confusion_matrix
from helper_function import plot_history_for_callback, save_history_for_callback

def str2bool(value):
    return value.lower() == 'true'

def generate_folder(folder_name):
	if not os.path.exists(folder_name):
		os.system('mkdir -p {}'.format(folder_name))

def read_txt(txt_dir):
    lines = []
    with open(txt_dir, 'r+') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

parser = argparse.ArgumentParser()
parser.add_argument("--docker", type=str2bool, default = True)
parser.add_argument("--gpu", type=str, default = '0')
parser.add_argument("--net_type", type=str, default = 'AtUnet')  #Unet, Linknet, PSPNet, FPN
parser.add_argument("--backbone", type=str, default = 'efficientnetb0')
parser.add_argument("--feat_version", type=int, default = None)
parser.add_argument("--epoch", type=int, default = 2)
parser.add_argument("--dim", type=int, default = 512)
parser.add_argument("--batch_size", type=int, default = 2)
parser.add_argument("--dataset", type=str, default = 'cycle_736x752')
parser.add_argument("--upsample", type=str, default = 'upsampling')
parser.add_argument("--pyramid_agg", type=str, default = 'sum')
parser.add_argument("--filters", type=int, default = 256)
parser.add_argument("--rot", type=float, default = 0)
parser.add_argument("--lr", type=float, default = 1e-3)
parser.add_argument("--bk", type=float, default = 0.5)
parser.add_argument("--focal_weight", type=float, default = 1)
parser.add_argument("--pre_train", type=str2bool, default = True)
parser.add_argument("--train", type=int, default = None)
parser.add_argument("--loss", type=str, default = 'focal+dice')
parser.add_argument("--reduce_factor", type=float, default = 1.0)
args = parser.parse_args()
print(args)

model_name = 'binary-net-{}-bone-{}-pre-{}-epoch-{}-batch-{}-lr-{}-dim-{}-train-{}-rot-{}-set-{}-loss-{}-up-{}-filters-{}-red_factor-{}-pyr_agg-{}-bk-{}-fl_weight-{}-fv-{}'.format(args.net_type,\
		 	args.backbone, args.pre_train, args.epoch, args.batch_size, args.lr, args.dim,\
		 	args.train, args.rot, args.dataset, args.loss, args.upsample, args.filters, args.reduce_factor, args.pyramid_agg, args.bk, args.focal_weight, args.feat_version)
print(model_name)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.dataset == 'cycle_736x752':
	val_dim1, val_dim2 = 736, 768
	test_dim1, test_dim2 = 736, 768
	dim1, dim2 = 736, 752

DATA_DIR = '/data/datasets/{}'.format(args.dataset) if args.docker else './datasets/{}'.format(args.dataset)

images_dir = DATA_DIR+'/images'
masks_dir = DATA_DIR+'/masks'

train_fns = read_txt(os.path.join(DATA_DIR, 'train_list.txt'))
val_fns = read_txt(os.path.join(DATA_DIR, 'valid_list.txt'))
test_fns = read_txt(os.path.join(DATA_DIR, 'test_list.txt'))

print('train:{}, valid:{}, test:{}'.format(len(train_fns),len(val_fns), len(test_fns)))

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
            fn_list,
            classes=None,
            nb_data=None,
            augmentation=None, 
            preprocessing=None,
    ):
        id_list = fn_list
        if nb_data ==None:
            self.ids = id_list
        else:
            self.ids = id_list[:int(min(nb_data,len(id_list)))]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        print(len(self.images_fps)); print(len(self.masks_fps))
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = io.imread(self.images_fps[i])
        mask = io.imread(self.masks_fps[i])
        # image to RGB
        image = np.uint8((image-image.min())*255/(image.max()-image.min()))
        image = np.stack([image,image,image],axis =-1)
        
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
#         map_batch = batch[1]
#         map_batch_list = [map_batch]
#         for i in range(4):
#             map_batch_list.append(map_batch[:,::2,::2,:])
#             map_batch = map_batch[:,::2,::2,:]
#         map_batch_list.reverse()
#         map_tuple = ()
#         for i in range(5):
#             map_tuple = map_tuple+(map_batch_list[i],)
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

# define heavy augmentations
def get_training_augmentation(dim = 512, rot_limit = 45):
    train_transform = [

        A.HorizontalFlip(p=0.5),


        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=rot_limit, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=dim, min_width=dim, always_apply=True, border_mode=0),
        A.RandomCrop(height=dim, width=dim, always_apply=True),

        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation(dim = 832):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(dim, dim),
        A.RandomCrop(height=dim, width=dim, always_apply=True)
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


# BACKBONE = 'efficientnetb3'
BACKBONE = args.backbone
BATCH_SIZE = args.batch_size
# CLASSES = ['live', 'inter', 'dead']
CLASSES = ['bk']
LR = args.lr
EPOCHS = args.epoch

preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

#create model
net_func = globals()[args.net_type]

encoder_weights='imagenet' if args.pre_train else None

if args.net_type == 'PSPNet':
	model = net_func(BACKBONE, encoder_weights=encoder_weights, input_shape = (args.dim, args.dim, 3), classes=n_classes, activation=activation)
elif args.net_type == 'FPN':
    model = net_func(BACKBONE, encoder_weights=encoder_weights, classes=n_classes, activation=activation, pyramid_aggregation = args.pyramid_agg) 
else:
    model = net_func(BACKBONE, encoder_weights=encoder_weights, classes=n_classes, activation=activation,\
    		decoder_block_type = args.upsample, feature_version = args.feat_version,\
    		decoder_filters=(int(args.filters),int(args.filters/2), int(args.filters/4), int(args.filters/8), int(args.filters/16)))
    print('{}'.format((int(args.filters),int(args.filters/2), int(args.filters/4), int(args.filters/8), int(args.filters/16))))
# else:
#     model = net_func(BACKBONE, encoder_weights=encoder_weights, input_shape = (args.dim, args.dim, 3), classes=n_classes, activation=activation)
# model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

# define optomizer
optim = tf.keras.optimizers.Adam(LR)

# class_weights = [1,1,1,args.bk]
# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
if args.loss =='focal+dice':
	dice_loss = sm.losses.DiceLoss()
	focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
	total_loss = dice_loss + (args.focal_weight * focal_loss)
elif args.loss =='dice':
	total_loss = sm.losses.DiceLoss()
elif args.loss =='jaccard':
	total_loss = sm.losses.JaccardLoss()
elif args.loss =='focal+jaccard':
	dice_loss = sm.losses.JaccardLoss()
	focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
	total_loss = dice_loss + (args.focal_weight * focal_loss)
elif args.loss =='focal+jaccard+dice':
	dice_loss = sm.losses.JaccardLoss()
	jaccard_loss = sm.losses.JaccardLoss()
	focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
	total_loss = dice_loss + jaccard_loss+ (args.focal_weight * focal_loss)
elif args.loss == 'focal':
	total_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
# 	focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
# 	total_loss = dice_loss + (args.forcal_weight * focal_loss)
# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
model.compile(optimizer=optim, loss=total_loss, metrics = metrics)

# Dataset for train images
train_dataset = Dataset(
    images_dir, 
    masks_dir, 
    train_fns,
    classes=CLASSES,
    nb_data=args.train,
    augmentation=get_training_augmentation(args.dim, args.rot),
    preprocessing=get_preprocessing(preprocess_input),
)

if args.net_type == 'PSPNet':
    val_dim = args.dim

# Dataset for validation images
valid_dataset = Dataset(
    images_dir, 
    masks_dir,
    val_fns,
    classes=CLASSES, 
    augmentation=get_validation_augmentation(val_dim1),
    preprocessing=get_preprocessing(preprocess_input),
)

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

print(train_dataloader[0][0].shape)
# check shapes for errors
assert train_dataloader[0][0].shape == (BATCH_SIZE, args.dim, args.dim, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, args.dim, args.dim, n_classes)

model_folder = '/data/cycle_models/{}'.format(model_name) if args.docker else './models/cycle_models/{}'.format(model_name)
generate_folder(model_folder)

def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

def save_images(file_name, vols):
# 		vols = vols[:,:,:,1] if vols.shape[-1] >= 2 else vols[:,:,:,0]
		shp = vols.shape
		ls, lx, ly, lc = shp
		sx, sy = int(lx/256), int(ly/256)
		vols = vols[:,::sx,::sy,:]
		slice_list, rows = [], []
		for si in range(vols.shape[0]):
				slice = vols[si,:,:,:]
				rows.append(slice)
				if si%4 == 3 and not si == vols.shape[0]-1:
						slice_list.append(rows)
						rows = []
		save_img = concat_tile(slice_list)		
		cv2.imwrite(file_name, save_img)

def map2rgb(maps):
	shp = maps.shape
	rgb_maps = np.zeros((shp[0], shp[1], shp[2], 3), dtype=np.uint8)
	rgb_maps[:,:,:,0] = np.uint8((maps==0)*255)
	rgb_maps[:,:,:,1] = np.uint8((maps==1)*255)
	rgb_maps[:,:,:,2] = np.uint8((maps==2)*255)
	return rgb_maps
	
class HistoryPrintCallback(tf.keras.callbacks.Callback):
		def __init__(self):
				super(HistoryPrintCallback, self).__init__()
				self.history = {}

		def on_epoch_end(self, epoch, logs=None):
				if logs:
						for key in logs.keys():
								if epoch == 0:
										self.history[key] = []
								self.history[key].append(logs[key])
				if epoch%5 == 0:
						plot_history_for_callback(model_folder+'/train_history.png', self.history)
						save_history_for_callback(model_folder, self.history)
						gt_vols, pr_vols = [],[]
						for i in range(0, len(valid_dataset),int(len(valid_dataset)/36)):
								gt_vols.append(valid_dataloader[i][1])
								pr_vols.append(self.model.predict(valid_dataloader[i]))
						gt_vols = np.concatenate(gt_vols, axis = 0); gt_map = map2rgb(gt_vols.squeeze())
						pr_vols = np.concatenate(pr_vols, axis = 0); pr_map = map2rgb(pr_vols.squeeze()>0.5)
						if epoch == 0:
								save_images(model_folder+'/ground_truth.png'.format(epoch), gt_map)
						save_images(model_folder+'/pr-{}.png'.format(epoch), pr_map)

# define callbacks for learning rate scheduling and best checkpoints saving
if args.reduce_factor < 1.0:
	callbacks = [
		tf.keras.callbacks.ModelCheckpoint(model_folder+'/best_model-{epoch:03d}.h5', save_weights_only=True, save_best_only=True, mode='min'),
		tf.keras.callbacks.ReduceLROnPlateau(factor=args.reduce_factor),
		HistoryPrintCallback(),
	]
else:
	callbacks = [
		tf.keras.callbacks.ModelCheckpoint(model_folder+'/best_model-{epoch:03d}.h5', save_weights_only=True, save_best_only=True, mode='min'),
		HistoryPrintCallback(),
	]

# train model
history = model.fit_generator(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
)

# save the training information
# plot_history(model_folder+'/train_history.png',history)
# record_dir = model_folder+'/train_dir'
# generate_folder(record_dir)
# save_history(record_dir, history)