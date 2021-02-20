import os
import numpy as np
from skimage import io

unet_folder = '/data/U-net_results/'
unet_fl1_dir = unet_folder +'result_train_glim_to_fl1_fil_20200819_181527_final_model/pred_test/'
unet_fl2_dir = unet_folder +'result_train_glim_to_fl2_fil_20200820_141534_final_model/pred_test/'
output_dir = unet_folder+'/png_results'

if not os.path.exists(output_dir):
	os.system('mkdir -p {}'.format(output_dir))

fl1_fnames = os.listdir(unet_fl1_dir)
fl2_fnames = fl1_fnames

fl1_maps = [io.imread(unet_fl1_dir+'/{}'.format(fname)) for fname in fl1_fnames]
fl2_maps = [io.imread(unet_fl2_dir+'/{}'.format(fname)) for fname in fl1_fnames]

fl1_maps = np.stack(fl1_maps)
fl2_maps = np.stack(fl2_maps)

for i in range(fl1_maps.shape[0]):
	fl1_map = fl1_maps[i]; fl2_map = fl2_maps[i]
	fl1_map = np.uint8(255.*fl1_map/fl1_map.max()); fl2_map = np.uint8(255.*fl2_map/fl2_map.max())
	zero_map = np.zeros(fl1_map.shape).astype(np.uint8)
	fl_map_rgb = np.stack([fl1_map, fl2_map, zero_map], axis = -1).astype(np.uint8)
	io.imsave(output_dir+'/{}'.format(fl1_fnames[i].replace('tif', 'png')), fl_map_rgb)