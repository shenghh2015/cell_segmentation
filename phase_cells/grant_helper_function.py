import numpy as np
import os

def generate_folder(folder):
	import os
	if not os.path.exists(folder):
		os.system('mkdir -p {}'.format(folder))

def plot_flu_prediction(file_name, images, gt_maps, pr_maps, nb_images, rand_seed = 3, colorbar = True):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	import random
	seed = rand_seed #3
	random.seed(seed)
	font_size = 28; label_size = 20
	indices = random.sample(range(gt_maps.shape[0]),nb_images)
	rows, cols, size = nb_images,4,5
	widths = [0.8, 1, 1, 1]; heights = [1, 1, 1, 1]; gs_kw = dict(width_ratios=widths, height_ratios=heights)
	
	fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(nrows=rows,ncols=cols, gridspec_kw=gs_kw)
	for i in range(len(indices)):
		idx = indices[i]
		image, gt_map, pr_map = images[idx,:].squeeze(), gt_maps[idx,:].squeeze(), pr_maps[idx,:].squeeze()
		image = np.uint8((image-image.min())/(image.max()-image.min())*255)
		err_map = np.abs(gt_map-pr_map)*255; err_map_rgb = np.zeros(image.shape).astype(np.uint8); err_map_rgb[:,:,:-1] = err_map
		gt_map_rgb = np.zeros(image.shape,dtype=np.uint8); # gt_map_rgb[:,:,:-1]=np.uint8((gt_map-gt_map.min())/(gt_map.max()-gt_map.min())*255)
		gt_map_rgb[:,:,:-1]=np.uint8(gt_map*255)
		pr_map_rgb = np.zeros(image.shape,dtype=np.uint8); # pr_map_rgb[:,:,:-1]=np.uint8((pr_map-pr_map.min())/(pr_map.max()-pr_map.min())*255)
		pr_map_rgb[:,:,:-1]=np.uint8(pr_map*255)
		cx0 = ax[i,0].imshow(image); cx1 = ax[i,1].imshow(gt_map_rgb); 
		cx2 = ax[i,2].imshow(pr_map_rgb); cx3 = ax[i,3].imshow(err_map_rgb)
# 		cx0 = ax[i,0].imshow(image[::4,::4,:]); cx1 = ax[i,1].imshow(gt_map_rgb[::4,::4,:]); 
# 		cx2 = ax[i,2].imshow(pr_map_rgb[::4,::4,:]); cx3 = ax[i,3].imshow(err_map_rgb[::4,::4])
		ax[i,0].set_xticks([]);ax[i,0].set_yticks([]);
		#ax[i,0].tick_params(axis = 'x', labelsize = label_size);ax[i,0].tick_params(axis = 'y', labelsize = label_size);
		ax[i,1].set_xticks([]);ax[i,2].set_xticks([]);ax[i,3].set_xticks([])
		ax[i,1].set_yticks([]);ax[i,2].set_yticks([]);ax[i,3].set_yticks([])
		if colorbar:
			#fig.colorbar(cx0, ax = ax[i,0], shrink = 0); 
			cbar = fig.colorbar(cx1, ax = ax[i,1], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
			cbar = fig.colorbar(cx2, ax = ax[i,2], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
			cbar = fig.colorbar(cx3, ax = ax[i,3], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
		if i == 0:
			ax[i,0].set_title('Image',fontsize=font_size); ax[i,1].set_title('Ground Truth',fontsize=font_size); 
			ax[i,2].set_title('Prediction',fontsize=font_size); ax[i,3].set_title('Err Map',fontsize=font_size);
	fig.tight_layout(pad=-2)
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=80)

def plot_flu_prediction2(fig_root_dir, images, gt_maps, pr_maps, nb_images, rand_seed = 3, colorbar = True):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	import random
	from skimage import io
	seed = rand_seed #3
	random.seed(seed)
	font_size = 28; label_size = 18
	indices = random.sample(range(gt_maps.shape[0]),nb_images)
	rows, cols, size = nb_images,4,6

	for i in range(len(indices)):
		example_folder = os.path.join(fig_root_dir, 'example_{:03d}'.format(i)); generate_folder(example_folder)
		cols, rows = 1, 1
		fig1 = Figure(tight_layout=True,figsize=(size-1.0, size-1.2))
		fig2 = Figure(tight_layout=True,figsize=(size, size-1.2))
		fig3 = Figure(tight_layout=True,figsize=(size, size-1.2))
		fig4 = Figure(tight_layout=True,figsize=(size, size-1.2))
		ax = fig1.subplots(nrows=rows,ncols=cols); #ax.set_xticks([]);ax.set_yticks([])
		bx = fig2.subplots(nrows=rows,ncols=cols); #ax.set_xticks([]);ax.set_yticks([])
		cx = fig3.subplots(nrows=rows,ncols=cols); #ax.set_xticks([]);ax.set_yticks([])
		dx = fig4.subplots(nrows=rows,ncols=cols); #ax.set_xticks([]);ax.set_yticks([])
		idx = indices[i]
		image, gt_map, pr_map = images[idx,:].squeeze(), gt_maps[idx,:].squeeze(), pr_maps[idx,:].squeeze()
		image = np.uint8((image-image.min())/(image.max()-image.min())*255)
		err_map = np.abs(gt_map-pr_map)*255; err_map_rgb = np.zeros(image.shape).astype(np.uint8); err_map_rgb[:,:,:-1] = err_map
		gt_map_rgb = np.zeros(image.shape,dtype=np.uint8); # gt_map_rgb[:,:,:-1]=np.uint8((gt_map-gt_map.min())/(gt_map.max()-gt_map.min())*255)
		gt_map_rgb[:,:,:-1]=np.uint8(gt_map*255)
		pr_map_rgb = np.zeros(image.shape,dtype=np.uint8); # pr_map_rgb[:,:,:-1]=np.uint8((pr_map-pr_map.min())/(pr_map.max()-pr_map.min())*255)
		pr_map_rgb[:,:,:-1]=np.uint8(pr_map*255)
		cax = ax.imshow(image); #cbar = fig1.colorbar(cax, ax = ax, shrink = 0.68); cbar.ax.tick_params(labelsize=label_size)
		cbx = bx.imshow(gt_map_rgb); cbar = fig2.colorbar(cbx, ax = bx, shrink = 0.97); cbar.ax.tick_params(labelsize=label_size)
		ccx = cx.imshow(pr_map_rgb); cbar = fig3.colorbar(ccx, ax = cx, shrink = 0.97); cbar.ax.tick_params(labelsize=label_size)
		cdx = dx.imshow(err_map_rgb); cbar = fig4.colorbar(cdx, ax = dx, shrink = 0.97); cbar.ax.tick_params(labelsize=label_size)
		ax.tick_params(axis = 'x', labelsize = label_size); ax.tick_params(axis = 'y', labelsize = label_size);
		bx.tick_params(axis = 'x', labelsize = label_size); bx.tick_params(axis = 'y', labelsize = label_size);
		cx.tick_params(axis = 'x', labelsize = label_size); cx.tick_params(axis = 'y', labelsize = label_size);
		dx.tick_params(axis = 'x', labelsize = label_size); dx.tick_params(axis = 'y', labelsize = label_size);
		canvas = FigureCanvasAgg(fig1); canvas.print_figure(example_folder+'/Image.png', dpi=120)
		canvas = FigureCanvasAgg(fig2); canvas.print_figure(example_folder+'/Ground_truth.png', dpi=120)
		canvas = FigureCanvasAgg(fig3); canvas.print_figure(example_folder+'/Prediction.png', dpi=120)
		canvas = FigureCanvasAgg(fig4); canvas.print_figure(example_folder+'/Error_map.png', dpi=120)

def plot_flu_prediction3(file_name, images, gt_maps, pr_maps, cut_folder, image_ids, nb_images, rand_seed = 3, colorbar = True):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	from skimage import io
	import random
	seed = rand_seed #3
	random.seed(seed)
	font_size = 28; label_size = 20
	indices = random.sample(range(gt_maps.shape[0]),nb_images)
	rows, cols, size = nb_images,5,5
	widths = [0.8, 1, 1, 1, 1]; heights = [1, 1, 1, 1]; gs_kw = dict(width_ratios=widths, height_ratios=heights)

	fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(nrows=rows,ncols=cols, gridspec_kw=gs_kw)
	for i in range(len(indices)):
		idx = indices[i]
		image, gt_map, pr_map = images[idx,:].squeeze(), gt_maps[idx,:].squeeze(), pr_maps[idx,:].squeeze()
		image = np.uint8((image-image.min())/(image.max()-image.min())*255)
		err_map = np.abs(gt_map-pr_map)*255; err_map_rgb = np.zeros(image.shape).astype(np.uint8); err_map_rgb[:,:,:-1] = err_map
		gt_map_rgb = np.zeros(image.shape,dtype=np.uint8); # gt_map_rgb[:,:,:-1]=np.uint8((gt_map-gt_map.min())/(gt_map.max()-gt_map.min())*255)
		gt_map_rgb[:,:,:-1]=np.uint8(gt_map*255)
		pr_map_rgb = np.zeros(image.shape,dtype=np.uint8); # pr_map_rgb[:,:,:-1]=np.uint8((pr_map-pr_map.min())/(pr_map.max()-pr_map.min())*255)
		pr_map_rgb[:,:,:-1]=np.uint8(pr_map*255)
		cut_map = io.imread(cut_folder+'/{}'.format(image_ids[idx]))
		cx0 = ax[i,0].imshow(image); cx1 = ax[i,1].imshow(gt_map_rgb); 
		cx2 = ax[i,2].imshow(pr_map_rgb); cx3 = ax[i,3].imshow(err_map_rgb); cx4 = ax[i,4].imshow(cut_map);
		ax[i,0].set_xticks([]);ax[i,0].set_yticks([]);
		#ax[i,0].tick_params(axis = 'x', labelsize = label_size);ax[i,0].tick_params(axis = 'y', labelsize = label_size);
		ax[i,1].set_xticks([]);ax[i,2].set_xticks([]);ax[i,3].set_xticks([]); ax[i,4].set_xticks([]);
		ax[i,1].set_yticks([]);ax[i,2].set_yticks([]);ax[i,3].set_yticks([]); ax[i,4].set_yticks([]);
		if colorbar:
			#fig.colorbar(cx0, ax = ax[i,0], shrink = 0); 
			cbar = fig.colorbar(cx1, ax = ax[i,1], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
			cbar = fig.colorbar(cx2, ax = ax[i,2], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
			cbar = fig.colorbar(cx3, ax = ax[i,3], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size)
			cbar = fig.colorbar(cx4, ax = ax[i,4], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
		if i == 0:
			ax[i,0].set_title('Image',fontsize=font_size);
			ax[i,1].set_title('Ground Truth',fontsize=font_size);
			ax[i,2].set_title('Paired Method',fontsize=font_size);
			ax[i,3].set_title('Err Map (Paired)',fontsize=font_size);
			ax[i,4].set_title('Unpaired Method',fontsize=font_size);
	fig.tight_layout(pad=-2)
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=80)

def plot_flu_prediction4(fig_root_dir, images, gt_maps, pr_maps, cut_folder, image_ids, nb_images, rand_seed = 3, colorbar = True):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	from skimage import io
	import random
	seed = rand_seed #3
	random.seed(seed)
	font_size = 28; label_size = 18
	indices = random.sample(range(gt_maps.shape[0]),nb_images)
	rows, cols, size = nb_images,4,6

	for i in range(len(indices)):
		example_folder = os.path.join(fig_root_dir, 'example_{:03d}'.format(i)); generate_folder(example_folder)
		cols, rows = 1, 1
		fig1 = Figure(tight_layout=True,figsize=(size-1.0, size-1.2))
		fig2 = Figure(tight_layout=True,figsize=(size, size-1.2))
		fig3 = Figure(tight_layout=True,figsize=(size, size-1.2))
		fig4 = Figure(tight_layout=True,figsize=(size, size-1.2))
		fig5 = Figure(tight_layout=True,figsize=(size, size-1.2))
		ax = fig1.subplots(nrows=rows,ncols=cols); #ax.set_xticks([]);ax.set_yticks([])
		bx = fig2.subplots(nrows=rows,ncols=cols); #ax.set_xticks([]);ax.set_yticks([])
		cx = fig3.subplots(nrows=rows,ncols=cols); #ax.set_xticks([]);ax.set_yticks([])
		dx = fig4.subplots(nrows=rows,ncols=cols); #ax.set_xticks([]);ax.set_yticks([])
		ex = fig5.subplots(nrows=rows,ncols=cols);
		idx = indices[i]
		image, gt_map, pr_map = images[idx,:].squeeze(), gt_maps[idx,:].squeeze(), pr_maps[idx,:].squeeze()
		image = np.uint8((image-image.min())/(image.max()-image.min())*255)
		err_map = np.abs(gt_map-pr_map)*255; err_map_rgb = np.zeros(image.shape).astype(np.uint8); err_map_rgb[:,:,:-1] = err_map
		gt_map_rgb = np.zeros(image.shape,dtype=np.uint8); # gt_map_rgb[:,:,:-1]=np.uint8((gt_map-gt_map.min())/(gt_map.max()-gt_map.min())*255)
		gt_map_rgb[:,:,:-1]=np.uint8(gt_map*255)
		pr_map_rgb = np.zeros(image.shape,dtype=np.uint8); # pr_map_rgb[:,:,:-1]=np.uint8((pr_map-pr_map.min())/(pr_map.max()-pr_map.min())*255)
		pr_map_rgb[:,:,:-1]=np.uint8(pr_map*255)
		unpaired_map = io.imread(cut_folder+'/{}'.format(image_ids[idx]))
		cax = ax.imshow(image); #cbar = fig1.colorbar(cax, ax = ax, shrink = 0.68); cbar.ax.tick_params(labelsize=label_size)
		cbx = bx.imshow(gt_map_rgb); cbar = fig2.colorbar(cbx, ax = bx, shrink = 0.97); cbar.ax.tick_params(labelsize=label_size)
		ccx = cx.imshow(pr_map_rgb); cbar = fig3.colorbar(ccx, ax = cx, shrink = 0.97); cbar.ax.tick_params(labelsize=label_size)
		cdx = dx.imshow(err_map_rgb); cbar = fig4.colorbar(cdx, ax = dx, shrink = 0.97); cbar.ax.tick_params(labelsize=label_size)
		cex = ex.imshow(unpaired_map); cbar = fig5.colorbar(cdx, ax = ex, shrink = 0.97); cbar.ax.tick_params(labelsize=label_size)
		ax.tick_params(axis = 'x', labelsize = label_size); ax.tick_params(axis = 'y', labelsize = label_size);
		bx.tick_params(axis = 'x', labelsize = label_size); bx.tick_params(axis = 'y', labelsize = label_size);
		cx.tick_params(axis = 'x', labelsize = label_size); cx.tick_params(axis = 'y', labelsize = label_size);
		dx.tick_params(axis = 'x', labelsize = label_size); dx.tick_params(axis = 'y', labelsize = label_size);
		ex.tick_params(axis = 'x', labelsize = label_size); ex.tick_params(axis = 'y', labelsize = label_size);
		canvas = FigureCanvasAgg(fig1); canvas.print_figure(example_folder+'/Image.png', dpi=120)
		canvas = FigureCanvasAgg(fig2); canvas.print_figure(example_folder+'/Ground_truth.png', dpi=120)
		canvas = FigureCanvasAgg(fig3); canvas.print_figure(example_folder+'/Prediction.png', dpi=120)
		canvas = FigureCanvasAgg(fig4); canvas.print_figure(example_folder+'/Error_map.png', dpi=120)
		canvas = FigureCanvasAgg(fig5); canvas.print_figure(example_folder+'/Unpaired_pred.png', dpi=120)

def plot_flu_prediction5(file_name, images, gt_maps, pr_maps, cut_folder, unet_fl1_dir, unet_fl2_dir, image_ids, nb_images, rand_seed = 3, colorbar = True):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	from skimage import io
	import random
	seed = rand_seed #3
	random.seed(seed)
	font_size = 28; label_size = 20
	indices = random.sample(range(gt_maps.shape[0]),nb_images)
	rows, cols, size = nb_images,7,6
	widths = [0.8, 1, 1, 1, 1,1,1]; heights = [1, 1, 1, 1]; gs_kw = dict(width_ratios=widths, height_ratios=heights)

	fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(nrows=rows,ncols=cols, gridspec_kw=gs_kw)
	for i in range(len(indices)):
		idx = indices[i]
		image, gt_map, pr_map = images[idx,:].squeeze(), gt_maps[idx,:].squeeze(), pr_maps[idx,:].squeeze()
		image = np.uint8((image-image.min())/(image.max()-image.min())*255)
		err_map = np.abs(gt_map-pr_map)*255; err_map_rgb = np.zeros(image.shape).astype(np.uint8); err_map_rgb[:,:,:-1] = err_map
		gt_map_rgb = np.zeros(image.shape,dtype=np.uint8); # gt_map_rgb[:,:,:-1]=np.uint8((gt_map-gt_map.min())/(gt_map.max()-gt_map.min())*255)
		gt_map_rgb[:,:,:-1]=np.uint8(gt_map*255)
		pr_map_rgb = np.zeros(image.shape,dtype=np.uint8); # pr_map_rgb[:,:,:-1]=np.uint8((pr_map-pr_map.min())/(pr_map.max()-pr_map.min())*255)
		pr_map_rgb[:,:,:-1]=np.uint8(pr_map*255)
		cut_map = io.imread(cut_folder+'/{}'.format(image_ids[idx]))
		cx0 = ax[i,0].imshow(image); cx1 = ax[i,1].imshow(gt_map_rgb); 
		cx2 = ax[i,2].imshow(pr_map_rgb); cx3 = ax[i,3].imshow(err_map_rgb); cx4 = ax[i,6].imshow(cut_map);
		unet_fl1 = io.imread(unet_fl1_dir+'/pred_{}'.format(image_ids[idx].replace('png','tif'))); unet_fl2 = io.imread(unet_fl2_dir+'/pred_{}'.format(image_ids[idx].replace('png','tif')))
# 		unet_fl2 = np.zeros(unet_fl1.shape)
		unet_fl1 = unet_fl1/unet_fl1.max(); unet_fl2 = unet_fl2/unet_fl2.max()
		unet_map = np.stack([unet_fl1, unet_fl2], axis = -1); print(unet_map.min(), unet_map.max())
		#unet_map = unet_map/unet_map.max(); print(unet_map.min(), unet_map.max())
		unet_err_map = np.abs(gt_map-unet_map)*255; unet_err_map_rgb = np.zeros(image.shape).astype(np.uint8); unet_err_map_rgb[:,:,:-1] = unet_err_map
		unet_map_rgb = np.zeros(image.shape,dtype=np.uint8);
		unet_map_rgb[:,:,:-1]=np.uint8(unet_map*255)
		cx5 = ax[i,4].imshow(unet_map_rgb); cx6 = ax[i,5].imshow(unet_err_map_rgb); 
		ax[i,0].set_xticks([]);ax[i,0].set_yticks([]);
		#ax[i,0].tick_params(axis = 'x', labelsize = label_size);ax[i,0].tick_params(axis = 'y', labelsize = label_size);
		ax[i,1].set_xticks([]);ax[i,2].set_xticks([]);ax[i,3].set_xticks([]); ax[i,4].set_xticks([]); ax[i,5].set_xticks([]); ax[i,6].set_xticks([]);
		ax[i,1].set_yticks([]);ax[i,2].set_yticks([]);ax[i,3].set_yticks([]); ax[i,4].set_yticks([]); ax[i,5].set_yticks([]); ax[i,6].set_yticks([]);
		if colorbar:
			#fig.colorbar(cx0, ax = ax[i,0], shrink = 0); 
			cbar = fig.colorbar(cx1, ax = ax[i,1], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
			cbar = fig.colorbar(cx2, ax = ax[i,2], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
			cbar = fig.colorbar(cx3, ax = ax[i,3], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size)
			cbar = fig.colorbar(cx4, ax = ax[i,4], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size)
			cbar = fig.colorbar(cx5, ax = ax[i,5], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
			cbar = fig.colorbar(cx6, ax = ax[i,6], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
		if i == 0:
			ax[i,0].set_title('Image',fontsize=font_size);
			ax[i,1].set_title('Ground Truth',fontsize=font_size);
			ax[i,2].set_title('E-Net (Paired)',fontsize=font_size);
			ax[i,3].set_title('Err Map (E-Net)',fontsize=font_size);
			ax[i,4].set_title('U-Net (Paired)',fontsize=font_size);
			ax[i,5].set_title('Err Map (U-Net)',fontsize=font_size);
			ax[i,6].set_title('Unpaired Method',fontsize=font_size);
	fig.tight_layout(pad=-2)
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=120)

def plot_flu_prediction6(fig_root_dir, images, gt_maps, pr_maps, cut_folder, unet_fl1_dir, unet_fl2_dir, image_ids, nb_images, rand_seed = 3, colorbar = True):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	from skimage import io
	import random
	seed = rand_seed #3
	random.seed(seed)
	font_size = 28; label_size = 18
	indices = random.sample(range(gt_maps.shape[0]),nb_images)
	rows, cols, size = nb_images,4,6

	for i in range(len(indices)):
		example_folder = os.path.join(fig_root_dir, 'example_{:03d}'.format(i)); generate_folder(example_folder)
		cols, rows = 1, 1
		fig1 = Figure(tight_layout=True,figsize=(size-1.0, size-1.2))
		fig2 = Figure(tight_layout=True,figsize=(size, size-1.2))
		fig3 = Figure(tight_layout=True,figsize=(size, size-1.2))
		fig4 = Figure(tight_layout=True,figsize=(size, size-1.2))
		fig5 = Figure(tight_layout=True,figsize=(size, size-1.2))
		fig6 = Figure(tight_layout=True,figsize=(size, size-1.2))
		fig7 = Figure(tight_layout=True,figsize=(size, size-1.2))
		ax = fig1.subplots(nrows=rows,ncols=cols); #ax.set_xticks([]);ax.set_yticks([])
		bx = fig2.subplots(nrows=rows,ncols=cols); #ax.set_xticks([]);ax.set_yticks([])
		cx = fig3.subplots(nrows=rows,ncols=cols); #ax.set_xticks([]);ax.set_yticks([])
		dx = fig4.subplots(nrows=rows,ncols=cols); #ax.set_xticks([]);ax.set_yticks([])
		ex = fig5.subplots(nrows=rows,ncols=cols);
		fx = fig6.subplots(nrows=rows,ncols=cols);
		gx = fig7.subplots(nrows=rows,ncols=cols);
		idx = indices[i]
		image, gt_map, pr_map = images[idx,:].squeeze(), gt_maps[idx,:].squeeze(), pr_maps[idx,:].squeeze()
		image = np.uint8((image-image.min())/(image.max()-image.min())*255)
		err_map = np.abs(gt_map-pr_map)*255; err_map_rgb = np.zeros(image.shape).astype(np.uint8); err_map_rgb[:,:,:-1] = err_map
		gt_map_rgb = np.zeros(image.shape,dtype=np.uint8); # gt_map_rgb[:,:,:-1]=np.uint8((gt_map-gt_map.min())/(gt_map.max()-gt_map.min())*255)
		gt_map_rgb[:,:,:-1]=np.uint8(gt_map*255)
		pr_map_rgb = np.zeros(image.shape,dtype=np.uint8); # pr_map_rgb[:,:,:-1]=np.uint8((pr_map-pr_map.min())/(pr_map.max()-pr_map.min())*255)
		pr_map_rgb[:,:,:-1]=np.uint8(pr_map*255)
		unpaired_map = io.imread(cut_folder+'/{}'.format(image_ids[idx]))
		unet_fl1 = io.imread(unet_fl1_dir+'/pred_{}'.format(image_ids[idx].replace('png','tif'))); unet_fl2 = io.imread(unet_fl2_dir+'/pred_{}'.format(image_ids[idx].replace('png','tif')))
		unet_fl1 = unet_fl1/unet_fl1.max(); unet_fl2 = unet_fl2/unet_fl2.max()
		unet_map = np.stack([unet_fl1, unet_fl2], axis = -1); print(unet_map.min(), unet_map.max())
		unet_err_map = np.abs(gt_map-unet_map)*255; unet_err_map_rgb = np.zeros(image.shape).astype(np.uint8); unet_err_map_rgb[:,:,:-1] = unet_err_map
		unet_map_rgb = np.zeros(image.shape,dtype=np.uint8);
		unet_map_rgb[:,:,:-1]=np.uint8(unet_map*255)

		cax = ax.imshow(image); #cbar = fig1.colorbar(cax, ax = ax, shrink = 0.68); cbar.ax.tick_params(labelsize=label_size)
		cbx = bx.imshow(gt_map_rgb); cbar = fig2.colorbar(cbx, ax = bx, shrink = 0.97); cbar.ax.tick_params(labelsize=label_size)
		ccx = cx.imshow(pr_map_rgb); cbar = fig3.colorbar(ccx, ax = cx, shrink = 0.97); cbar.ax.tick_params(labelsize=label_size)
		cdx = dx.imshow(err_map_rgb); cbar = fig4.colorbar(cdx, ax = dx, shrink = 0.97); cbar.ax.tick_params(labelsize=label_size)
		cex = ex.imshow(unpaired_map); cbar = fig5.colorbar(cdx, ax = ex, shrink = 0.97); cbar.ax.tick_params(labelsize=label_size)
		cfx = fx.imshow(unet_map_rgb); cbar = fig6.colorbar(cdx, ax = fx, shrink = 0.97); cbar.ax.tick_params(labelsize=label_size)
		cgx = gx.imshow(unet_err_map_rgb); cbar = fig7.colorbar(cdx, ax = gx, shrink = 0.97); cbar.ax.tick_params(labelsize=label_size)
		ax.tick_params(axis = 'x', labelsize = label_size); ax.tick_params(axis = 'y', labelsize = label_size);
		bx.tick_params(axis = 'x', labelsize = label_size); bx.tick_params(axis = 'y', labelsize = label_size);
		cx.tick_params(axis = 'x', labelsize = label_size); cx.tick_params(axis = 'y', labelsize = label_size);
		dx.tick_params(axis = 'x', labelsize = label_size); dx.tick_params(axis = 'y', labelsize = label_size);
		ex.tick_params(axis = 'x', labelsize = label_size); ex.tick_params(axis = 'y', labelsize = label_size);
		fx.tick_params(axis = 'x', labelsize = label_size); fx.tick_params(axis = 'y', labelsize = label_size);
		gx.tick_params(axis = 'x', labelsize = label_size); gx.tick_params(axis = 'y', labelsize = label_size);
		canvas = FigureCanvasAgg(fig1); canvas.print_figure(example_folder+'/Image.png', dpi=120)
		canvas = FigureCanvasAgg(fig2); canvas.print_figure(example_folder+'/Ground_truth.png', dpi=120)
		canvas = FigureCanvasAgg(fig3); canvas.print_figure(example_folder+'/EfficientNet_Prediction.png', dpi=120)
		canvas = FigureCanvasAgg(fig4); canvas.print_figure(example_folder+'/EfficientNet_Error_map.png', dpi=120)
		canvas = FigureCanvasAgg(fig5); canvas.print_figure(example_folder+'/Unpaired_pred.png', dpi=120)
		canvas = FigureCanvasAgg(fig6); canvas.print_figure(example_folder+'/UNet_Prediction.png', dpi=120)
		canvas = FigureCanvasAgg(fig7); canvas.print_figure(example_folder+'/UNet_Error_map.png', dpi=120)

def plot_flu_prediction7(file_name, images, gt_maps, pr_maps, cut_folder, unet_fl1_dir, unet_fl2_dir, image_ids, nb_images, rand_seed = 3, colorbar = True):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	from skimage import io
	import random
	seed = rand_seed #3
	random.seed(seed)
	font_size = 28; label_size = 20
	indices = random.sample(range(gt_maps.shape[0]),nb_images)

	pre_select_fnames = ['f0_t0_i0_ch0_c2_r3_z1_mhilbert.png', 'f0_t4_i0_ch0_c4_r1_z0_mhilbert.png', 
				'f0_t4_i0_ch0_c2_r2_z1_mhilbert.png', 'f0_t4_i0_ch0_c1_r0_z1_mhilbert.png']
	
	indices = [image_ids.index(fname) for fname in pre_select_fnames]

	rows, cols, size = nb_images,7,6
	widths = [0.8, 1, 1, 1, 1,1,1]; heights = [1, 1, 1, 1]; gs_kw = dict(width_ratios=widths, height_ratios=heights)

	fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(nrows=rows,ncols=cols, gridspec_kw=gs_kw)
	for i in range(len(indices)):
		idx = indices[i]
		image, gt_map, pr_map = images[idx,:].squeeze(), gt_maps[idx,:].squeeze(), pr_maps[idx,:].squeeze()
		image = np.uint8((image-image.min())/(image.max()-image.min())*255)
		err_map = np.abs(gt_map-pr_map)*255; err_map_rgb = np.zeros(image.shape).astype(np.uint8); err_map_rgb[:,:,:-1] = err_map
		gt_map_rgb = np.zeros(image.shape,dtype=np.uint8); # gt_map_rgb[:,:,:-1]=np.uint8((gt_map-gt_map.min())/(gt_map.max()-gt_map.min())*255)
		gt_map_rgb[:,:,:-1]=np.uint8(gt_map*255)
		pr_map_rgb = np.zeros(image.shape,dtype=np.uint8); # pr_map_rgb[:,:,:-1]=np.uint8((pr_map-pr_map.min())/(pr_map.max()-pr_map.min())*255)
		pr_map_rgb[:,:,:-1]=np.uint8(pr_map*255)
		cut_map = io.imread(cut_folder+'/{}'.format(image_ids[idx]))
		cx0 = ax[i,0].imshow(image); cx1 = ax[i,1].imshow(gt_map_rgb); 
		cx2 = ax[i,2].imshow(pr_map_rgb); cx3 = ax[i,3].imshow(err_map_rgb); cx4 = ax[i,6].imshow(cut_map);
		unet_fl1 = io.imread(unet_fl1_dir+'/pred_{}'.format(image_ids[idx].replace('png','tif'))); unet_fl2 = io.imread(unet_fl2_dir+'/pred_{}'.format(image_ids[idx].replace('png','tif')))
# 		unet_fl2 = np.zeros(unet_fl1.shape)
		unet_fl1 = unet_fl1/unet_fl1.max(); unet_fl2 = unet_fl2/unet_fl2.max()
		unet_map = np.stack([unet_fl1, unet_fl2], axis = -1); print(unet_map.min(), unet_map.max())
		#unet_map = unet_map/unet_map.max(); print(unet_map.min(), unet_map.max())
		unet_err_map = np.abs(gt_map-unet_map)*255; unet_err_map_rgb = np.zeros(image.shape).astype(np.uint8); unet_err_map_rgb[:,:,:-1] = unet_err_map
		unet_map_rgb = np.zeros(image.shape,dtype=np.uint8);
		unet_map_rgb[:,:,:-1]=np.uint8(unet_map*255)
		cx5 = ax[i,4].imshow(unet_map_rgb); cx6 = ax[i,5].imshow(unet_err_map_rgb); 
		ax[i,0].set_xticks([]);ax[i,0].set_yticks([]);
		#ax[i,0].tick_params(axis = 'x', labelsize = label_size);ax[i,0].tick_params(axis = 'y', labelsize = label_size);
		ax[i,1].set_xticks([]);ax[i,2].set_xticks([]);ax[i,3].set_xticks([]); ax[i,4].set_xticks([]); ax[i,5].set_xticks([]); ax[i,6].set_xticks([]);
		ax[i,1].set_yticks([]);ax[i,2].set_yticks([]);ax[i,3].set_yticks([]); ax[i,4].set_yticks([]); ax[i,5].set_yticks([]); ax[i,6].set_yticks([]);
		if colorbar:
			#fig.colorbar(cx0, ax = ax[i,0], shrink = 0); 
			cbar = fig.colorbar(cx1, ax = ax[i,1], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
			cbar = fig.colorbar(cx2, ax = ax[i,2], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
			cbar = fig.colorbar(cx3, ax = ax[i,3], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size)
			cbar = fig.colorbar(cx4, ax = ax[i,4], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size)
			cbar = fig.colorbar(cx5, ax = ax[i,5], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
			cbar = fig.colorbar(cx6, ax = ax[i,6], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
		if i == 0:
			ax[i,0].set_title('Image',fontsize=font_size);
			ax[i,1].set_title('Ground Truth',fontsize=font_size);
			ax[i,2].set_title('E-Net (Paired)',fontsize=font_size);
			ax[i,3].set_title('Err Map (E-Net)',fontsize=font_size);
			ax[i,4].set_title('U-Net (Paired)',fontsize=font_size);
			ax[i,5].set_title('Err Map (U-Net)',fontsize=font_size);
			ax[i,6].set_title('Unpaired Method',fontsize=font_size);
	fig.tight_layout(pad=-2)
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=120)

def plot_flu_prediction8(fig_root_dir, images, gt_maps, pr_maps, cut_folder, unet_fl1_dir, unet_fl2_dir, image_ids, nb_images, rand_seed = 3, colorbar = True):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	from skimage import io
	import random
	seed = rand_seed #3
	random.seed(seed)
	font_size = 28; label_size = 18
	#indices = random.sample(range(gt_maps.shape[0]),nb_images)
	rows, cols, size = nb_images,4,6
	
	pre_select_fnames = ['f0_t0_i0_ch0_c2_r3_z1_mhilbert.png', 'f0_t4_i0_ch0_c4_r1_z0_mhilbert.png', 
				'f0_t4_i0_ch0_c2_r2_z1_mhilbert.png', 'f0_t4_i0_ch0_c1_r0_z1_mhilbert.png']
	
	indices = [image_ids.index(fname) for fname in pre_select_fnames]
	
	for i in range(len(indices)):
		example_folder = os.path.join(fig_root_dir, 'example_{:03d}'.format(i)); generate_folder(example_folder)
		cols, rows = 1, 1
		fig1 = Figure(tight_layout=True,figsize=(size-1.0, size-1.2))
		fig2 = Figure(tight_layout=True,figsize=(size, size-1.2))
		fig3 = Figure(tight_layout=True,figsize=(size, size-1.2))
		fig4 = Figure(tight_layout=True,figsize=(size, size-1.2))
		fig5 = Figure(tight_layout=True,figsize=(size, size-1.2))
		fig6 = Figure(tight_layout=True,figsize=(size, size-1.2))
		fig7 = Figure(tight_layout=True,figsize=(size, size-1.2))
		ax = fig1.subplots(nrows=rows,ncols=cols); #ax.set_xticks([]);ax.set_yticks([])
		bx = fig2.subplots(nrows=rows,ncols=cols); #ax.set_xticks([]);ax.set_yticks([])
		cx = fig3.subplots(nrows=rows,ncols=cols); #ax.set_xticks([]);ax.set_yticks([])
		dx = fig4.subplots(nrows=rows,ncols=cols); #ax.set_xticks([]);ax.set_yticks([])
		ex = fig5.subplots(nrows=rows,ncols=cols);
		fx = fig6.subplots(nrows=rows,ncols=cols);
		gx = fig7.subplots(nrows=rows,ncols=cols);
		idx = indices[i]
		image, gt_map, pr_map = images[idx,:].squeeze(), gt_maps[idx,:].squeeze(), pr_maps[idx,:].squeeze()
		image = np.uint8((image-image.min())/(image.max()-image.min())*255)
		err_map = np.abs(gt_map-pr_map)*255; err_map_rgb = np.zeros(image.shape).astype(np.uint8); err_map_rgb[:,:,:-1] = err_map
		gt_map_rgb = np.zeros(image.shape,dtype=np.uint8); # gt_map_rgb[:,:,:-1]=np.uint8((gt_map-gt_map.min())/(gt_map.max()-gt_map.min())*255)
		gt_map_rgb[:,:,:-1]=np.uint8(gt_map*255)
		pr_map_rgb = np.zeros(image.shape,dtype=np.uint8); # pr_map_rgb[:,:,:-1]=np.uint8((pr_map-pr_map.min())/(pr_map.max()-pr_map.min())*255)
		pr_map_rgb[:,:,:-1]=np.uint8(pr_map*255)
		unpaired_map = io.imread(cut_folder+'/{}'.format(image_ids[idx]))
		unet_fl1 = io.imread(unet_fl1_dir+'/pred_{}'.format(image_ids[idx].replace('png','tif'))); unet_fl2 = io.imread(unet_fl2_dir+'/pred_{}'.format(image_ids[idx].replace('png','tif')))
		unet_fl1 = unet_fl1/unet_fl1.max(); unet_fl2 = unet_fl2/unet_fl2.max()
		unet_map = np.stack([unet_fl1, unet_fl2], axis = -1); print(unet_map.min(), unet_map.max())
		unet_err_map = np.abs(gt_map-unet_map)*255; unet_err_map_rgb = np.zeros(image.shape).astype(np.uint8); unet_err_map_rgb[:,:,:-1] = unet_err_map
		unet_map_rgb = np.zeros(image.shape,dtype=np.uint8);
		unet_map_rgb[:,:,:-1]=np.uint8(unet_map*255)

		cax = ax.imshow(image); #cbar = fig1.colorbar(cax, ax = ax, shrink = 0.68); cbar.ax.tick_params(labelsize=label_size)
		cbx = bx.imshow(gt_map_rgb); cbar = fig2.colorbar(cbx, ax = bx, shrink = 0.97); cbar.ax.tick_params(labelsize=label_size)
		ccx = cx.imshow(pr_map_rgb); cbar = fig3.colorbar(ccx, ax = cx, shrink = 0.97); cbar.ax.tick_params(labelsize=label_size)
		cdx = dx.imshow(err_map_rgb); cbar = fig4.colorbar(cdx, ax = dx, shrink = 0.97); cbar.ax.tick_params(labelsize=label_size)
		cex = ex.imshow(unpaired_map); cbar = fig5.colorbar(cdx, ax = ex, shrink = 0.97); cbar.ax.tick_params(labelsize=label_size)
		cfx = fx.imshow(unet_map_rgb); cbar = fig6.colorbar(cdx, ax = fx, shrink = 0.97); cbar.ax.tick_params(labelsize=label_size)
		cgx = gx.imshow(unet_err_map_rgb); cbar = fig7.colorbar(cdx, ax = gx, shrink = 0.97); cbar.ax.tick_params(labelsize=label_size)
		ax.tick_params(axis = 'x', labelsize = label_size); ax.tick_params(axis = 'y', labelsize = label_size);
		bx.tick_params(axis = 'x', labelsize = label_size); bx.tick_params(axis = 'y', labelsize = label_size);
		cx.tick_params(axis = 'x', labelsize = label_size); cx.tick_params(axis = 'y', labelsize = label_size);
		dx.tick_params(axis = 'x', labelsize = label_size); dx.tick_params(axis = 'y', labelsize = label_size);
		ex.tick_params(axis = 'x', labelsize = label_size); ex.tick_params(axis = 'y', labelsize = label_size);
		fx.tick_params(axis = 'x', labelsize = label_size); fx.tick_params(axis = 'y', labelsize = label_size);
		gx.tick_params(axis = 'x', labelsize = label_size); gx.tick_params(axis = 'y', labelsize = label_size);
		canvas = FigureCanvasAgg(fig1); canvas.print_figure(example_folder+'/Image.png', dpi=120)
		canvas = FigureCanvasAgg(fig2); canvas.print_figure(example_folder+'/Ground_truth.png', dpi=120)
		canvas = FigureCanvasAgg(fig3); canvas.print_figure(example_folder+'/EfficientNet_Prediction.png', dpi=120)
		canvas = FigureCanvasAgg(fig4); canvas.print_figure(example_folder+'/EfficientNet_Error_map.png', dpi=120)
		canvas = FigureCanvasAgg(fig5); canvas.print_figure(example_folder+'/Unpaired_pred.png', dpi=120)
		canvas = FigureCanvasAgg(fig6); canvas.print_figure(example_folder+'/UNet_Prediction.png', dpi=120)
		canvas = FigureCanvasAgg(fig7); canvas.print_figure(example_folder+'/UNet_Error_map.png', dpi=120)

# def plot_flu_prediction5(file_name, images, gt_maps, pr_maps, cut_folder, image_ids, png_comm_fnames, tif_comm_fnames, nb_images, rand_seed = 3, colorbar = True):
# 	import matplotlib.pyplot as plt
# 	from matplotlib.backends.backend_agg import FigureCanvasAgg
# 	from matplotlib.figure import Figure
# 	from skimage import io
# 	import random
# 	seed = rand_seed #3
# 	random.seed(seed)
# 	font_size = 28; label_size = 20
# # 	indices = random.sample(range(gt_maps.shape[0]),nb_images)
# 	indices = random.sample(range(len(png_comm_fnames)),nb_images)
# 	rows, cols, size = nb_images,7, 5
# 	widths = [0.8, 1, 1, 1, 1, 1, 1]; heights = [1, 1, 1, 1]; gs_kw = dict(width_ratios=widths, height_ratios=heights)
# 
# 	fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(nrows=rows,ncols=cols, gridspec_kw=gs_kw)
# 	for i in range(len(indices)):
# 		idx_idx = indices[i]; eff_img_id = png_comm_fnames[idx_idx]; eff_index = image_ids.index(eff_img_id)
# 		unet_img_id = tif_comm_fnames[idx_idx]; idx = eff_img_id
# 		image, gt_map, pr_map = images[idx,:].squeeze(), gt_maps[idx,:].squeeze(), pr_maps[idx,:].squeeze()
# 		image = np.uint8((image-image.min())/(image.max()-image.min())*255)
# 		err_map = np.abs(gt_map-pr_map)*255; err_map_rgb = np.zeros(image.shape).astype(np.uint8); err_map_rgb[:,:,:-1] = err_map
# 		gt_map_rgb = np.zeros(image.shape,dtype=np.uint8); # gt_map_rgb[:,:,:-1]=np.uint8((gt_map-gt_map.min())/(gt_map.max()-gt_map.min())*255)
# 		gt_map_rgb[:,:,:-1]=np.uint8(gt_map*255)
# 		pr_map_rgb = np.zeros(image.shape,dtype=np.uint8); # pr_map_rgb[:,:,:-1]=np.uint8((pr_map-pr_map.min())/(pr_map.max()-pr_map.min())*255)
# 		pr_map_rgb[:,:,:-1]=np.uint8(pr_map*255)
# 		cut_map = io.imread(cut_folder+'/{}'.format(image_ids[idx]))
# 		cx0 = ax[i,0].imshow(image); cx1 = ax[i,1].imshow(gt_map_rgb); 
# 		cx2 = ax[i,2].imshow(pr_map_rgb); cx3 = ax[i,3].imshow(err_map_rgb); cx4 = ax[i,6].imshow(cut_map);
# 		unet_folder = '/data/U-net_results/'
# 		fl1_folder = unet_folder +'result_train_glim_to_fl1_fil_20200819_181527_final_model/pred_test/'
# 		fl2_folder = unet_folder +'result_train_glim_to_fl2_fil_20200820_141534_final_model/pred_test/'
# 		unet_pred_fl1 = os.imread(fl1_folder+'pred_{}'.format(unet_img_id)); unet_pred_fl2 = np.zeros(unet_pred_fl1.shape).astype(np.uint8) #unet_pred_fl2 = os.imread()
# 		zero_channel = np.zeros(unet_pred_fl1.shape).astype(np.uint8); print(unet_pred_fl1.max())
# 		unet_pred_fl1_norm = np.uint8((unet_pred_fl1-unet_pred_fl1.min())*255.0/(unet_pred_fl1.max()-unet_pred_fl1.min()))
# 		unet_pred = np.stack([unet_pred_fl1, unet_pred_fl2, zero_channel], axis = -1)
# 		cx4 = ax[i,4].imshow(unet_pred); cx5 = ax[i,5].imshow(unet_pred);
# 		ax[i,0].set_xticks([]);ax[i,0].set_yticks([]);
# 		#ax[i,0].tick_params(axis = 'x', labelsize = label_size);ax[i,0].tick_params(axis = 'y', labelsize = label_size);
# 		ax[i,1].set_xticks([]);ax[i,2].set_xticks([]);ax[i,3].set_xticks([]); ax[i,4].set_xticks([]); ax[i,5].set_xticks([]); ax[i,6].set_xticks([]);
# 		ax[i,1].set_yticks([]);ax[i,2].set_yticks([]);ax[i,3].set_yticks([]); ax[i,4].set_yticks([]); ax[i,5].set_yticks([]); ax[i,6].set_yticks([]);
# 		if colorbar:
# 			#fig.colorbar(cx0, ax = ax[i,0], shrink = 0); 
# 			cbar = fig.colorbar(cx1, ax = ax[i,1], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
# 			cbar = fig.colorbar(cx2, ax = ax[i,2], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
# 			cbar = fig.colorbar(cx3, ax = ax[i,3], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size)
# 			cbar = fig.colorbar(cx4, ax = ax[i,4], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size)
# 			cbar = fig.colorbar(cx5, ax = ax[i,5], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
# 			cbar = fig.colorbar(cx6, ax = ax[i,6], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
# 		if i == 0:
# 			ax[i,0].set_title('Image',fontsize=font_size);
# 			ax[i,1].set_title('Ground Truth',fontsize=font_size);
# 			ax[i,2].set_title('E-Net(Paired)',fontsize=font_size);
# 			ax[i,3].set_title('Err Map (E-Net)',fontsize=font_size);
# 			ax[i,4].set_title('U-Net(Paired)',fontsize=font_size);
# 			ax[i,5].set_title('Err Map (U-Net)',fontsize=font_size);
# 			ax[i,6].set_title('Unpaired Method',fontsize=font_size);
# 	fig.tight_layout(pad=-2)
# 	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=100)
