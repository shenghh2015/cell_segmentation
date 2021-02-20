import numpy as np
import os
import sys

def plot_history(file_name, loss_list, metric_list, title_list):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	rows, cols, size = 1,2,5
	font_size = 15
	fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(rows,cols)
	ax[0].plot(loss_list[0]);ax[0].plot(loss_list[1])
	ax[0].set_ylabel(title_list[0], fontsize = font_size);ax[0].set_xlabel('Epochs',fontsize = font_size);ax[0].legend(['train','valid'],fontsize = font_size)
	ax[1].plot(metric_list[0]);ax[1].plot(metric_list[1])
	ax[1].set_ylabel(title_list[1],fontsize = font_size);ax[1].set_xlabel('Epochs',fontsize = font_size);ax[1].legend(['train','valid'],fontsize = font_size)
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=100)


model_root_dir = '/data/models/lr_reduce_factor/'
model_names = os.listdir(model_root_dir)
#model_name = 'single-net-Unet-bone-efficientnetb2-pre-True-epoch-200-batch-3-lr-0.0005-dim-992-train-1100-rot-0-set-cell_cycle_1984-ext-True-loss-focal+dice-up-upsampling-filters-256'
for model_name in model_names:
	train_dir = os.path.join(model_root_dir, model_name, 'train_dir')
	if os.path.exists(train_dir):
		train_loss = np.loadtxt(train_dir+'/train_loss.txt')
		val_loss = np.loadtxt(train_dir+'/val_loss.txt')

		train_dice = np.loadtxt(train_dir+'/train_f1-score.txt')
		val_dice = np.loadtxt(train_dir+'/val_f1-score.txt')

		file_name = train_dir + '/loss_dice_history.png'
		loss_list = [train_loss, val_loss]
		metric_list = [train_dice, val_dice]
		title_list = ['Dice_focal_loss', 'Dice_cofficient']
		plot_history(file_name, loss_list, metric_list, title_list)
