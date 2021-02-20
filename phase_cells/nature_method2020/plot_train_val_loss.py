import numpy as np
import os
import sys

def plot_separate(file_name, loss_list, title_list):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	rows, cols, size = 1,1,5
	font_size = 30; label_size = 25; line_width = 2.5
	fig = Figure(tight_layout=True,figsize=(8, 6)); ax = fig.subplots(rows,cols)
	ax.plot(loss_list[0],linewidth = line_width);ax.plot(loss_list[1], linewidth=line_width)
	ax.set_ylabel(title_list[0], fontsize = font_size);ax.set_xlabel('Epochs',fontsize = font_size);ax.legend(['Train','Valid'],fontsize = font_size)
	ax.tick_params(axis = 'x', labelsize = label_size); ax.tick_params(axis = 'y', labelsize = label_size);
	ax.set_xlim([0,len(loss_list[0])]); 
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=100)


model_root_dir = '/data/models/report_results/'
model_names = os.listdir(model_root_dir)
nb_epochs = 100
model_name = 'single-net-Unet-bone-efficientnetb3-pre-True-epoch-400-batch-14-lr-0.0005-dim-512-train-900-rot-0-set-live_dead-ext-False-loss-focal-up-upsampling-filters-256-red_factor-0.8-pyr_agg-sum'
# for model_name in model_names:
train_dir = os.path.join(model_root_dir, model_name, 'train_dir')
if os.path.exists(train_dir):
	train_loss = 4*np.loadtxt(train_dir+'/train_loss.txt')
	val_loss = 4*np.loadtxt(train_dir+'/val_loss.txt')

	train_dice = np.loadtxt(train_dir+'/train_f1-score.txt')
	val_dice = np.loadtxt(train_dir+'/val_f1-score.txt')
	if nb_epochs < len(train_loss):
		train_loss = train_loss[:nb_epochs]
		val_loss = val_loss[:nb_epochs]
		train_dice = train_dice[:nb_epochs]
		val_dice = val_dice[:nb_epochs]
	file_name = train_dir + '/loss_dice_history.png'
	loss_list = [train_loss, val_loss]
	metric_list = [train_dice, val_dice]
# 	title_list = ['Focal loss', 'Dice score']
# 	plot_history(file_name, loss_list, metric_list, title_list)
	file_name = './loss_history.png'
	plot_separate(file_name, loss_list, ['Focal loss'])
# 	file_name = train_dir + '/dice_history.png'
# 	plot_separate(file_name, metric_list, ['Dice score'])