import numpy as np
import os
import sys

def plot_sample_efficiency(file_name, num_samples, pretrain_scores, scratch_scores, metric_name):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	rows, cols, size = 1,1,5 
	font_size = 30; label_size = 25; line_width = 2.5; markder_size = 10 
	fig = Figure(tight_layout=True,figsize=(8, 6)); ax = fig.subplots(rows,cols) 
	ax.plot(num_samples, pretrain_scores,linewidth = line_width, marker = 's', markersize=markder_size);ax.plot(num_samples, scratch_scores, linewidth=line_width, marker = 'o', markersize=markder_size) 
	ax.set_ylabel(metric_name, fontsize = font_size-2);ax.set_xlabel('Number of training images',fontsize = font_size);ax.legend(['Pretrain','Scratch'],fontsize = font_size) 
	ax.tick_params(axis = 'x', labelsize = label_size); ax.tick_params(axis = 'y', labelsize = label_size); 
	ax.set_xlim([100-10, num_samples[-1]+10]) 
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=100) 

pretrain_scores = np.array([0.5784, 0.6756, 0.7117, 0.7137, 0.7352])*100 
scratch_scores = np.array([0.4585, 0.5193, 0.5790, 0.6218, 0.6767])*100 
num_samples = [100,300,500,700,899] 
metric_name = 'Average pixel f1-score(%)' 
file_name = 'sample_efficiency.png' 
plot_sample_efficiency(file_name, num_samples, pretrain_scores, scratch_scores, metric_name)