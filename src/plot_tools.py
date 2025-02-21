# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# 绘图相关工具

if __name__ == '__main__':
	import sys
	sys.path.append('../')

import os
import numpy
import pandas

from matplotlib import pyplot as plt
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
from sklearn.metrics import auc, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc, precision_recall_curve

from setting import *

# 选择题模型训练作图
def train_plot_choice(model_name,
					  train_logging_dataframe,
					  valid_logging_dataframe=None,
					  train_plot_export_path=None,
					  valid_plot_export_path=None):

	plt.rcParams['figure.figsize'] = (12., 9.)
	# 提取需要使用的数据
	epochs			= train_logging_dataframe['epoch']
	losses			= train_logging_dataframe['loss']
	accuracys		= train_logging_dataframe['accuracy']
	strict_scores	= train_logging_dataframe['strict_score']
	loose_scores	= train_logging_dataframe['loose_score']

	# 横坐标为epoch, 但是要凑到跟其他数据一样的长度
	xs = numpy.linspace(0, epochs.max(), len(epochs))

	# 绘图
	fig = plt.figure(1)

	host = HostAxes(fig, [0.15, 0.1, .65, 0.8])		# 主图: [ 左，下，宽，高 ]
	par1 = ParasiteAxes(host, sharex=host)			# 右侧第一根轴: accuracy
	par2 = ParasiteAxes(host, sharex=host)			# 右侧第二根轴: strict_score
	par3 = ParasiteAxes(host, sharex=host)			# 右侧第二根轴: loose_score
	host.parasites.append(par1)
	host.parasites.append(par2)
	host.parasites.append(par3)

	host.set_ylabel('loss')							# 主纵轴
	host.set_xlabel('epoch')						# 横坐标

	host.axis['right'].set_visible(False)			# 主图的右侧纵轴不显示: 要留给par1显示
	par1.axis['right'].set_visible(True)

	par1.set_ylabel('accuracy')
	par1.axis['right'].major_ticklabels.set_visible(True)
	par1.axis['right'].label.set_visible(True)

	par2.set_ylabel('strict_score')
	new_axisline = par2._grid_helper.new_fixed_axis
	par2.axis['right2'] = new_axisline(loc='right', axes=par2, offset=(45, 0))
	par2.axis['right2'].label.set_visible(True)

	par3.set_ylabel('loose_score')
	new_axisline = par3._grid_helper.new_fixed_axis
	par3.axis['right4'] = new_axisline(loc='right', axes=par3, offset=(90, 0))
	par3.axis['right4'].label.set_visible(True)

	fig.add_axes(host)

	p1, = host.plot(xs, losses, label='loss')
	p2, = par1.plot(xs, accuracys, label='accuracy')
	p3, = par2.plot(xs, strict_scores, label='strict_score')
	p4, = par3.plot(xs, loose_scores, label='loose_score')

	par1.set_ylim(0, 1.2)
	par2.set_ylim(0, 1.2)
	par3.set_ylim(0, 1.2)

	host.axis['left'].label.set_color(p1.get_color())
	par1.axis['right'].label.set_color(p2.get_color())
	par2.axis['right2'].label.set_color(p3.get_color())
	par2.axis['right2'].set_axisline_style('-|>', size=1.5)				# 纵轴的小短横朝向左侧
	par3.axis['right4'].label.set_color(p4.get_color())
	par3.axis['right4'].set_axisline_style('-|>', size=1.5)				# 纵轴的小短横朝向左侧

	host.set_xticks(list(range(max(epochs) + 1)))
	host.legend()

	plt.title(f'Train Plot for {model_name}')
	if train_plot_export_path is None:
		plt.show()
	else:
		plt.savefig(train_plot_export_path)
		plt.close()	# 一定要close, 否则会重复画在一张图上

	# 验证集情况作图
	if valid_logging_dataframe is not None:
		epochs 			= valid_logging_dataframe['epoch']
		accuracys 		= valid_logging_dataframe['accuracy']
		strict_scores 	= valid_logging_dataframe['strict_score']
		loose_scores 	= valid_logging_dataframe['loose_score']
		plt.plot(epochs, accuracys, label='accuracy')
		plt.plot(epochs, strict_scores, label='strict_score')
		plt.plot(epochs, loose_scores, label='loose_score')
		plt.xlabel('epoch')
		plt.ylabel('metric')
		plt.title(f'Valid Plot for {model_name}')
		plt.legend()
		if valid_plot_export_path is None:
			plt.show()
		else:
			plt.savefig(valid_plot_export_path)
			plt.close()	# 一定要close, 否则会重复画在一张图上

# 判断题模型训练作图
def train_plot_judgment(model_name,
						train_logging_dataframe,
						valid_logging_dataframe=None,
						train_plot_export_path=None,
						valid_plot_export_path=None):
	plt.rcParams['figure.figsize'] = (12., 9.)
	columns = train_logging_dataframe.columns
	thresholds = []
	for column in columns:
		if column.startswith('accuracy'):
			thresholds.append(float(column[8: ]))

	# 提取需要使用的数据
	epochs = train_logging_dataframe['epoch']
	losses = train_logging_dataframe['loss']
	accuracys = {threshold: train_logging_dataframe[f'accuracy{threshold}'] for threshold in thresholds}

	# 横坐标为epoch, 但是要凑到跟其他数据一样的长度
	xs = numpy.linspace(0, epochs.max(), len(epochs))

	# 绘图
	fig = plt.figure(1)

	host = HostAxes(fig, [0.15, 0.1, .65, 0.8])		# 主图: [ 左，下，宽，高 ]
	par1 = ParasiteAxes(host, sharex=host)			# 右侧第一根轴: accuracy
	host.parasites.append(par1)

	host.set_ylabel('loss')							# 主纵轴
	host.set_xlabel('epoch')						# 横坐标

	host.axis['right'].set_visible(False)			# 主图的右侧纵轴不显示: 要留给par1显示
	par1.axis['right'].set_visible(True)

	par1.set_ylabel('accuracy')
	par1.axis['right'].major_ticklabels.set_visible(True)
	par1.axis['right'].label.set_visible(True)

	fig.add_axes(host)

	p1, = host.plot(xs, losses, label='loss')

	for threshold in thresholds:
		p2, = par1.plot(xs, accuracys[threshold], label=f'accuracy={threshold}')

	par1.set_ylim(0, 1.2)

	host.axis['left'].label.set_color(p1.get_color())
	par1.axis['right'].label.set_color(p2.get_color())

	host.set_xticks(list(range(max(epochs) + 1)))
	host.legend()

	plt.title(f'Train Plot for {model_name}')
	if train_plot_export_path is None:
		plt.show()
	else:
		plt.savefig(train_plot_export_path)
		plt.close()	# 一定要close, 否则会重复画在一张图上

	# 验证集情况作图
	if valid_logging_dataframe is not None:
		columns = valid_logging_dataframe.columns
		thresholds = []
		for column in columns:
			if column.startswith('accuracy'):
				thresholds.append(float(column[8: ]))
		epochs = valid_logging_dataframe['epoch']
		aucs = valid_logging_dataframe['auc']
		accuracys = {threshold: valid_logging_dataframe[f'accuracy{threshold}'] for threshold in thresholds}
		plt.plot(epochs, aucs, label=f'auc')
		for threshold in thresholds:
			plt.plot(epochs, accuracys[threshold], label=f'accuracy={threshold}')
		plt.xlabel('epoch')
		plt.ylabel('metric')
		plt.title(f'Valid Plot for {model_name}')
		plt.legend()
		if valid_plot_export_path is None:
			plt.show()
		else:
			plt.savefig(valid_plot_export_path)
			plt.close()	# 一定要close, 否则会重复画在一张图上

# 绘制ROC曲线
def plot_roc_curve(target, predict_proba, title=None, export_path=None):
	fpr, tpr, thresholds = roc_curve(target, predict_proba, pos_label=1)
	_auc = auc(fpr, tpr)
	plt.plot(fpr, tpr, 'g-', label=f'ROC(AUC={round(_auc, 3)})')
	plt.plot([0, 1], [0, 1], 'r--', label='Luck')
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve' if title is None else title)
	plt.legend()
	if export_path is None:
		plt.show()
	else:
		plt.savefig(export_path)
		plt.close()

# 绘制PR曲线
def plot_pr_curve(target, predict_proba, title=None, export_path=None):
	precision, recall, thresholds = precision_recall_curve(target, predict_proba)
	diff = float('inf')
	for _precision, _recall, threshold in zip(precision, recall, thresholds):
		if abs(_precision - _recall) < diff:
			balance_point = (_recall, _precision, threshold)
			diff = abs(_precision - _recall)
	plt.plot(recall, precision, 'g-', label=f'PR')
	plt.plot([0, 1], [0, 1], 'y--', label='Balance')
	plt.plot([balance_point[0]], [balance_point[1]], 'ro', label=f'Balance point: ({round(balance_point[0], 3)}, {round(balance_point[0], 3)})')
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('Recall Rate')
	plt.ylabel('Precision Rate')
	plt.title('PR Curve' if title is None else title)
	plt.legend()
	if export_path is None:
		plt.show()
	else:
		plt.savefig(export_path)
		plt.close()
