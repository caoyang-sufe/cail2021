# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# 模型训练绘图脚本

import os
import pandas


from setting import *

from src.plot_tools import train_plot_choice


# 2022/05/24 22:43:43 将训练生成好的记录放置在temp/summary/<modelname>/文件夹下, 即可自动生成训练图像至image文件夹下
def easy_plot_choice(model_name):
	for mode in ['train_kd', 'train_ca']:
		train_logging_dataframe = pandas.read_csv(os.path.join(TEMP_DIR, 'summary', model_name, f'{model_name}_{mode}.csv'), header=0, sep='\t')
		valid_logging_dataframe = pandas.read_csv(os.path.join(TEMP_DIR, 'summary', model_name, f'{model_name}_{mode.replace("train", "valid")}.csv'), header=0, sep='\t')
		train_plot_choice(model_name=f'{model_name}{mode.split("_")[-1].upper()}', 
						  train_logging_dataframe=train_logging_dataframe, 
						  valid_logging_dataframe=valid_logging_dataframe,
						  train_plot_export_path=os.path.join(IMAGE_DIR, f'{model_name}_{mode}.png'),
						  valid_plot_export_path=os.path.join(IMAGE_DIR, f'{model_name}_{mode.replace("train", "valid")}.png'))	
	
	

if __name__ == '__main__':

	model_name = 'BaseChoiceModel'
	easy_plot_choice(model_name)
