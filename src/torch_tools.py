# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
# torch模型相关工具

if __name__ == '__main__':
	import sys
	sys.path.append('../')

import torch

from setting import *

# 保存模型: 可以将优化器和规划器以及一些其他内容(如当前轮数与迭代次数)保存到checkpoint中
def save_checkpoint(model, save_path, optimizer=None, scheduler=None, **kwargs):
	checkpoint = kwargs.copy()
	checkpoint['model'] = model.state_dict()
	checkpoint['optimizer'] = optimizer.state_dict() if optimizer is not None else None
	checkpoint['scheduler'] = scheduler.state_dict() if scheduler is not None else None
	torch.save(checkpoint, save_path)

# 加载模型: 可以同时读取优化器和规划器以及一些其他内容(如当前轮数与迭代次数)
def load_checkpoint(model, save_path, optimizer=None, scheduler=None):
	saved_checkpoint = torch.load(save_path)
	model.load_state_dict(saved_checkpoint['model'])
	saved_checkpoint['model'] = model
	if optimizer is not None:
		optimizer.load_state_dict(saved_checkpoint['optimizer'])
		saved_checkpoint['optimizer'] = optimizer
	if scheduler is not None:
		scheduler.load_state_dict(saved_checkpoint['scheduler'])
		saved_checkpoint['scheduler'] = scheduler
	return saved_checkpoint

# 2022/09/16 17:49:51 之前的模型基本都是一个, 现在都是多个模块构成的大模型, 因此checkpoint中需要存储多个模型, 
def save_multi_model_checkpoint(models, save_path, optimizer=None, scheduler=None, **kwargs):
	checkpoint = kwargs.copy()
	checkpoint['models'] = {name: model.state_dict() for name, model in models.items()}
	checkpoint['optimizer'] = optimizer.state_dict() if optimizer is not None else None
	checkpoint['scheduler'] = scheduler.state_dict() if scheduler is not None else None
	torch.save(checkpoint, save_path)

# 2022/09/16 17:49:51 配套的加载模型方法
def load_multi_model_checkpoint(models, save_path, optimizer=None, scheduler=None):
	saved_checkpoint = torch.load(save_path)
	for name, model in models.items():
		model.load_state_dict(saved_checkpoint['models'][name])
		saved_checkpoint['models'][name] = model
	if optimizer is not None:
		optimizer.load_state_dict(saved_checkpoint['optimizer'])
		saved_checkpoint['optimizer'] = optimizer
	if scheduler is not None:
		scheduler.load_state_dict(saved_checkpoint['scheduler'])
		saved_checkpoint['scheduler'] = scheduler
	return saved_checkpoint
