# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
# 模型训练

import os
import time
import json
import torch
import numpy
import pandas
import logging

from pprint import pprint
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, BCELoss
from torch.optim import SGD, Adam, lr_scheduler
from sklearn.metrics import auc, confusion_matrix, accuracy_score, roc_auc_score

from setting import *
from config import QAModelConfig, DatasetConfig

from src.dataset import generate_basic_dataloader, generate_parse_tree_dataloader
from src.evaluation_tools import evaluate_qa_model_choice, evaluate_qa_model_judgment, evaluate_classifier
from src.plot_tools import plot_roc_curve, plot_pr_curve
from src.qa_model import *
from src.qa_module import *
from src.graph_tools import generate_pos_tags_from_parse_tree, traverse_parse_tree
from src.torch_tools import save_checkpoint, load_checkpoint, save_multi_model_checkpoint, load_multi_model_checkpoint
from src.utils import initialize_logger, terminate_logger, load_args, save_args


# 选择题模型训练
def train_choice_model(mode, model_name, current_epoch=0, checkpoint_path=None, do_save_checkpoint=True, is_debug=False, **kwargs):
	assert mode in ['train', 'train_kd', 'train_ca']
	logger = initialize_logger(filename=os.path.join(LOGGING_DIR, f'{time.strftime("%Y%m%d")}_{model_name}_{mode}'), filemode='w')
	logger.info(f'args: {kwargs}')
	
	logging.info(f'Using {DEVICE}')
	
	# 配置模型
	args_1 = load_args(Config=QAModelConfig)							# 问答模型配置
	# 根据kwargs调整问答模型配置的参数
	for key, value in kwargs.items():
		args_1.__setattr__(key, value)
	model = eval(model_name)(args=args_1).to(DEVICE)					# 构建模型

	# 配置训练集
	args_2 = load_args(Config=DatasetConfig)
	# 根据kwargs调整问答模型配置的参数
	for key, value in kwargs.items():
		args_2.__setattr__(key, value)																																
	train_dataloader = generate_basic_dataloader(args=args_2, mode=mode, do_export=False, pipeline='choice', for_debug=is_debug)	

	# 提取训练参数值
	num_epoch = args_1.num_epoch																				
	learning_rate = args_1.learning_rate
	lr_multiplier = args_1.lr_multiplier
	weight_decay = args_1.weight_decay

	# 配置验证集: do_valid为True时生效
	if args_2.do_valid:
		valid_dataloader = generate_basic_dataloader(args=args_2, mode=mode.replace('train', 'valid'), do_export=False, pipeline='choice', for_debug=is_debug)	
		valid_logging = {
			'epoch'			: [],
			'accuracy'		: [],
			'strict_score'	: [],
			'loose_score'	: [],
		}

	loss_function = CrossEntropyLoss()													# 构建损失函数
	optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)	# 构建优化器
	exp_lr_scheduler = None
	# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_multiplier)	# 构建学习率规划期

	# 模型训练
	train_logging = {
		'epoch'			: [],
		'iteration'		: [],
		'loss'			: [],
		'accuracy'		: [],	# 记录精确度: 指16分类的精确度
		'strict_score'	: [],	# 记录严格得分
		'loose_score'	: [],	# 记录松弛得分
	}
	
	# 2022/01/11 17:25:56 调用checkpoint
	if checkpoint_path is not None:
		checkpoint = load_checkpoint(model=model, save_path=checkpoint_path, optimizer=optimizer, scheduler=exp_lr_scheduler)
		model = checkpoint['model'].to(DEVICE)
		optimizer = checkpoint['optimizer']
		exp_lr_scheduler = checkpoint['scheduler']
		if checkpoint.get('train_logging') is not None:
			train_logging = checkpoint['train_logging'].copy()
		if args_2.do_valid and checkpoint.get('valid_logging') is not None:
			valid_logging = checkpoint['valid_logging'].copy()	

	for epoch in range(current_epoch, num_epoch):
		model.train()
		for iteration, data in enumerate(train_dataloader):
			for key in data.keys():
				if isinstance(data[key], torch.Tensor):
					data[key] = Variable(data[key]).to(DEVICE)
			optimizer.zero_grad()
			output = model(data)
			
			evaluation_summary = evaluate_qa_model_choice(input=data, output=output, mode='train')
			strict_score = evaluation_summary['strict_score']
			loose_score = evaluation_summary['loose_score']
			accuracy = evaluation_summary['accuracy']
			loss = loss_function(output, data['label_choice'])
			loss.backward()
			optimizer.step()
			
			logging.info(f'train | epoch: {epoch} - iteration - {iteration} - loss: {loss.item()} - accuracy: {accuracy} - score: {strict_score, loose_score}')
		
			train_logging['epoch'].append(epoch)
			train_logging['iteration'].append(iteration)
			train_logging['loss'].append(loss.item())
			train_logging['accuracy'].append(accuracy)
			train_logging['strict_score'].append(strict_score)
			train_logging['loose_score'].append(loose_score)
		
		if exp_lr_scheduler is not None:
			exp_lr_scheduler.step()
		
		# 验证集评估
		if args_1.do_valid:
			model.eval()
			accuracys = []
			strict_scores = []
			loose_scores = []
			total_size = 0
			with torch.no_grad():
				for data in valid_dataloader:
					_batch_size = len(data['id'])
					total_size += _batch_size
					for key in data.keys():
						if isinstance(data[key], torch.Tensor):
							data[key] = data[key].to(DEVICE)
					output = model(data)
					evaluation_summary = evaluate_qa_model_choice(input=data, output=output, mode='valid')
					strict_score = evaluation_summary['strict_score']
					loose_score = evaluation_summary['loose_score']
					accuracy = evaluation_summary['accuracy']
					accuracys.append(accuracy * _batch_size)
					strict_scores.append(strict_score * _batch_size)
					loose_scores.append(loose_score * _batch_size)
			mean_accuracy = numpy.sum(accuracys) / total_size
			mean_strict_score = numpy.sum(strict_scores) / total_size
			mean_loose_score = numpy.sum(loose_scores) / total_size
			valid_logging['epoch'].append(epoch)
			valid_logging['accuracy'].append(mean_accuracy)
			valid_logging['strict_score'].append(mean_strict_score)
			valid_logging['loose_score'].append(mean_loose_score)
			logging.info(f'valid | epoch: {epoch} - accuracy: {mean_accuracy} - score: {mean_strict_score, mean_loose_score}')
			
		# 保存模型
		if do_save_checkpoint:
			save_checkpoint(model=model, 
							save_path=os.path.join(CHECKPOINT_DIR, f'{model_name}_{mode}_{epoch}.h5'), 
							optimizer=optimizer, 
							scheduler=exp_lr_scheduler,
							train_logging=train_logging,
							valid_logging=valid_logging if args_2.do_valid else None)
	
		# 2021/12/20 22:20:40 每个epoch结束都记录一下结果
		train_logging_dataframe = pandas.DataFrame(train_logging, columns=list(train_logging.keys()))
		train_logging_dataframe.to_csv(os.path.join(TEMP_DIR, f'{model_name}_{mode}.csv'), header=True, index=False, sep='\t')
		if args_1.do_valid:
			valid_logging_dataframe = pandas.DataFrame(valid_logging, columns=list(valid_logging.keys()))
			valid_logging_dataframe.to_csv(os.path.join(TEMP_DIR, f'{model_name}_{mode.replace("train", "valid")}.csv'), header=True, index=False, sep='\t')
	
	terminate_logger(logger=logger)
	
	train_logging_dataframe = pandas.DataFrame(train_logging, columns=list(train_logging.keys()))
	train_logging_dataframe.to_csv(os.path.join(TEMP_DIR, f'{model_name}_{mode}.csv'), header=True, index=False, sep='\t')
	if args_1.do_valid:
		valid_logging_dataframe = pandas.DataFrame(valid_logging, columns=list(valid_logging.keys()))
		valid_logging_dataframe.to_csv(os.path.join(TEMP_DIR, f'{model_name}_{mode.replace("train", "valid")}.csv'), header=True, index=False, sep='\t')
		return train_logging_dataframe, valid_logging_dataframe
	return train_logging_dataframe

# 判断题模型训练
def train_judgment_model(mode, model_name, current_epoch=0, checkpoint_path=None, do_save_checkpoint=True, is_debug=False, **kwargs):
	assert mode in ['train', 'train_kd', 'train_ca']
	logger = initialize_logger(filename=os.path.join(LOGGING_DIR, f'{time.strftime("%Y%m%d")}_{model_name}_{mode}'), filemode='w')
	logger.info(f'args: {kwargs}')
	
	logging.info(f'Using {DEVICE}')
	
	# 配置模型
	args_1 = load_args(Config=QAModelConfig)							# 问答模型配置
	# 根据kwargs调整问答模型配置的参数
	for key, value in kwargs.items():
		args_1.__setattr__(key, value)
	model = eval(model_name)(args=args_1).to(DEVICE)					# 构建模型
	
	# 配置训练集
	args_2 = load_args(Config=DatasetConfig)
	# 根据kwargs调整问答模型配置的参数
	for key, value in kwargs.items():
		args_2.__setattr__(key, value)																																
	train_dataloader = generate_basic_dataloader(args=args_2, mode=mode, do_export=False, pipeline='judgment', for_debug=is_debug)	
	
	# 提取训练参数值
	num_epoch = args_1.num_epoch																				
	learning_rate = args_1.learning_rate
	lr_multiplier = args_1.lr_multiplier
	weight_decay = args_1.weight_decay
	test_thresholds = args_1.test_thresholds
	do_valid_plot = args_1.do_valid_plot

	# 配置验证集: do_valid为True时生效
	if args_2.do_valid:
		valid_dataloader = generate_basic_dataloader(args=args_2, mode=mode.replace('train', 'valid'), do_export=False, pipeline='judgment', for_debug=is_debug)	
		valid_logging = {
			'epoch'	: [],
			'auc'	: [],
		}
		# 记录每个测试阈值的精确度情况
		for threshold in test_thresholds:
			valid_logging[f'accuracy{threshold}'] = []
		valid_target = valid_dataloader.dataset.data['label_judgment'].values			# 验证集标签全集, 用于进行AUC等评估

	loss_function = BCELoss()															# 构建损失函数: 二分类交叉熵
	optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)	# 构建优化器
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_multiplier)	# 构建学习率规划期

	# 模型训练
	train_logging = {
		'epoch'		: [],
		'iteration'	: [],
		'loss'		: [],
	}
	for threshold in test_thresholds:
		train_logging[f'accuracy{threshold}'] = []
	
	# 2022/01/11 17:25:56 调用checkpoint
	if checkpoint_path is not None:
		checkpoint = load_checkpoint(model=model, save_path=checkpoint_path, optimizer=optimizer, scheduler=exp_lr_scheduler)
		model = checkpoint['model'].to(DEVICE)
		optimizer = checkpoint['optimizer']
		exp_lr_scheduler = checkpoint['scheduler']
		if checkpoint.get('train_logging') is not None:
			train_logging = checkpoint['train_logging'].copy()
		if args_2.do_valid and checkpoint.get('valid_logging') is not None:
			valid_logging = checkpoint['valid_logging'].copy()

	for epoch in range(current_epoch, num_epoch):
		model.train()
		for iteration, data in enumerate(train_dataloader):
			for key in data.keys():
				if isinstance(data[key], torch.Tensor):
					data[key] = Variable(data[key]).to(DEVICE)
			optimizer.zero_grad()
			output = model(data)
			evaluation_summary = evaluate_qa_model_judgment(input=data, output=output, mode='train', thresholds=test_thresholds)
			accuracy = evaluation_summary['accuracy']
			loss = loss_function(output, data['label_judgment'].float())# BCELoss或MSELoss要求两个输入都是浮点数
			loss.backward()
			optimizer.step()
			
			logging.info(f'train | epoch: {epoch} - iteration - {iteration} - loss: {loss.item()} - accuracy: {accuracy}')
			
			# 记录模型训练情况
			train_logging['epoch'].append(epoch)
			train_logging['iteration'].append(iteration)
			train_logging['loss'].append(loss.item())
			for threshold in test_thresholds:
				train_logging[f'accuracy{threshold}'].append(accuracy[threshold])

		exp_lr_scheduler.step()
		
		# 验证集评估
		if args_1.do_valid:
			model.eval()
			accuracys = {threshold: [] for threshold in test_thresholds}
			total_size = 0
			valid_predict_probas = []									# 存放模型预测输出的验证集概率值
			with torch.no_grad():
				for data in valid_dataloader:
					_batch_size = len(data['id'])
					total_size += _batch_size
					for key in data.keys():
						if isinstance(data[key], torch.Tensor):
							data[key] = data[key].to(DEVICE)
					output = model(data)
					valid_predict_probas.append(output)
					evaluation_summary = evaluate_qa_model_judgment(input=data, output=output, mode='valid', thresholds=test_thresholds)
					accuracy = evaluation_summary['accuracy']
					for threshold in test_thresholds:
						accuracys[threshold].append(accuracy[threshold] * _batch_size)

			# 计算AUC值
			valid_predict_proba = torch.cat(valid_predict_probas).cpu().numpy()		# 必须先转到CPU上才能转换为numpy数组
			auc = roc_auc_score(valid_target, valid_predict_proba)
			valid_logging['auc'].append(auc)			
			
			# 计算每个测试阈值下的精确度
			valid_logging['epoch'].append(epoch)
			threshold2accuracy = {}
			for threshold in test_thresholds:
				mean_accuracy = numpy.sum(accuracys[threshold]) / total_size
				threshold2accuracy[threshold] = mean_accuracy
				valid_logging[f'accuracy{threshold}'].append(mean_accuracy)

			logging.info(f'valid | epoch: {epoch} - accuracy: {threshold2accuracy} - AUC: {auc}')
			
			# 绘制ROC曲线与PR曲线
			if do_valid_plot:
				plot_roc_curve(target=valid_target, 
							   predict_proba=valid_predict_proba,
							   title=f'ROC Curve of {model_name} in Epoch {epoch} and Mode {mode}', 
							   export_path=os.path.join(IMAGE_DIR, f'roc_{model_name}_{mode}_{epoch}.png'))
				plot_pr_curve(target=valid_target, 
							  predict_proba=valid_predict_proba, 
							  title=f'PR Curve of {model_name} in Epoch {epoch} and Mode {mode}', 
							  export_path=os.path.join(IMAGE_DIR, f'pr_{model_name}_{mode}_{epoch}.png'))
		
		# 保存模型
		if do_save_checkpoint:
			save_checkpoint(model=model, 
							save_path=os.path.join(CHECKPOINT_DIR, f'{model_name}_{mode}_{epoch}.h5'), 
							optimizer=optimizer, 
							scheduler=exp_lr_scheduler,
							train_logging=train_logging,
							valid_logging=valid_logging if args_2.do_valid else None)
		
		# 2021/12/20 22:20:40 每个epoch结束都记录一下结果
		train_logging_dataframe = pandas.DataFrame(train_logging, columns=list(train_logging.keys()))
		train_logging_dataframe.to_csv(os.path.join(TEMP_DIR, f'{model_name}_{mode}.csv'), header=True, index=False, sep='\t')
		if args_1.do_valid:
			valid_logging_dataframe = pandas.DataFrame(valid_logging, columns=list(valid_logging.keys()))
			valid_logging_dataframe.to_csv(os.path.join(TEMP_DIR, f'{model_name}_{mode.replace("train", "valid")}.csv'), header=True, index=False, sep='\t')
	
	terminate_logger(logger=logger)	# 终止日志
	
	train_logging_dataframe = pandas.DataFrame(train_logging, columns=list(train_logging.keys()))
	train_logging_dataframe.to_csv(os.path.join(TEMP_DIR, f'{model_name}_{mode}.csv'), header=True, index=False, sep='\t')
	if args_1.do_valid:
		valid_logging_dataframe = pandas.DataFrame(valid_logging, columns=list(valid_logging.keys()))
		valid_logging_dataframe.to_csv(os.path.join(TEMP_DIR, f'{model_name}_{mode.replace("train", "valid")}.csv'), header=True, index=False, sep='\t')
		return train_logging_dataframe, valid_logging_dataframe
		
	return train_logging_dataframe

# 2022/09/06 22:48:16 训练句法树形式的模型
def train_parse_tree_choice_model(mode, model_name, current_epoch, checkpoint_path=None, do_save_checkpoint=True, is_debug=False, **kwargs):
	assert mode in ['train', 'train_kd', 'train_ca']
	logger = initialize_logger(filename=os.path.join(LOGGING_DIR, f'{time.strftime("%Y%m%d")}_{model_name}_{mode}'), filemode='w')
	logger.info(f'args: {kwargs}')
	
	logging.info(f'Using {DEVICE}')
	
	# 问答模型配置
	args_1 = load_args(Config=QAModelConfig)
	for key, value in kwargs.items():
		args_1.__setattr__(key, value)

	# 提取训练参数值
	assert args_1.train_batch_size == args_1.valid_batch_size == 1, f'Expect batch size be equal to 1 in this training architecture but got {args_1.train_batch_size} and {args_1.valid_batch_size}'
	num_epoch = args_1.num_epoch																				
	learning_rate = args_1.learning_rate
	lr_multiplier = args_1.lr_multiplier
	weight_decay = args_1.weight_decay
	
	parameters = []															# 存储模型的所有参数
	
	# 初始化各个树节点模块
	question_tree_node_encoders = {}										# 存储题库句法树节点模块的字典(题干)
	for tag_name in STANFORD_SYNTACTIC_TAG:									# 为每一个句法树节点标签生成
		tree_node_encoder = TreeRNNEncoder(args=args_1, tag_name=tag_name)		# 初始化节点模块	
		question_tree_node_encoders[tag_name] = tree_node_encoder				# 存储节点模块
		parameters.append({'params': tree_node_encoder.parameters()})			# 存储节点模块的参数	
	
	option_tree_node_encoders = {}											# 存储题库句法树节点模块的字典(选项)
	for tag_name in STANFORD_SYNTACTIC_TAG:									# 为每一个句法树节点标签生成
		tree_node_encoder = TreeRNNEncoder(args=args_1, tag_name=tag_name)		# 初始化节点模块	
		option_tree_node_encoders[tag_name] = tree_node_encoder					# 存储节点模块
		parameters.append({'params': tree_node_encoder.parameters()})			# 存储节点模块的参数	

	if args_1.use_reference:
		reference_tree_node_encoders = {}									# 存储参考文档句法树节点模块的字典
		for tag_name in STANFORD_SYNTACTIC_TAG:								# 为每一个句法树节点标签生成
			tree_node_encoder = TreeRNNEncoder(args=args_1, tag_name=tag_name)	# 初始化节点模块	
			reference_tree_node_encoders[tag_name] = tree_node_encoder			# 存储节点模块
			parameters.append({'params': tree_node_encoder.parameters()})		# 存储节点模块的参数
			
	# 初始化解题模型
	model = eval(model_name)(args=args_1).to(DEVICE)
	parameters.append({'params': model.parameters()})
	
	# 配置训练数据集
	args_2 = load_args(Config=DatasetConfig)
	for key, value in kwargs.items():
		args_2.__setattr__(key, value)																																
	train_dataloader = generate_parse_tree_dataloader(args=args_2, mode=mode, do_export=False, pipeline='choice', for_debug=is_debug)	

	# 配置验证数据集
	if args_2.do_valid:
		valid_dataloader = generate_parse_tree_dataloader(args=args_2, mode=mode.replace('train', 'valid'), do_export=False, pipeline='choice', for_debug=is_debug)	
		valid_logging = {
			'epoch'			: [],
			'accuracy'		: [],
			'strict_score'	: [],
			'loose_score'	: [],
		}
	
	loss_function = CrossEntropyLoss()													# 构建损失函数
	optimizer = Adam(params=parameters, lr=learning_rate, weight_decay=weight_decay)	# 构建优化器
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_multiplier)	# 构建学习率规划期

	# 模型训练
	train_logging = {
		'epoch'			: [],
		'iteration'		: [],
		'loss'			: [], 
		'accuracy'		: [],	# 记录精确度: 指16分类的精确度
		'strict_score'	: [],	# 记录严格得分
		'loose_score'	: [],	# 记录松弛得分
	}
	
	# 2022/09/14 10:54:08 调用checkpoint: 这个与上面的方法稍有区别, 因为模型不止一个, 暂时不实现
	if checkpoint_path is not None:
		# 整理所有模型
		models = {}
		for tag_name, encoder in question_tree_node_encoders.items():
			models[f'question_{tag_name}'] = encoder
		for tag_name, encoder in option_tree_node_encoders.items():
			models[f'option_{tag_name}'] = encoder
		if args_1.use_reference:
			for tag_name, encoder in reference_tree_node_encoders.items():
				models[f'reference_{tag_name}'] = encoder
		models['main'] = model
		checkpoint = load_multi_model_checkpoint(models=models, 
												 save_path=checkpoint_path, 
												 optimizer=optimizer, 
												 scheduler=exp_lr_scheduler)
		
		# 将模型部署到指定设备上
		for name in checkpoint['models']:
			if name == 'main':
				model = checkpoint['models'][name].to(DEVICE)
			else:
				suffix, tag_name = name.split('_', 1)
				if suffix == 'question':
					question_tree_node_encoders[tag_name] = checkpoint['models'][name].to(DEVICE)	
				elif suffix == 'option':
					option_tree_node_encoders[tag_name] = checkpoint['models'][name].to(DEVICE)
				elif suffix == 'reference':
					reference_tree_node_encoders[tag_name] = checkpoint['models'][name].to(DEVICE)
				else:
					raise Exception(f'Unknown model name: {name}')
		optimizer = checkpoint['optimizer']
		exp_lr_scheduler = checkpoint['scheduler']
		if checkpoint.get('train_logging') is not None:
			train_logging = checkpoint['train_logging'].copy()
		if args_2.do_valid and checkpoint.get('valid_logging') is not None:
			valid_logging = checkpoint['valid_logging'].copy()	
	
	logging.info('开始训练模型...')
	
	# 2022/09/16 14:49:18 根据句法树设计TreeRNNEncoder模块
	# 2022/09/16 14:49:18 题干与选项的处理是完全相同的, _field可选值为question, option_a, option_b, option_c, option_d
	def _easy_build_module_question(_field='question'):
		_encoder_outputs = []	# 每个字段中会包含若干句法树, 每个句法树都会有一个输出
		_index_start_from = 0	# 每段话通常由有多个句法树构成, 需要记录应当从词嵌入的第几个位置开始索引
		for _parse_tree in data[0][f'{_field}_parse_tree']:
			_traverse_result, _node_id2data = traverse_parse_tree(parse_tree=_parse_tree)							# 得到句法树的层次遍历结果
			_text_nodes = list(filter(lambda _node_id: _node_id2data[_node_id]['is_text'], _node_id2data))			# 筛选出所有的文本, 即叶子节点
			_text_nodes_index = {_text_node: _i + _index_start_from for _i, _text_node in enumerate(_text_nodes)}	# 逆向索引方便调用对应的词嵌入
			_index_start_from += len(_text_nodes)
			
			# 2022/09/16 s14:59:34 复制一份与_traverse_result结构完全相同的列表_traverse_tensor, 用于存储句法树上每个节点目前的张量情况
			_traverse_tensor = []
			for _i in range(len(_traverse_result)):
				# 遍历每个节点层并复制结构
				_traverse_tensor.append([])
				for _j in range(len(_traverse_result[_i])):
					# 遍历当前节点层的每个节点组并复制结构
					_traverse_tensor[-1].append([])
					for _k in range(len(_traverse_result[_i][_j])):
						# 遍历当前节点组的每个节点并初始化
						if _traverse_result[_i][_j][_k] in _text_nodes:
							# 若为文本节点直接初始化为其词嵌入
							_traverse_tensor[-1][-1].append(data[0][f'{_field}_vector'][_text_nodes_index[_traverse_result[_i][_j][_k]]])
						else:
							# 否则初始化为None
							_traverse_tensor[-1][-1].append(None)

			for _i in range(len(_traverse_result) - 1, 0, -1):
				# 自底向上逆序遍历traverse_result
				_child_node_groups = _traverse_result[_i]		# 当前子节点层
				_parent_node_groups = _traverse_result[_i - 1]	# 当前父节点层
				
				# 2022/09/16 11:15:33 将父节点层中的所有节点展开, 此时每个父节点对应子节点层中的每个节点组
				# 2022/09/16 11:15:33 实际情况中由于靠后的若干父节点都没有子节点, 因此很可能子节点层中的节点组数量少于父节点数量, 但是绝不可能多于
				_parent_nodes = []								# 存储展开后的父节点
				_parent_nodes_index = {}						# 存储展开的父节点的原先位置, 便于在traverse_tensor中进行索引
				for _j in range(len(_parent_node_groups)):
					for _k in range(len(_parent_node_groups[_j])):
						_parent_nodes.append(_parent_node_groups[_j][_k])
						_parent_nodes_index[_parent_node_groups[_j][_k]] = (_i - 1, _j, _k)
				
				_num_child_node_groups = len(_child_node_groups)
				_num_parent_nodes = len(_parent_nodes)

				assert _num_child_node_groups <= _num_parent_nodes, f'The number of node groups in child layer ({_num_child_node_groups}) is larger than the number of nodes in parent layer ({_num_parent_nodes})'
				for _j in range(_num_child_node_groups, _num_parent_nodes):
					assert _node_id2data[_parent_nodes[_j]]['is_text'], 'Those which have no child must be text !'
				
				# 搭建模块
				for _j in range(_num_child_node_groups):
					# 遍历所有子节点组(及其对应的父节点)
					_child_node_group = _child_node_groups[_j]														# 子节点组
					if not _child_node_group:
						# 子节点组为空, 说明上一层对应的节点已经是叶子节点了, 无需处理
						continue
					_parent_node = _parent_nodes[_j]																# 对应的父节点
					_parent_node_name = _node_id2data[_parent_node]['name']											# 获得父节点所表示的STANFORD_SYNTACTIC_TAG
					_index_i, _index_j, _index_k = _parent_nodes_index[_parent_node]								# 父节点在句法树中的坐标
					assert not list(filter(lambda x: x is None, _traverse_tensor[_i][_j])), 'Child node is None'	# 此时子节点在traverse_tensor中的对应元素应该非None, 而是已经计算好的张量
					assert _traverse_tensor[_index_i][_index_j][_index_k] is None									# 此时父节点在traverse_tensor中的对应元素应该为None, 这里需要计算好再填进去
					
					# 调用父节点对应的TreeRNNEncoder模块计算子节点组中所有张量的编码表示(y)
					if isinstance(_traverse_tensor[_i][_j][0], int):
						x = torch.LongTensor(_traverse_tensor[_i][_j]).unsqueeze(0)
					elif isinstance(_traverse_tensor[_i][_j][0], numpy.ndarray):
						x = torch.FloatTensor(numpy.vstack(_traverse_tensor[_i][_j])).unsqueeze(0)
					elif isinstance(_traverse_tensor[_i][_j][0], torch.Tensor):
						x = torch.vstack(_traverse_tensor[_i][_j]).unsqueeze(0)
					else:
						raise Exception(f'Unknown data type: {type(_traverse_tensor[_i][_j])}')
					if _field == 'question':
						y = question_tree_node_encoders[_parent_node_name](x)
					elif _field.startswith('option'):
						y = option_tree_node_encoders[_parent_node_name](x)
					else:
						raise Exception(f'Unknown field name: {_field}')
					_traverse_tensor[_index_i][_index_j][_index_k] = y									# 更新_traverse_tensor
			# 根节点处的输出作为最终输出
			_output = _traverse_tensor[0][0][0]									
			_encoder_outputs.append(_output.to(DEVICE))
		
		# 2022/09/16 14:43:34 理论上句法树中所有的叶子节点数量刚好与data中的分词序列长度相同
		# 2022/09/16 14:43:34 虽然没有进行断言验证, 但是它们的顺序也是刚好相同, 这样就非常便于下面的处理
		_num_text_nodes = _index_start_from
		_num_tokens = len(data[0][f'{_field}_vector'])
		assert _num_text_nodes == _num_tokens, f'The number of text node in question parse tree ({_num_text_nodes}) do not match that in dataloader ({_num_tokens})'	
		return _encoder_outputs

	# 2022/09/19 13:01:24 为参考
	def _easy_build_module_reference(_filed='reference'):
		
		pass

	for epoch in range(current_epoch, num_epoch):
		for iteration, data in enumerate(train_dataloader):
			# data的数据格式如下, data是一个列表, 其中包含batch_size个pandas.Series(DataFrame的一行)
			# [id                                                                1_6055
			# type                                                                   0
			# question               [3119, 2268, 674, 13, 1361, 8, 676, 45, 13, 24...
			# option_a				 [811, 14976, 674, 6088, 5334, 654, 19633, 674,...
			# option_b       		 [674, 7988, 4619, 13, 149, 1026, 4594, 5, 2258...
			# option_c       		 [2426, 2258, 2635, 277, 1640, 1362, 1278, 7990...
			# option_d        		 [674, 676, 8, 676, 13, 24023, 811, 1371, 2172]
			# question_parse_tree    [(ROOT (FRAG (DNP (NP (NP (ADJP (JJ 下列)) (DNP ...
			# option_a_parse_tree    [(ROOT (IP (VP (PP (P 由) (NP (NN 每届) (NN 全国人民代...
			# option_b_parse_tree    [(ROOT (NP (CP (IP (NP (NN 全国人民代表大会) (NN 任期)) ...
			# option_c_parse_tree    [(ROOT (IP (VP (ADVP (CS 如果)) (VP (VV 全国人民代表大会...
			# option_d_parse_tree    [(ROOT (IP (NP (DNP (NP (NP (NR 全国人民代表大会)) (NP...
			# label_choice                                                          15
			# Name: 5, dtype: object]
			
			model.train()
			
			# 2022/09/17 18:38:25 把几个字段处理成跟之前一样的形式, 便于统一调用evaluation_tools.py中的方法
			data[0] = data[0].to_dict()
			data[0]['id'] = [data[0]['id']]
			data[0]['label_choice'] = torch.LongTensor([data[0]['label_choice']]).to(DEVICE)
			
			model_input_data = {
				'question_parse_tree_outputs': _easy_build_module_question(_field='question'),
				'option_a_parse_tree_outputs': _easy_build_module_question(_field='option_a'),
				'option_b_parse_tree_outputs': _easy_build_module_question(_field='option_b'),
				'option_c_parse_tree_outputs': _easy_build_module_question(_field='option_c'),
				'option_d_parse_tree_outputs': _easy_build_module_question(_field='option_d'),
			}

			optimizer.zero_grad()
			output = model(model_input_data)
			evaluation_summary = evaluate_qa_model_choice(input=data[0], output=output, mode='train')
			strict_score = evaluation_summary['strict_score']
			loose_score = evaluation_summary['loose_score']
			accuracy = evaluation_summary['accuracy']
			loss = loss_function(output, data[0]['label_choice'])
			loss.backward()
			optimizer.step()
			
			logging.info(f'train | epoch: {epoch} - iteration - {iteration} - loss: {loss.item()} - accuracy: {accuracy} - score: {strict_score, loose_score}')
		
			train_logging['epoch'].append(epoch)
			train_logging['iteration'].append(iteration)
			train_logging['loss'].append(loss.item())
			train_logging['accuracy'].append(accuracy)
			train_logging['strict_score'].append(strict_score)
			train_logging['loose_score'].append(loose_score)	

		if exp_lr_scheduler is not None:
			exp_lr_scheduler.step()
		
		# 验证集评估
		if args_1.do_valid:
			model.eval()
			accuracys = []
			strict_scores = []
			loose_scores = []
			total_size = 0
			with torch.no_grad():
				for data in valid_dataloader:
					# 2022/09/17 18:38:25 把几个字段处理成跟之前一样的形式, 便于统一调用evaluation_tools.py中的方法
					data[0]['id'] = [data[0]['id']]
					data[0]['label_choice'] = torch.LongTensor([data[0]['label_choice']])
					
					_batch_size = len(data)								# 默认的批处理量为1
					assert _batch_size == 1, f'Expect batch size be 1 but got {_batch_size}'
					total_size += _batch_size
					model_input_data = {
						'question_parse_tree_outputs': _easy_build_module_question(_field='question'),
						'option_a_parse_tree_outputs': _easy_build_module_question(_field='option_a'),
						'option_b_parse_tree_outputs': _easy_build_module_question(_field='option_b'),
						'option_c_parse_tree_outputs': _easy_build_module_question(_field='option_c'),
						'option_d_parse_tree_outputs': _easy_build_module_question(_field='option_d'),
					}
					output = model(model_input_data)
					evaluation_summary = evaluate_qa_model_choice(input=data[0], output=output, mode='valid')
					strict_score = evaluation_summary['strict_score']
					loose_score = evaluation_summary['loose_score']
					accuracy = evaluation_summary['accuracy']
					accuracys.append(accuracy * _batch_size)
					strict_scores.append(strict_score * _batch_size)
					loose_scores.append(loose_score * _batch_size)
			mean_accuracy = numpy.sum(accuracys) / total_size
			mean_strict_score = numpy.sum(strict_scores) / total_size
			mean_loose_score = numpy.sum(loose_scores) / total_size
			valid_logging['epoch'].append(epoch)
			valid_logging['accuracy'].append(mean_accuracy)
			valid_logging['strict_score'].append(mean_strict_score)
			valid_logging['loose_score'].append(mean_loose_score)
			logging.info(f'valid | epoch: {epoch} - accuracy: {mean_accuracy} - score: {mean_strict_score, mean_loose_score}')	

		# 保存模型
		if do_save_checkpoint:
			# 整理所有模型
			models = {}
			for tag_name, encoder in question_tree_node_encoders.items():
				models[f'question_{tag_name}'] = encoder
			for tag_name, encoder in option_tree_node_encoders.items():
				models[f'option_{tag_name}'] = encoder
			if args_1.use_reference:
				for tag_name, encoder in reference_tree_node_encoders.items():
					models[f'reference_{tag_name}'] = encoder
			models['main'] = model
			save_multi_model_checkpoint(models=models, 
										save_path=os.path.join(CHECKPOINT_DIR, f'{model_name}_{mode}_{epoch}.h5'), 
										optimizer=optimizer, 
										scheduler=exp_lr_scheduler,
										train_logging=train_logging,
										valid_logging=valid_logging if args_2.do_valid else None)
	
		# 2021/12/20 22:20:40 每个epoch结束都记录一下结果
		train_logging_dataframe = pandas.DataFrame(train_logging, columns=list(train_logging.keys()))
		train_logging_dataframe.to_csv(os.path.join(TEMP_DIR, f'{model_name}_{mode}.csv'), header=True, index=False, sep='\t')
		if args_1.do_valid:
			valid_logging_dataframe = pandas.DataFrame(valid_logging, columns=list(valid_logging.keys()))
			valid_logging_dataframe.to_csv(os.path.join(TEMP_DIR, f'{model_name}_{mode.replace("train", "valid")}.csv'), header=True, index=False, sep='\t')
	
	terminate_logger(logger=logger)
	
	train_logging_dataframe = pandas.DataFrame(train_logging, columns=list(train_logging.keys()))
	train_logging_dataframe.to_csv(os.path.join(TEMP_DIR, f'{model_name}_{mode}.csv'), header=True, index=False, sep='\t')
	if args_1.do_valid:
		valid_logging_dataframe = pandas.DataFrame(valid_logging, columns=list(valid_logging.keys()))
		valid_logging_dataframe.to_csv(os.path.join(TEMP_DIR, f'{model_name}_{mode.replace("train", "valid")}.csv'), header=True, index=False, sep='\t')
		return train_logging_dataframe, valid_logging_dataframe
	return train_logging_dataframe
	
# 选择题模型训练脚本
def run_choice():
	# # BaseChoiceModel
	# kwargs = {
		# 'num_epoch'			: 32, 
		# 'train_batch_size'	: 128, 
		# 'valid_batch_size'	: 128, 
		# 'use_reference'		: False,
		# 'word_embedding'	: None,
		# 'document_embedding': None,
	# }
	# train_choice_model(mode='train_kd', 
					   # model_name='BaseChoiceModel', 
					   # current_epoch=0,
					   # checkpoint_path=None,
					   # do_save_checkpoint=True,
					   # is_debug=False,
					   # **kwargs)

	# train_choice_model(mode='train_ca',					   
					   # model_name='BaseChoiceModel', 
					   # current_epoch=0,
					   # checkpoint_path=None,
					   # do_save_checkpoint=True,
					   # is_debug=False,
					   # **kwargs)
	
	# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
	
	# 2022/05/20 20:14:46 重启项目, 开始引入use_pos_tags与parse_trees参数
	# # BaseChoiceModelPOS
	# kwargs = {
		# 'num_epoch'			: 32, 
		# 'train_batch_size'	: 32, 
		# 'valid_batch_size'	: 32, 
		# 'use_reference'		: False,
		# 'word_embedding'	: None,
		# 'document_embedding': None,
		# 'use_pos_tags'		: True,
		# 'use_parse_tree'	: False,
	# }
	# train_choice_model(mode='train_kd', 
					   # model_name='BaseChoiceModelPOS', 
					   # current_epoch=0,
					   # checkpoint_path=None,
					   # do_save_checkpoint=True,
					   # is_debug=False,
					   # **kwargs)

	# train_choice_model(mode='train_ca',					   
					   # model_name='BaseChoiceModelPOS', 
					   # current_epoch=0,
					   # checkpoint_path=None,
					   # do_save_checkpoint=True,
					   # is_debug=False,
					   # **kwargs)	
	
	# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
	
	# # ReferenceChoiceModel
	# kwargs = {
		# 'num_epoch'				: 32, 
		# 'train_batch_size'		: 32, 
		# 'valid_batch_size'		: 32, 
		# 'use_reference'			: True,
		# 'word_embedding'		: None,
		# 'document_embedding'	: None,
		# 'num_best'				: 32,
		# 'num_top_subject'		: 2,
		# 'num_best_per_subject'	: 8,
	# }
	# train_choice_model(mode='train_kd', 
					   # model_name='ReferenceChoiceModel', 
					   # current_epoch=0,
					   # checkpoint_path=None, 
					   # do_save_checkpoint=True, 
					   # is_debug=False,
					   # **kwargs)

	# train_choice_model(mode='train_ca', 
					   # model_name='ReferenceChoiceModel',
					   # current_epoch=0,
					   # checkpoint_path=None, 
					   # do_save_checkpoint=True,
					   # is_debug=False, 
					   # **kwargs)

	# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
	
	# # ReferenceChoiceModelPOS
	# kwargs = {
		# 'num_epoch'				: 32, 
		# 'train_batch_size'		: 4, 
		# 'valid_batch_size'		: 4, 
		# 'use_reference'			: True,
		# 'word_embedding'		: None,
		# 'document_embedding'	: None,
		# 'num_best'				: 32,
		# 'num_top_subject'		: 3,
		# 'num_best_per_subject'	: 6,
		# 'use_pos_tags'			: True,
		# 'use_parse_tree'		: False,
	# }
	# train_choice_model(mode='train_kd', 
					   # model_name='ReferenceChoiceModelPOS', 
					   # current_epoch=0,
					   # checkpoint_path=None, 
					   # do_save_checkpoint=True,
					   # is_debug=False, 
					   # **kwargs)

	# train_choice_model(mode='train_ca', 
					   # model_name='ReferenceChoiceModelPOS',
					   # current_epoch=0,
					   # checkpoint_path=None, 
					   # do_save_checkpoint=True, 
					   # is_debug=False,
					   # **kwargs)

	# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
	
	# # ReferenceDoc2VecChoiceModel
	# kwargs = {
		# 'num_epoch'				: 32, 
		# 'train_batch_size'		: 32, 
		# 'valid_batch_size'		: 32, 
		# 'use_reference'			: True,
		# 'word_embedding'		: None,
		# 'document_embedding'	: 'doc2vec',
		# 'num_best'				: 32,
		# 'num_top_subject'		: 2,
		# 'num_best_per_subject'	: 8,
	# }
	# train_choice_model(mode='train_kd', 
					   # model_name='ReferenceDoc2VecChoiceModel', 
					   # current_epoch=0,
					   # checkpoint_path=None, 
					   # do_save_checkpoint=True, 
					   # is_debug=False,
					   # **kwargs)

	# train_choice_model(mode='train_ca', 
					   # model_name='ReferenceDoc2VecChoiceModel',
					   # current_epoch=0,
					   # checkpoint_path=None,  
					   # do_save_checkpoint=True, 
					   # is_debug=False,
					   # **kwargs)
	
	# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
	
	# # Doc2VecChoiceModel
	# kwargs = {
		# 'num_epoch'			: 32, 
		# 'train_batch_size'	: 32, 
		# 'valid_batch_size'	: 32, 
		# 'use_reference'		: False,
		# 'word_embedding'	: None,
		# 'document_embedding': 'doc2vec',
	# }
	# train_choice_model(mode='train_kd', 
					   # model_name='Doc2VecChoiceModel', 
					   # current_epoch=0,
					   # checkpoint_path=None,
					   # do_save_checkpoint=True, 
					   # is_debug=False,
					   # **kwargs)

	# train_choice_model(mode='train_ca',					   
					   # model_name='Doc2VecChoiceModel', 
					   # current_epoch=0,
					   # checkpoint_path=None,
					   # do_save_checkpoint=True, 
					   # is_debug=False,
					   # **kwargs)
					   
	# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
	
	# # Word2VecChoiceModel
	# kwargs = {
		# 'num_epoch'			: 32, 
		# 'train_batch_size'	: 32, 
		# 'valid_batch_size'	: 32, 
		# 'use_reference'		: False,
		# 'word_embedding'	: 'fasttext',
		# 'document_embedding': None,
	# }
	# train_choice_model(mode='train_kd', 
					   # model_name='Word2VecChoiceModel', 
					   # current_epoch=0,
					   # checkpoint_path=None,
					   # do_save_checkpoint=True, 
					   # is_debug=False,
					   # **kwargs)

	# train_choice_model(mode='train_ca',
					   # model_name='Word2VecChoiceModel',
					   # current_epoch=0,
					   # checkpoint_path=None,
					   # do_save_checkpoint=True, 
					   # is_debug=False,
					   # **kwargs)
	
	# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
	
	# ReferenceWord2VecChoiceModel: 无法运行(OOM)
	kwargs = {
		'num_epoch'				: 32, 
		'train_batch_size'		: 2, 
		'valid_batch_size'		: 2, 
		'use_reference'			: True,
		'word_embedding'		: 'word2vec',
		'document_embedding'	: None,
		'num_best'				: 32,
		'num_top_subject'		: 2,
		'num_best_per_subject'	: 8,
	}
	train_choice_model(mode='train_kd', 
					   model_name='ReferenceWord2VecChoiceModel', 
					   current_epoch=0,
					   checkpoint_path=None, 
					   do_save_checkpoint=True, 
					   is_debug=False,
					   **kwargs)

	train_choice_model(mode='train_ca', 
					   model_name='ReferenceWord2VecChoiceModel',
					   current_epoch=0,
					   checkpoint_path=None,  
					   do_save_checkpoint=True, 
					   is_debug=False,
					   **kwargs)
					   
	# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
	
	# # BertChoiceModelX: 本地跑很快, 本地跑这些不带reference的都很快
	# kwargs = {
		# 'num_epoch'			: 32, 
		# 'train_batch_size'	: 32, 
		# 'valid_batch_size'	: 32, 
		# 'use_reference'		: False,
		# 'word_embedding'	: None,
		# 'document_embedding': 'bert-base-chinese',
	# }
	# train_choice_model(mode='train_kd', 
					   # model_name='BertChoiceModelA', 
					   # current_epoch=0,
					   # checkpoint_path=None,
					   # do_save_checkpoint=False, 
					   # is_debug=False,
					   # **kwargs)

	# train_choice_model(mode='train_ca',					   
					   # model_name='BertChoiceModelA',
					   # current_epoch=0,
					   # checkpoint_path=None,
					   # do_save_checkpoint=False, 
					   # is_debug=False,
					   # **kwargs)
	
	# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
	
	# # ReferenceBertChoiceModelX
	# kwargs = {
		# 'num_epoch'				: 32, 
		# 'train_batch_size'		: 32, 
		# 'valid_batch_size'		: 32, 
		# 'use_reference'			: True,
		# 'word_embedding'		: None,
		# 'document_embedding'	: 'bert-base-chinese',
		# 'num_best'				: 32,
		# 'num_top_subject'		: 2,
		# 'num_best_per_subject'	: 8,
	# }
	# train_choice_model(mode='train_kd', 
					   # model_name='ReferenceBertChoiceModel', 
					   # current_epoch=0,
					   # checkpoint_path=None, 
					   # do_save_checkpoint=True, 
					   # is_debug=False,
					   # **kwargs)

	# train_choice_model(mode='train_ca', 
					   # model_name='ReferenceBertChoiceModel',
					   # current_epoch=0,
					   # checkpoint_path=None,  
					   # do_save_checkpoint=True, 
					   # is_debug=False,
					   # **kwargs)
	pass

# 判断题模型训练脚本
def run_judgment():
	# # BaseJudgmentModel
	# kwargs = {
		# 'num_epoch'			: 32, 
		# 'train_batch_size'	: 128, 
		# 'valid_batch_size'	: 128, 
		# 'use_reference'		: False,
		# 'word_embedding'		: None,
		# 'document_embedding'	: None,
		# 'num_best'			: 32,
	# }
	
	# train_judgment_model(mode='train_kd', 
						 # model_name='BaseJudgmentModel',
						 # current_epoch=0,
						 # checkpoint_path=None, 
						 # do_save_checkpoint=True, 
						 # is_debug=False,
						 # **kwargs)
	# train_judgment_model(mode='train_ca', 
						 # model_name='BaseJudgmentModel',
						 # current_epoch=0,
						 # checkpoint_path=None,  
						 # do_save_checkpoint=True, 
						 # is_debug=False,
						 # **kwargs)
						 
	# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
	
	# ReferenceJudgmentModel
	kwargs = {
		'num_epoch'			: 32, 
		'train_batch_size'	: 32, 
		'valid_batch_size'	: 128, 
		'use_reference'		: True,
		'word_embedding'	: None,
		'document_embedding': None,
		'num_best'			: 32,
	}
	train_judgment_model(mode='train_kd', 
						 model_name='ReferenceJudgmentModel', 
						 current_epoch=3,
						 checkpoint_path=os.path.join(CHECKPOINT_DIR, 'ReferenceJudgmentModel_train_kd_2.h5'),
						 do_save_checkpoint=True, 
						 is_debug=False,
						 **kwargs)
	
	train_judgment_model(mode='train_ca', 
						 model_name='ReferenceJudgmentModel', 
						 current_epoch=7, 
						 checkpoint_path=os.path.join(CHECKPOINT_DIR, 'ReferenceJudgmentModel_train_ca_6.h5'),
						 do_save_checkpoint=True, 
						 is_debug=False,
						 **kwargs)
	pass

# 2022/09/06 22:48:16 句法树形式的模型训练脚本
def run_tree_choice():
	# TreeReferenceChoiceModel
	kwargs = {
		'num_epoch'										: 32, 
		'train_batch_size'								: 1, 			# 必须是1
		'valid_batch_size'								: 1, 			# 必须是1
		'use_reference'									: False,
		'word_embedding'								: 'word2vec',
		'document_embedding'							: None,
		'num_best'										: 32,
		'tree_rnn_encoder_node_hidden_size'				: 128,
		'tree_rnn_encoder_root_output_size'				: 256,
		'tree_rnn_encoder_rnn_type'						: 'GRU',
		'tree_rnn_encoder_num_layers'					: 2,
		'tree_rnn_encoder_bidirectional'				: False,
		'tree_rnn_encoder_squeeze_strategy'				: 'final',
		'tree_model_aggregation_module_output_size'		: 256,
		'tree_model_aggregation_module_num_layers'		: 2,
		'tree_model_aggregation_module_bidirectional'	: False,
		'default_embedding_size'						: 128,
		'default_max_child'								: 256,
		'learning_rate'									: 0.1,
	}
	
	train_parse_tree_choice_model(mode='train_kd', 
								  model_name='TreeChoiceModel',
								  current_epoch=0,
						 		  checkpoint_path=None, 
						 		  do_save_checkpoint=True, 
						 		  is_debug=False,
						 		  **kwargs)
						 		  
	train_parse_tree_choice_model(mode='train_ca', 
								  model_name='TreeChoiceModel',
								  current_epoch=0,
						 		  checkpoint_path=None, 
						 		  do_save_checkpoint=True, 
						 		  is_debug=False,
						 		  **kwargs)

if __name__ == '__main__':
	# run_choice()
	# run_judgment()
	run_tree_choice()
