# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
# 模型测试

import os
import time
import json
import torch
import numpy
import pandas
import logging

from setting import *
from config import QAModelConfig, DatasetConfig

from src.dataset import generate_basic_dataloader
from src.qa_model import *
from src.torch_tools import load_checkpoint
from src.evaluation_tools import evaluate_qa_model_choice, evaluate_qa_model_judgment
from src.utils import load_args, save_args, timer


# 选择题模型测试
@timer
def test_choice_model(mode, model_name, model_path, **kwargs):
	assert mode in ['test', 'test_ca', 'test_kd', 'valid', 'valid_ca', 'valid_kd']
	
	# 配置模型并从存档点中提取训练后的权重
	args_1 = load_args(Config=QAModelConfig)							# 问答模型配置
	# 根据kwargs调整问答模型配置的参数
	for key, value in kwargs.items():
		args_1.__setattr__(key, value)
	model = eval(model_name)(args=args_1)
	checkpoint = load_checkpoint(model=model, save_path=model_path, optimizer=None, scheduler=None)
	saved_model = checkpoint['model'].to(DEVICE)
	
	# 配置训练集
	args_2 = load_args(Config=DatasetConfig)
	# 根据kwargs调整问答模型配置的参数
	for key, value in kwargs.items():
		args_2.__setattr__(key, value)																																
	test_dataloader = generate_basic_dataloader(args=args_2, mode=mode, do_export=False, pipeline='choice', for_debug=False)
	
	# 开始测试
	saved_model.eval()
	answer = {}
	with torch.no_grad():
		for data in test_dataloader:
			for key in data.keys():
				if isinstance(data[key], torch.Tensor):
					data[key] = data[key].to(DEVICE)
			output = saved_model(data)
			evaluation_summary = evaluate_qa_model_choice(input=data, output=output, mode='test')
			answer.update(evaluation_summary)
	
	# 导出答案
	with open(os.path.join(TEMP_DIR, f'answer_{model_name}_{mode}.json'), 'w', encoding='utf8') as f:
		json.dump(answer, f, indent=4)
	
	return answer

# 判断题模型测试
@timer
def test_judgment_model(mode, model_name, model_path, threshold=.5, **kwargs):
	assert mode in ['test', 'test_ca', 'test_kd', 'valid', 'valid_ca', 'valid_kd']
	
	# 配置模型并从存档点中提取训练后的权重
	args_1 = load_args(Config=QAModelConfig)							# 问答模型配置
	# 根据kwargs调整问答模型配置的参数
	for key, value in kwargs.items():
		args_1.__setattr__(key, value)
	model = eval(model_name)(args=args_1)
	checkpoint = load_checkpoint(model=model, save_path=model_path, optimizer=None, scheduler=None)
	saved_model = checkpoint['model'].to(DEVICE)
	
	# 配置训练集
	args_2 = load_args(Config=DatasetConfig)
	# 根据kwargs调整问答模型配置的参数
	for key, value in kwargs.items():
		args_2.__setattr__(key, value)																																
	test_dataloader = generate_basic_dataloader(args=args_2, mode=mode, do_export=False, pipeline='judgment', for_debug=False)
	
	# 开始测试
	saved_model.eval()
	evaluation_summary = {
		'id'		: [],
		'option_id'	: [],
		'predict'	: [],
	}
	with torch.no_grad():
		for data in test_dataloader:
			for key in data.keys():
				if isinstance(data[key], torch.Tensor):
					data[key] = data[key].to(DEVICE)
			output = saved_model(data)
			_evaluation_summary = evaluate_qa_model_judgment(input=data, output=output, mode='test', thresholds=[threshold])
			evaluation_summary['id'].extend(_evaluation_summary['id'])
			evaluation_summary['option_id'].extend(_evaluation_summary['option_id'])
			evaluation_summary['predict'].extend(_evaluation_summary['predict'])
	evaluation_summary_dataframe = pandas.DataFrame(evaluation_summary, columns=list(evaluation_summary.keys()))
	
	
	answer = {}
	for _id, _dataframe in evaluation_summary_dataframe.groupby(['id']):
		assert _dataframe.shape[0] == 4
		_answer = []
		for i in _dataframe.index:
			if _dataframe.loc[i, 'predict'] == 1:
				_answer.append(_dataframe.loc[i, 'option_id'])
		answer[_id] = _answer

	# 导出答案
	with open(os.path.join(TEMP_DIR, f'answer_{model_name}_{mode}.json'), 'w', encoding='utf8') as f:
		json.dump(answer, f, indent=4)

	return answer

# 生成特定的答案文件
@timer
def generate_artificial_answer():
	
	test_dataframe = pandas.concat([pandas.read_csv(filepath, sep='\t', header=0) for filepath in TESTSET_PATHs]) 
	
	answer = {}

	for _id in test_dataframe['id']:
		answer[_id] = ['D']
		
	with open(os.path.join(TEMP_DIR, f'answer_d.txt'), 'w', encoding='utf8') as f:
		json.dump(answer, f, indent=4)	

if __name__ == '__main__':
	
	
	
	# answer_kd = test_choice_model(mode='test_kd', 
								  # model_name='BaseChoiceModel', 
								  # model_path=os.path.join(CHECKPOINT_DIR, 'BaseChoiceModel', 'BaseChoiceModel_train_kd_31.h5'), 
								  # **{'test_batch_size': 32, 'use_reference': False})
	# answer_ca = test_choice_model(mode='test_ca', 
								  # model_name='BaseChoiceModel', 
								  # model_path=os.path.join(CHECKPOINT_DIR, 'BaseChoiceModel', 'BaseChoiceModel_train_ca_31.h5'), 
								  # **{'test_batch_size': 32, 'use_reference': False})
	# with open(os.path.join(TEMP_DIR, f'answer_BaseChoiceModel.json'), 'w', encoding='utf8') as f:
		# json.dump({**answer_kd, **answer_ca}, f, indent=4)	
					  
	# answer_kd = test_judgment_model(mode='test_kd', 
									# model_name='BaseJudgmentModel', 
									# model_path=os.path.join(CHECKPOINT_DIR, 'BaseJudgmentModel', 'BaseJudgmentModel_train_kd_1.h5'), 
									# **{'test_batch_size': 32, 'use_reference': False})
					  
	# answer_ca = test_judgment_model(mode='test_ca', 
								    # model_name='BaseJudgmentModel', 
								    # model_path=os.path.join(CHECKPOINT_DIR, 'BaseJudgmentModel', 'BaseJudgmentModel_train_ca_3.h5'), 
								    # **{'test_batch_size': 32, 'use_reference': False})
	# with open(os.path.join(TEMP_DIR, f'answer_BaseJudgmentModel.json'), 'w', encoding='utf8') as f:
		# json.dump({**answer_kd, **answer_ca}, f, indent=4)		
	
	model_name = 'BaseChoiceModel'
	answer_kd = test_choice_model(mode='test_kd',
								  model_name=model_name,
								  model_path=os.path.join(CHECKPOINT_DIR, model_name, f'{model_name}_train_kd_29.h5'), 
								  **{'test_batch_size'		: 32, 
									 'use_reference'		: False, 
									 'word_embedding'		: None, 
									 'document_embedding'	: None, 
									 'use_pos_tags'			: False,
									 'use_parse_tree'		: False,})
	
	model_name = 'BaseChoiceModel'
	answer_ca = test_choice_model(mode='test_ca', 
								  model_name=model_name, 
								  model_path=os.path.join(CHECKPOINT_DIR, model_name, f'{model_name}_train_ca_29.h5'), 
								  **{'test_batch_size'		: 32, 
									 'use_reference'		: False, 
									 'word_embedding'		: None, 
									 'document_embedding'	: None, 
									 'use_pos_tags'			: False,
									 'use_parse_tree'		: False,})

	with open(os.path.join(TEMP_DIR, f'answer_{model_name}.json'), 'w', encoding='utf8') as f:
		json.dump({**answer_kd, **answer_ca}, f, indent=4)		

	# generate_artificial_answer()
