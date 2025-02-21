# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# 模型评估相关工具

if __name__ == '__main__':
	import sys
	sys.path.append('../')

import json
import torch
import numpy
import pandas
import logging

from gensim.corpora import Dictionary
from collections import Counter
from sklearn.metrics import auc, confusion_matrix, accuracy_score, roc_auc_score

from config import RetrievalModelConfig, EmbeddingModelConfig
from setting import *

from src.data_tools import decode_answer
from src.retrieval_model import GensimRetrievalModel
from src.embedding_model import GensimEmbeddingModel
from src.utils import load_args, timer


# 评估所有的gensim模型预测subject的效果
@timer
def evaluate_gensim_model_in_filling_subject(gensim_retrieval_model_names=None, 
											 gensim_embedding_model_names=None,
											 hits=[1, 3, 5, 10]):
	if gensim_retrieval_model_names is None:
		gensim_retrieval_model_names = list(GENSIM_RETRIEVAL_MODEL_SUMMARY.keys())
	if gensim_embedding_model_names is None:
		gensim_embedding_model_names = list(GENSIM_EMBEDDING_MODEL_SUMMARY.keys())
	evaluation_summary = {model_name: {f'hit@{hit}': 0 for hit in hits} for model_name in gensim_retrieval_model_names + gensim_embedding_model_names}
	dictionary = Dictionary.load(REFERENCE_DICTIONARY_PATH)
	grm = GensimRetrievalModel(args=load_args(Config=RetrievalModelConfig))
	gem = GensimEmbeddingModel(args=load_args(Config=EmbeddingModelConfig))
		
	# 加载训练集中有subject标签的部分
	trainset_dataframe = pandas.concat([pandas.read_csv(filepath, sep='\t', header=0) for filepath in TRAINSET_PATHs])
	trainset_dataframe_with_subject = trainset_dataframe[~trainset_dataframe['subject'].isna()].reset_index(drop=True)

	# 构建相似度
	grm_similaritys = {model_name: grm.build_similarity(model_name=model_name, dictionary=None, corpus=None, num_best=None) for model_name in gensim_retrieval_model_names}
	gem_similaritys = {model_name: gem.build_similarity(model_name=model_name, dictionary=None, corpus=None, num_best=None) for model_name in gensim_embedding_model_names}
	grm_sequences = {model_name: GensimRetrievalModel.load_sequence(model_name=model_name, subject=None) for model_name in gensim_retrieval_model_names}
	
	# 加载参考书目文档
	reference_dataframe = pandas.read_csv(REFERENCE_PATH, sep='\t', header=0)
	index2subject = {index: '法制史' if law == '目录和中国法律史' else law for index, law in enumerate(reference_dataframe['law'])}			


	# 根据query结果计算
	def _update_evaluation_summary(_model_name, _true_subject, _query_result, _hits):
		_subject_rank_list = list(map(lambda x: index2subject[x[0]], _query_result))
		_weighted_count = {}																
		for _rank, _subject in enumerate(_subject_rank_list):
			if _subject in _weighted_count:
				_weighted_count[_subject] += 1 / (_rank + 1)							
			else:
				_weighted_count[_subject] = 1 / (_rank + 1)
		_counter = Counter(_weighted_count).most_common()
		_predicted_subjects = list(map(lambda x: x[0], _counter))
		for _hit in _hits:
			if _true_subject in _predicted_subjects[: _hit]:
				evaluation_summary[_model_name][f'hit@{_hit}'] += 1		

	for i in range(trainset_dataframe_with_subject.shape[0]): 
		statement = eval(trainset_dataframe_with_subject.loc[i, 'statement'])
		option_a = eval(trainset_dataframe_with_subject.loc[i, 'option_a'])
		option_b = eval(trainset_dataframe_with_subject.loc[i, 'option_b'])
		option_c = eval(trainset_dataframe_with_subject.loc[i, 'option_c'])
		option_d = eval(trainset_dataframe_with_subject.loc[i, 'option_d'])
		_type = trainset_dataframe_with_subject.loc[i, 'type']
		subject = trainset_dataframe_with_subject.loc[i, 'subject']
		
		query_tokens = statement + option_a + option_b + option_c + option_d
		
		for model_name, similarity in grm_similaritys.items():
			sequence = grm_sequences[model_name]
			grm_query_result = grm.query(query_tokens, dictionary, similarity, sequence)
			_update_evaluation_summary(_model_name=model_name, _true_subject=subject, _query_result=grm_query_result, _hits=hits)
			
		for model_name, similarity in gem_similaritys.items():
			gem_query_result = gem.query(query_tokens, dictionary, similarity)
			_update_evaluation_summary(_model_name=model_name, _true_subject=subject, _query_result=gem_query_result, _hits=hits)
					
	with open(os.path.join(TEMP_DIR, 'evaluate_gensim_model_in_filling_subject.json'), 'w') as f:
		json.dump(evaluation_summary, f, indent=4)
		
	return evaluation_summary
	

# 评估预测选择题的问答模型输出结果
# train或valid情况下返回当前批次题目的得分与精确度, 以及每道题的具体得分与精确度
# test模式下返回每道题的答案
def evaluate_qa_model_choice(input, output, mode='train'):
	batch_size = output.shape[0]
	# 训练或验证都需要根据已有的标签输出得分与精确度
	if mode == 'train' or mode == 'valid':
		evaluation_summary = {'summary': []}
		total_strict_score = 0.
		total_loose_score = 0.
		total_accuracy = 0.
		for i in range(batch_size):
			_id = input['id'][i]
			_output = output[i]
			target_encoded_answer = int(input['label_choice'][i])		# 0-15的正确选择题答案编码值
			predicted_encoded_answer = int(torch.max(_output, dim=0)[1])# 0-15的预测选择题答案编码值, _output的维数是16, 取torch.max函数返回结果的第二个即最大值所在索引
			strict_score, loose_score, accuracy = calc_score_and_accuracy_for_choice_answer(target_encoded_answer=target_encoded_answer, predicted_encoded_answer=predicted_encoded_answer)
			total_strict_score += strict_score
			total_loose_score += loose_score
			total_accuracy += accuracy
			summary = {
				_id: {
					'target'		: target_encoded_answer, 
					'predict'		: predicted_encoded_answer, 
					'strict_score'	: strict_score, 
					'loose_score'	: loose_score, 
					'accuracy'		: accuracy
				},
			}
			evaluation_summary['summary'].append(summary)
		evaluation_summary['strict_score'] = total_strict_score / batch_size
		evaluation_summary['loose_score'] = total_loose_score / batch_size
		evaluation_summary['accuracy'] = total_accuracy / batch_size

	# 测试没有正确答案标签, 直接输出预测结果
	elif mode == 'test':
		evaluation_summary = {}
		for i in range(batch_size):
			_id = input['id'][i]
			_output = output[i]
			predicted_encoded_answer = int(torch.max(_output, dim=0)[1])# 0-15的编码值, _output的维数是16, 取torch.max函数返回结果的第二个即最大值所在索引
			predicted_decoded_answer = decode_answer(encoded_answer=predicted_encoded_answer, result_type=str)
			summary = {_id: predicted_decoded_answer}					# 这个是submit_sample中指定的提交格式文件形式
			evaluation_summary.update(summary)							
	else:
		raise NotImplementedError
	return evaluation_summary


# 评估预测判断题的问答模型输出结果
# train或valid情况下返回当前批次题目的精确度, 以及每道题的具体情况
# test模式下返回每道题的每个选项的预测真伪
# 20211217更新: 增加thresholds参数用于测试不同的阈值对模型精确性的影响
def evaluate_qa_model_judgment(input, output, mode='train', thresholds=[.25, .5, .75]):
	batch_size = output.shape[0]
	
	# 训练或验证都需要根据已有的标签输出
	if mode == 'train' or mode == 'valid':
		predicts = [(output > threshold) * 1 for threshold in thresholds]
		evaluation_summary = {
			'summary': {
				'id'		: [],
				'option_id'	: [],
				'target'	: [],
				'predict'	: [],
			},
		}

		for i in range(batch_size):
			_id = input['id'][i]
			option_id = input['option_id'][i]	
			target_answer = input['label_judgment'][i]					# 0或1的正确判断题答案编码值
			predicted_answers = {}
			for predict, threshold in zip(predicts, thresholds):
				predicted_answer = predict[i]							# 0或1的预测选择题答案编码值
				predicted_answers[threshold] = predicted_answer
			
			evaluation_summary['summary']['id'].append(_id)
			evaluation_summary['summary']['option_id'].append(option_id)
			evaluation_summary['summary']['target'].append(target_answer)
			evaluation_summary['summary']['predict'].append(predicted_answers)
		evaluation_summary['accuracy'] = {threshold: torch.sum(predict == input['label_judgment']).item() / batch_size for threshold, predict in zip(thresholds, predicts)}

	# 测试没有正确答案标签, 直接输出预测结果
	elif mode == 'test':
		assert len(thresholds) == 1
		threshold = thresholds[0]
		predict = (output > threshold) * 1
		evaluation_summary = {
			'id'		: [],
			'option_id'	: [],
			'predict'	: [],
		}
		for i in range(batch_size):
			_id = input['id'][i]
			option_id = input['option_id'][i]	
			predicted_answer = predict[i]								# 0或1的预测选择题答案编码值
			evaluation_summary['id'].append(_id)
			evaluation_summary['option_id'].append(option_id)
			evaluation_summary['predict'].append(predicted_answer)			
	else:
		raise NotImplementedError
	return evaluation_summary


# (多项)选择题的得分评判: 错选得0分, 少选得0.5分, 完全正确得1分, 但是在严格的条件下, 只要错误就不给分
# 比如target_encoded_answer为11(ABD), predicted_encoded_answer为13(ACD), 则严格得分为0, 松弛得分还是零, 准确率为50%
def calc_score_and_accuracy_for_choice_answer(target_encoded_answer, predicted_encoded_answer):
	if target_encoded_answer == predicted_encoded_answer:	
		# 完全正确
		strict_score = 1.	# 除非正确作答, 否则严格得分都是零
		loose_score = 1.
		accuracy = 1.	
	elif predicted_encoded_answer == 0:						
		# 交白卷: 不给分
		strict_score = 0.
		loose_score = 0.
		target_decoded_answer = decode_answer(encoded_answer=target_encoded_answer, result_type=str)
		accuracy = (TOTAL_OPTIONS - len(target_decoded_answer)) / 4
	else:													
		# 正常作答
		strict_score = 0.
		target_decoded_answer = decode_answer(encoded_answer=target_encoded_answer, result_type=str)
		predicted_decoded_answer = decode_answer(encoded_answer=predicted_encoded_answer, result_type=str)
		flag = True		# 判断是否存在错选的情况
		for option in predicted_decoded_answer:
			if option not in target_decoded_answer:
				flag = False
		loose_score = .5 if flag else 0
		accuracy = 0.
		for option in OPTION2INDEX:
			if (option in target_decoded_answer) == (option in predicted_decoded_answer):
				accuracy += 1 / TOTAL_OPTIONS
	return strict_score, loose_score, accuracy

# 计算二分类任务的混淆矩阵, 精确度, AUC值
def evaluate_classifier(target, predict, predict_proba):
	_confusion_matrix = confusion_matrix(target, predict)
	_accuracy_score = accuracy_score(target, predict)
	_roc_auc_score = roc_auc_score(target, predict_proba)
	return _confusion_matrix, _accuracy_score, _roc_auc_score
