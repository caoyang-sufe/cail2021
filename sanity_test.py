# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
# 可用性测试脚本

import re
import json
import time
import torch
import numpy
import gensim
import pandas
import networkx

from gensim.corpora import Dictionary, MmCorpus
from matplotlib import pyplot as plt
from torch.nn import BatchNorm1d

from config import DatasetConfig, RetrievalModelConfig, EmbeddingModelConfig
from setting import *
from preprocess import build_gensim_retrieval_models

from src.dataset import BasicDataset, generate_basic_dataloader, generate_parse_tree_dataloader
from src.retrieval_model import GensimRetrievalModel
from src.embedding_model import GensimEmbeddingModel
from src.evaluation_tools import evaluate_gensim_model_in_filling_subject
from src.utils import load_args, initialize_logger
from src.plot_tools import train_plot_judgment, train_plot_choice
from src.graph import Graph
from src.graph_tools import generate_pos_tags, generate_pos_tags_from_parse_tree, generate_parse_tree, parse_tree_to_graph, generate_dependency
from src.data_tools import generate_userdict

# 测试BasicDataset运行情况
def test_basic_dataset():
	args = load_args(Config=DatasetConfig)
	args.use_reference = True
	args.retrieval_model_name = 'tfidf'
	args.train_batch_size = 2
	args.valid_batch_size = 2
	args.test_batch_size = 2
	args.use_parse_tree = True
	args.use_pos_tags = True
	
	# args.word_embedding = 'word2vec'
	# args.document_embedding = None
	# for pipeline in ['judgment']:
		# for mode in ['train', 'valid', 'test']:
			# print(pipeline, mode, args.word_embedding, args.document_embedding)
			# dataloader = generate_basic_dataloader(args=args, mode=mode, do_export=False, pipeline=pipeline, for_debug=True)
			# for i, data in enumerate(dataloader):
				# print(i)
	
	# 测试2
	for word_embedding in [None]:
		args.word_embedding = word_embedding
		for document_embedding in [None]:
			args.document_embedding = document_embedding
			if args.word_embedding is not None and args.document_embedding is not None:
				continue
			print(args.word_embedding, args.word_embedding)
			for pipeline in ['choice']:
				for mode in ['train', 'valid', 'test']:
					print(pipeline, mode, args.word_embedding, args.document_embedding)
					# dataset = BasicDataset(args=args, mode=mode, do_export=False, pipeline=pipeline, for_debug=True)
					# print(dataset.data)
					# dataset.data.to_csv(f'{mode}_dataset_{time.strftime("%Y%m%d%H%M%S")}.csv', header=True, index=False, sep='\t')
					dataloader = generate_basic_dataloader(args=args, mode=mode, do_export=False, pipeline=pipeline, for_debug=True)
					for i, data in enumerate(dataloader):
						print(f'Batch {i}')
						print('option_a', data['option_a_tree'], data['option_a_pos_tags'], data['option_a_pos_tags'].shape)
						print('option_b', data['option_b_tree'], data['option_b_pos_tags'], data['option_b_pos_tags'].shape)
						print('option_c', data['option_c_tree'], data['option_c_pos_tags'], data['option_c_pos_tags'].shape)
						print('option_d', data['option_d_tree'], data['option_d_pos_tags'], data['option_d_pos_tags'].shape)
						print('reference', data['reference_tree'], data['reference_pos_tags'], data['reference_pos_tags'].shape) # [2, 18, 512]
						input()
	
	# 测试3 BERT
	args.word_embedding = None
	args.document_embedding = 'bert-base-chinese'
	dataloader = generate_basic_dataloader(args=args, mode='train', do_export=False, pipeline='choice', for_debug=True)
	
	for i, data in enumerate(dataloader):
		print(i)
		print(data['options'].shape)
		print(data['question'].shape)
		print(data['reference'].shape)
		print('#' * 64)
	

def test_parse_tree_dataset():

	args = load_args(Config=DatasetConfig)
	args.use_reference = True
	args.retrieval_model_name = 'tfidf'
	args.word_embedding = None
	args.document_embedding = None
	args.train_batch_size = 2
	args.valid_batch_size = 2
	args.test_batch_size = 2
	
	mode = 'train'
	pipeline = 'choice'	
	
	dataloader = generate_parse_tree_dataloader(args=args, mode=mode, do_export=False, pipeline=pipeline, for_debug=True)

	for i, data in enumerate(dataloader):
		print(i)
		print(data)

		# print(data['question'])
		# print('-' * 64)
		# print(data['option_a_vector'])
		# print('-' * 64)
		# print(data['option_b_vector'])
		# print('-' * 64)
		# print(data['option_c_vector'])
		# print('-' * 64)
		# print(data['option_d_vector'])

		print('#' * 64)
	
				
# tfidf调参
def test_tfidf():
	args = load_args(Config=RetrievalModelConfig)	
	summary = []
	count = 0
	for pivot in [None, 1.]:
		args.pivot_tfidf = pivot
		for slope in [.25, .5]:
			args.slope_tfidf = slope
			for a in ['b', 'n', 'a', 'l', 'd']:
				for b in ['n', 'f', 't', 'p']:
					for c in ['n', 'c', 'u', 'b']:
						try:
							count += 1
							args.smartirs_tfidf = a + b + c
							print(count, args.pivot_tfidf, args.slope_tfidf, args.smartirs_tfidf)
							build_gensim_retrieval_models(args=args, model_names=['tfidf'], update_reference_corpus=False)
							_summary = evaluate_gensim_model_in_filling_subject(gensim_retrieval_model_names=['tfidf'], 
																			    gensim_embedding_model_names=[],
																			    hits=[1, 3, 5, 10])													   
							temp_summary = {'args': {'smartirs': args.smartirs_tfidf, 'pivot': args.pivot_tfidf, 'slope': args.slope_tfidf}, 'result': _summary}
							summary.append(temp_summary)
						except Exception as e:
							with open('error.txt', 'a') as f:
								f.write(f'{args.pivot_tfidf} - {args.slope_tfidf} - {args.smartirs_tfidf}')
								f.write(str(e))
								f.write('\n')

	with open(os.path.join(TEMP_DIR, 'test_smartirs.json'), 'w', encoding='utf8') as f:
		json.dump(summary, f, indent=4)


# stanford包测试
def stanford_demo():
	
	# args = load_args(Config=GraphConfig)
	# graph = Graph(args=args)
	# graph.long_tokens_to_dependencys(tokens=['—', '—', '坚持', '中国共产党', '的', '领导', '。', '习近平', '总书记', '在', '党', '的', '十九', '大', '报告', '中', '强调', ':', '“', '党政军民', '学', '，', '东西南北中', '，', '党', '是', '领导', '一切', '的', '。', '必须', '增强', '政治', '意识', '.', '大局意识', '.', '核心', '意识', '.', '看齐', '意识', '，', '自觉', '维护', '党中央', '权威', '和', '集中统一', '领导', '，', '自觉', '在思想上', '政治', '上', '行动', '上同', '党中央', '保持', '高度一致', '，', '完善', '坚持', '党的领导', '的', '体制', '机制', '，', '坚持', '稳中求进', '工作', '总', '基调', '，', '统筹', '推进', '‘', '五位一体', '’', '总体布局', '，', '协调', '推进', '‘', '四个', '全面', '’', '战略', '布局', '，', '提高', '党', '把', '方向', '.', '谋', '大局', '.', '定', '政策', '.', '促', '改革', '的', '能力', '和', '定力', '，', '确保', '党', '始终', '总揽全局', '.', '协调', '各方', '。', '”', '中国共产党', '领导', '是', '中国', '特色', '社会主义', '最', '本质', '的', '特征', '，', '是', '社会主义', '法治', '最', '根本', '的', '保证', '。', '把', '党的领导', '贯彻', '到', '依法治国', '全过程', '和', '各', '方面', '，', '是', '我国', '社会主义', '法治', '建设', '的', '一条', '基本', '经验', '。', '我国', '宪法', '确立', '了', '中国共产党', '的', '领导', '地位', '。', '坚持', '党的领导', '，', '是', '社会主义', '法治', '的', '根本', '要求', '，', '是', '党和国家', '的', '根本', '所在', '.', '命脉', '所在', '，', '是', '全国', '各族人民', '的', '利益', '所系', '.', '幸福', '所系', '，', '是', '全面', '依法治国', '的', '题', '中', '应有', '之义', '。', '党的领导', '和', '社会主义', '法治', '是', '一致', '的', '，', '社会主义', '法治', '必须', '坚持', '党的领导', '，', '党的领导', '必须', '依靠', '社会主义', '法治', '。', '只有', '在', '党的领导', '下', '依法治国', '.', '厉行', '法治', '，', '人民', '当家作主', '才能', '充分', '实现', '，', '国家', '和', '社会', '生活', '法治化', '才能', '有序', '推进', '。', '依法', '执政', '，', '既', '要求', '党', '依据', '宪法', '法律', '治国', '理政', '，', '也', '要求', '党', '依据', '党内', '法规', '管党', '治党', '。', '必须', '坚持', '党', '领导', '立法', '.', '保证', '执法', '.', '支持', '司法', '.', '带头', '守法', '，', '把', '依法治国', '基本', '方略', '同', '依法', '执政', '基本', '方式', '统一', '起来', '，', '把', '党', '总揽全局', '.', '协调', '各方', '同', '人大', '.', '政府', '.', '政协', '.', '审判机关', '.', '检察机关', '依法', '依', '章程', '履行', '职能', '.', '开展', '工作', '统一', '起来', '，', '把', '党', '领导', '人民', '制定', '和', '实施', '宪法', '法律', '同党', '坚持', '在', '宪法', '法律', '范围', '内', '活动', '统一', '起来', '，', '善于', '使党', '的', '主张', '通过', '法定程序', '成为', '国家', '意志', '，', '善于', '使', '党组织', '推荐', '的', '人选', '通过', '法定程序', '成为', '国家', '政权机关', '的', '领导人员', '，', '善于', '通过', '国家', '政权机关', '实施', '党', '对', '国家', '和', '社会', '的', '领导', '，', '善于', '运用', '民主集中制', '原则', '维护', '中央', '权威', '.', '维护', '全党全国', '团结', '统一', '。'])

	# lst = generate_dependency(tokens=['党', '是', '领导', '一切', '的', '。'])
	# lst = generate_dependency(tokens=['我', '是', '一个', '学生', '。'])
	# lst = generate_dependency(tokens=['世界贸易组织', '的', '法律', '制度', '是', '一个', '以', '世界贸易组织协定', '为', '核心', '统一', '的', '多边贸易', '法律', '制度', '，', '由', '一系列', '规则', '组成', '。', '该', '制度', '是', '在', '继承', '关税与贸易总协定', '框架', '下', '的', '规则', '的', '基础', '上', '发展', '起来', '的', '。', '各', '协议', '.', '规则', '各自', '规定', '了', '不同', '的', '独立', '义务', '.', '共同', '约束', '世界贸易组织', '的', '成员', '。'])
	# lst = generate_dependency(tokens=['第', '1', '节', '\xa0', '\xa0', '世界贸易组织', '概述'])
	print(lst)
	print(lst[0].root)

	for i in lst[0].triples():
		print(i)


if __name__ == '__main__':
	
	# test_parse_tree_dataset()
	
	# 这个案例还挺有意思, 说明不能直接修改pandas.Series中某个元素的值
	df = pandas.DataFrame({'a': [1,2,3,4], 'b':[2,3,4,5]})
	x = df.loc[0, :]

	# x['a'] = torch.FloatTensor([x['a']])
	# x['a'] = [1]
	x['a'] = [2]
	print(x['a'], type(x['a']))
	
	# t1 = torch.FloatTensor([[1,2,3,4], [200,400,600,800]])
	# print(BatchNorm1d(4)(t1))
