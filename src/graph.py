# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# 图模型

if __name__ == '__main__':
	import sys
	sys.path.append('../')


import os
import re
import time
import numpy
import torch
import jieba
import jieba.analyse	# 2021/12/31 18:47:09 知识盲区, 这个模块得单独import
import pandas
import gensim
import logging
import networkx

from copy import deepcopy
from gensim.corpora import MmCorpus, Dictionary
from torch.nn import Module, Embedding, Linear, Sigmoid, CrossEntropyLoss, functional as F

from wordcloud import WordCloud
from matplotlib import pyplot as plt

from setting import *

from src.data_tools import load_stopwords, filter_stopwords
from src.graph_tools import plot_reference_wordcloud, generate_pos_tags, generate_parse_tree, generate_dependency
from src.qa_module import BaseLSTMEncoder, BaseAttention
from src.utils import load_args, timer, is_chinese, is_number, is_alphabet, is_symbol


class Graph:
	"""图模型相关"""
	def __init__(self, args):
		"""
		:param args	: GraphConfig配置
		"""
		self.args = deepcopy(args)

		self.split_symbols = args.split_symbols			# 用于分隔语句的字符
		self.regex_compiler = re.compile(r'\s+', re.I)	# 用于匹配连续空格的正则

	def build_reference_parse_tree(self, export_path=REFERENCE_PARSE_TREE_PATH):
		"""2022/03/14 20:08:06
		生成参考书目文档的句法树:
		全部生成耗时很长, 差不多需要整整两天才能跑完所有参考文档
		"""
		reference_dataframe = pandas.read_csv(REFERENCE_PATH, sep='\t', header=0, dtype=str)	# 预处理后的参数书目文档
		reference_dataframe = reference_dataframe.fillna('')									# 参考书目的section字段存在缺失, 可使用空字符串填充

		with open(export_path, 'w', encoding='utf8') as f:
			# 新建文件并写入表头
			f.write('id\tcontent\n')

		for i in range(reference_dataframe.shape[0]):
			print(i)
			content_tokens = eval(reference_dataframe.loc[i, 'content'])
			# 可能会因为分句过长而无法解析的情况, 报错原因通常是OOM, 采取异常抛出
			try:
				parse_trees = self.long_tokens_to_parse_trees(tokens=content_tokens)
			except Exception as exception:
				parse_trees = exception
			with open(export_path, 'a', encoding='utf8') as f:
				f.write(f'{i}\t{parse_trees}\n')

	def build_question_parse_tree(self, export_path=QUESTION_PARSE_TREE_PATH):
		"""2022/03/14 20:08:06
		生成题库的句法树:
		这里直接读取预处理得到的0_train.csv, 1_train.csv, 0_valid.csv, 1_valid.csv, 0_test.csv, 1_test.csv
		全部跑完的时间要比
		"""
		# 合并所有题库(训练集, 验证集, 测试集), 并标明每道题来源于哪个数据集
		dataframes = []
		filepaths = TRAINSET_PATHs + VALIDSET_PATHs + TESTSET_PATHs
		selected_columns = ['id', 'statement', 'option_a', 'option_b', 'option_c', 'option_d']
		for filepath in filepaths:
			dataframe = pandas.read_csv(filepath, sep='\t', header=0)[selected_columns]
			dataframe['source'] = os.path.split(filepath)[-1]
			dataframes.append(dataframe)
		question_dataframe = pandas.concat(dataframes).reset_index(drop=True)
		selected_columns.append('source')

		# 切换数据类型
		question_dataframe['id'] = question_dataframe['id'].astype(str)				# 字段id转为字符串
		question_dataframe['statement'] = question_dataframe['statement'].map(eval)	# 字段statement用eval函数转为分词列表
		question_dataframe['option_a'] = question_dataframe['option_a'].map(eval)	# 字段option_a用eval函数转为分词列表
		question_dataframe['option_b'] = question_dataframe['option_b'].map(eval)	# 字段option_b用eval函数转为分词列表
		question_dataframe['option_c'] = question_dataframe['option_c'].map(eval)	# 字段option_c用eval函数转为分词列表
		question_dataframe['option_d'] = question_dataframe['option_d'].map(eval)	# 字段option_d用eval函数转为分词列表

		# 新建文件并写入表头
		with open(export_path, 'w', encoding='utf8') as f:
			f.write('\t'.join(selected_columns) + '\n')

		# 开始处理
		for i in range(question_dataframe.shape[0]):
			print(i)
			_id, statement_tokens, option_a_tokens, option_b_tokens, option_c_tokens, option_d_tokens, source = question_dataframe.loc[i, selected_columns]
			logging.info(f'{i}|{_id}|{source}')
			row_string = f'{_id}\t'
			for tokens in [statement_tokens, option_a_tokens, option_b_tokens, option_c_tokens, option_d_tokens]:
				try:
					parse_trees = self.long_tokens_to_parse_trees(tokens=tokens)
				except Exception as exception:
					parse_trees = exception
				row_string += f'{parse_trees}\t'
			row_string += f'{source}\n'
			with open(export_path, 'a', encoding='utf8') as f:
				f.write(row_string)

	def build_reference_dependency(self, export_path=REFERENCE_DEPENDENCY_PATH):
		"""2022/06/12 10:29:39
		生成参考书目文档的依存结构"""
		reference_dataframe = pandas.read_csv(REFERENCE_PATH, sep='\t', header=0, dtype=str)	# 预处理后的参数书目文档
		reference_dataframe = reference_dataframe.fillna('')									# 参考书目的section字段存在缺失, 可使用空字符串填充

		# with open(export_path, 'w', encoding='utf8') as f:
			# # 新建文件并写入表头
			# f.write('id\tcontent\n')

		for i in range(19933, reference_dataframe.shape[0]):
			print(i)
			content_tokens = eval(reference_dataframe.loc[i, 'content'])
			# 可能会因为分句过长而无法解析的情况, 报错原因通常是OOM, 采取异常抛出
			try:
				dependencys = self.long_tokens_to_dependencys(tokens=content_tokens)
			except Exception as exception:
				dependencys = exception
			with open(export_path, 'a', encoding='utf8') as f:
				f.write(f'{i}\t{dependencys}\n')

	def build_question_dependency(self, export_path=QUESTION_DEPENDENCY_PATH):
		"""2022/06/12 10:29:39
		生成句法树文档的依存结构"""
		# 合并所有题库(训练集, 验证集, 测试集), 并标明每道题来源于哪个数据集
		dataframes = []
		filepaths = TRAINSET_PATHs + VALIDSET_PATHs + TESTSET_PATHs
		selected_columns = ['id', 'statement', 'option_a', 'option_b', 'option_c', 'option_d']
		for filepath in filepaths:
			dataframe = pandas.read_csv(filepath, sep='\t', header=0)[selected_columns]
			dataframe['source'] = os.path.split(filepath)[-1]
			dataframes.append(dataframe)
		question_dataframe = pandas.concat(dataframes).reset_index(drop=True)
		selected_columns.append('source')

		# 切换数据类型
		question_dataframe['id'] = question_dataframe['id'].astype(str)				# 字段id转为字符串
		question_dataframe['statement'] = question_dataframe['statement'].map(eval)	# 字段statement用eval函数转为分词列表
		question_dataframe['option_a'] = question_dataframe['option_a'].map(eval)	# 字段option_a用eval函数转为分词列表
		question_dataframe['option_b'] = question_dataframe['option_b'].map(eval)	# 字段option_b用eval函数转为分词列表
		question_dataframe['option_c'] = question_dataframe['option_c'].map(eval)	# 字段option_c用eval函数转为分词列表
		question_dataframe['option_d'] = question_dataframe['option_d'].map(eval)	# 字段option_d用eval函数转为分词列表

		# 新建文件并写入表头
		with open(export_path, 'w', encoding='utf8') as f:
			f.write('\t'.join(selected_columns) + '\n')

		# 开始处理
		for i in range(question_dataframe.shape[0]):
			print(i)
			_id, statement_tokens, option_a_tokens, option_b_tokens, option_c_tokens, option_d_tokens, source = question_dataframe.loc[i, selected_columns]
			logging.info(f'{i}|{_id}|{source}')
			row_string = f'{_id}\t'
			for tokens in [statement_tokens, option_a_tokens, option_b_tokens, option_c_tokens, option_d_tokens]:
				try:
					dependencys = self.long_tokens_to_dependencys(tokens=tokens)
				except Exception as exception:
					dependencys = exception
				row_string += f'{dependencys}\t'
			row_string += f'{source}\n'
			with open(export_path, 'a', encoding='utf8') as f:
				f.write(row_string)

	def long_tokens_to_parse_trees(self, tokens):
		"""2022/03/16 21:25:23
		长分词序列转解析树: 
		利用self.split_symbols中的字符进行分割, 一段话可能会生成多棵句法解析树"""
		parse_trees = []
		parsed_tokens = []

		for token in tokens:
			parsed_tokens.append(token)
			if token in self.split_symbols and parsed_tokens:
				# 2022/03/10 16:13:24 根据给定的分隔符对分词列表进行划分处理
				parse_tree = generate_parse_tree(tokens=parsed_tokens, language='chinese')
				parse_tree = list(map(lambda tree: self.regex_compiler.sub(' ', str(tree)), parse_tree))
				if len(parse_tree) > 1:
					# 绝大多数情况下只会有一棵树: 其实我也不知道什么情况下才能生成多于一棵树
					raise Exception('More than one tree !')
				parse_trees.append(str(parse_tree[0]))
				parsed_tokens = []

		# 最后一部分parsed_tokens处理
		if parsed_tokens:
			parse_tree = generate_parse_tree(tokens=parsed_tokens, language='chinese')
			parse_tree = list(map(lambda tree: self.regex_compiler.sub(' ', str(tree)), parse_tree))
			if len(parse_tree) > 1:
				raise Exception('More than one tree !')
			parse_trees.append(str(parse_tree[0]))
		return parse_trees

	def long_tokens_to_dependencys(self, tokens):
		"""2022/06/14 22:36:52
		长分词序列转依存关系(改自long_tokens_to_parse_tree): 
		利用self.split_symbols中的字符进行分割, 一段话可能会生成多个依存关系图"""
		dependencys = []
		parsed_tokens = []
		
		for token in tokens:
			parsed_tokens.append(token)
			if token in self.split_symbols and parsed_tokens:
				# 2022/03/10 16:13:24 根据给定的分隔符对分词列表进行划分处理
				dependency = generate_dependency(tokens=parsed_tokens, language='chinese')
				if len(dependency) > 1: 
					# 绝大多数情况下只会有一张图: 其实我也不知道什么情况下才能生成多于一张图
					raise Exception('More than one dependency !')
				print(parsed_tokens, dependency[0].triples())
				dependencys.append(list(dependency[0].triples()))
				parsed_tokens = []
		
		# 最后一部分parsed_tokens处理
		if parsed_tokens:
			dependency = generate_dependency(tokens=parsed_tokens, language='chinese')
			if len(dependency) > 1:
				raise Exception('More than one dependency !')
			dependencys.append(list(dependency[0].triples()))
		return dependencys


