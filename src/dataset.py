# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# 数据预处理相关方法

if __name__ == '__main__':
	import sys
	sys.path.append('../')

import os
import time
import dill
import torch
import numpy
import pandas
import gensim
import logging
import warnings

from copy import deepcopy
from functools import partial
from collections import Counter
from dgl.data import DGLDataset
from gensim.corpora import Dictionary, MmCorpus
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from setting import *
from config import RetrievalModelConfig, EmbeddingModelConfig

from src.data_tools import load_stopwords, encode_answer, decode_answer, chinese_to_number, filter_stopwords
from src.retrieval_model import GensimRetrievalModel
from src.embedding_model import GensimEmbeddingModel, TransformersEmbeddingModel
from src.graph_tools import generate_pos_tags_from_parse_tree, parse_tree_to_graph, traverse_parse_tree
from src.utils import load_args, timer


# 生成数据加载器
# 把这个函数写在Dataset类里作为类方法会报错找不到__collate_id函数, 目前没有找到很好的解决方案
# 如果放到data_tools.py中的话会导致循环调用, 因此只能先放在这里了
def generate_basic_dataloader(args, mode='train', do_export=False, pipeline='choice', for_debug=False):
	dataset = BasicDataset(args=args, 
						   mode=mode, 
						   do_export=do_export, 
						   pipeline=pipeline,
						   for_debug=for_debug)
	column = dataset.data.columns.tolist()
	if mode.startswith('train'):
		batch_size = args.train_batch_size
		shuffle = True
	if mode.startswith('valid'):
		batch_size = args.valid_batch_size
		shuffle = False
	if mode.startswith('test'):
		batch_size = args.test_batch_size
		shuffle = False
	
	def _collate_fn(_batch_data):
		# _batch_data样例参考
		# [id                                                               1_1304
		# question              [3119, 2443, 990, 13, 6, 259, 0, 0, 0, 0, 0, 0...
		# options               [[29205, 13, 31019, 8388, 45, 2488, 29147, 215...
		# type                                                                  0
		# reference             [[744, 1071, 746, 4260, 31216, 45, 34, 4260, 3...
		# subject                                                      [15, 3, 2]
		# statement_pos_tags    [JJ, NN, VA, DEC, VC, PU, , , , , , , , , , , ...
		# option_a_pos_tags     [NR, DEG, NN, NN, NN, P, NN, NN, VV, JJ, NN, P...
		# option_b_pos_tags     [NR, AD, VV, VV, CD, NN, DEG, NN, , , , , , , ...
		# option_c_pos_tags     [NR, NN, AD, VV, NN, NN, NN, , , , , , , , , ,...
		# option_d_pos_tags     [NR, P, NN, LC, PU, VV, NN, NN, CC, NN, NN, NN...
		# reference_pos_tags    [[NR, CD, JJ, NN, VV, NN, PU, NN, NN, NN, VC, ...
		# statement_tree        [(ROOT (IP (NP (CP (IP (NP (ADJP (JJ 下列)) (NP ...
		# option_a_tree         [(ROOT (IP (NP (DNP (NP (NR 汉代)) (DEG 的)) (NP ...
		# option_b_tree         [(ROOT (IP (NP (NR 唐律)) (VP (ADVP (AD 最早)) (VP...
		# option_c_tree         [(ROOT (IP (NP (NP (NR 北魏)) (NP (NN 太武帝))) (VP...
		# option_d_tree         [(ROOT (IP (NP (NR 明朝)) (VP (PP (P 在) (LCP (NP...
		# reference_tree        [[(ROOT (IP (NP (NP (NR -LRB-)) (QP (CD 5)) (A...
		# label_choice                                                         13
		# Name: 2, dtype: object, id                                                               1_1329
		# question              [3119, 699, 1917, 13, 1190, 2443, 5, 2351, 13,...
		# options               [[2668, 3265, 655, 1432, 13, 759, 5, 534, 277,...
		# type                                                                  0
		# reference             [[1826, 820, 3055, 5, 1917, 1627, 3066, 71, 33...
		# subject                                                     [2, 11, 18]
		# statement_pos_tags    [JJ, JJ, NN, DEG, JJ, NN, PU, VA, DEC, VC, PU,...
		# option_a_pos_tags     [AD, VV, NN, NN, DEG, NN, PU, AD, VE, NN, VV, ...
		# option_b_pos_tags     [NN, LC, CC, NN, LC, VE, NN, DEG, NN, PU, NR, ...
		# option_c_pos_tags     [NN, AD, VV, SP, PU, NN, VV, VV, PN, AD, VV, ,...
		# option_d_pos_tags     [NN, AD, VV, VV, VV, CC, VV, LC, VV, VV, SP, P...
		# reference_pos_tags    [[P, NN, VV, PU, NN, AD, VV, AD, VV, SP, PU, N...
		# statement_tree        [(ROOT (NP (DNP (NP (ADJP (JJ 下列)) (ADJP (JJ 有...
		# option_a_tree         [(ROOT (IP (VP (ADVP (AD 凡是)) (VP (VV 了解) (NP ...
		# option_b_tree         [(ROOT (IP (LCP (LCP (NP (NN 生理)) (LC 上)) (CC ...
		# option_c_tree         [(ROOT (CP (IP (NP (NN 未成年人)) (VP (ADVP (AD 无法...
		# option_d_tree         [(ROOT (CP (IP (NP (NN 证人)) (VP (ADVP (AD 没有))...
		# reference_tree        [[(ROOT (IP (VP (PP (P 经) (NP (NN 人民法院))) (VP ...
		# label_choice                                                         15
		# Name: 7, dtype: object]
		
		def __collate_id():
			return [__data['id'] for __data in _batch_data]

		def __collate_type():
			return [__data['type'] for __data in _batch_data]								# 这个其实没有什么用, 因为我已经把他给扔了
			
		def __collate_subject():
			return torch.LongTensor([__data['subject'] for __data in _batch_data])			# 2022/04/10 17:37:59 这个之后要用的话可能还得转为onehot
		
		def __collate_label_choice():
			return torch.LongTensor([__data['label_choice'] for __data in _batch_data])		# 0-15的选择题答案编码值: 数据类型Long
			
		def __collate_label_judgment():
			return torch.LongTensor([__data['label_judgment'] for __data in _batch_data])	# 零一的判断题答案编码值: 数据类型Long
		
		# 2022/04/10 17:42:29 目前基本上已经告别judgment的pipeline了, 因此这个字段处理差不多算是废弃了
		def __collate_option_id():
			return [__data['option_id'] for __data in _batch_data]							# 选项号需要记录进来: 限判断题
		
		# 2022/05/20 16:04:41 设置一个
		def __collate_reference_index_by_subject():
			return [__data['reference_index_by_subject'] for __data in _batch_data]
		
		# 2022/04/10 15:16:07 词嵌入的处理, 分为顺序编码(数据类型为Long)与标准嵌入(数据类型为Float)
		if args.word_embedding is None and args.document_embedding is None:
			# 不使用词向量或文档向量的情况, 即使用顺序编号编码, 数据类型是long
			def __collate_question():
				return torch.LongTensor([__data['question'] for __data in _batch_data])

			def __collate_reference():
				return torch.LongTensor([__data['reference'] for __data in _batch_data])
				
			def __collate_options():
				# 选择题特有字段: 四个选项
				return torch.LongTensor([__data['options'] for __data in _batch_data])
				
			def __collate_option():
				# 判断题特有字段: 一个选项
				return torch.LongTensor([__data['option'] for __data in _batch_data])
				
		else:
			# 否则即使用向量转化, 此时转化为float类型
			def __collate_question():
				if isinstance(_batch_data[0]['question'], numpy.ndarray):
					return torch.FloatTensor([__data['question'] for __data in _batch_data])
				elif isinstance(_batch_data[0]['question'], torch.Tensor):
					return torch.stack([__data['question'] for __data in _batch_data])
				else:
					raise NotImplementedError(type(_batch_data[0]['question']))
					
			def __collate_reference():
				if isinstance(_batch_data[0]['reference'], numpy.ndarray):
					return torch.FloatTensor([__data['reference'] for __data in _batch_data])
				elif isinstance(_batch_data[0]['reference'], torch.Tensor):
					return torch.stack([__data['reference'] for __data in _batch_data])
				else:
					raise NotImplementedError(type(_batch_data[0]['reference']))
				
			def __collate_options():
				if isinstance(_batch_data[0]['options'], numpy.ndarray):
					return torch.FloatTensor([__data['options'] for __data in _batch_data])
				elif isinstance(_batch_data[0]['options'], torch.Tensor):
					return torch.stack([__data['options'] for __data in _batch_data])
				elif isinstance(_batch_data[0]['options'], list):
					# 2021/12/27 22:35:27 目前只有options可能存在batch_data的每一个元素是一个列表, 该列表里面是四个选项的嵌入向量
					# 2021/12/27 22:36:51 原因是options可能会涉及要转为judgment形式, 需要expand, 我担心如果不是list的话可能会失败, 因此只保留了options的list格式, reference之前也是list, 我已经转为numpy.ndarray了
					if isinstance(_batch_data[0]['options'][0], numpy.ndarray):
						# 2022/02/21 12:07:21 如果使用的是doc2vec, 则是这种情况, 因为doc2vec是针对段落直接进行编码
						return torch.FloatTensor(numpy.stack([numpy.stack(__data['options']) for __data in _batch_data]))
					elif isinstance(_batch_data[0]['options'][0], torch.Tensor):
						# 2022/02/21 12:08:24 如果使用的是BERT或者word2vec, 则会在dataset相关处理中被转为torch.Tensor
						return torch.stack([torch.stack(__data['options']) for __data in _batch_data])
					else:
						raise NotImplementedError(type(_batch_data[0]['options'][0]))
				else:
					raise NotImplementedError(type(_batch_data[0]['options']))
				
			def __collate_option():
				if isinstance(_batch_data[0]['option'], numpy.ndarray):
					return torch.FloatTensor([__data['option'] for __data in _batch_data])		
				elif isinstance(_batch_data[0]['option'], torch.Tensor):
					return torch.stack([__data['option'] for __data in _batch_data])
				else:
					raise NotImplementedError(type(_batch_data[0]['option']))	

		# 2022/04/09 10:26:18 增加词性特征和句法树的处理逻辑
		if args.use_pos_tags:
			
			def __collate_pos_tags(__column):
				# 2022/05/18 23:22:09 为了对应padding字符的编号为0, 决定将填补从-1改为0, 其余所有编号依次+1
				# 2022/05/18 23:40:40 __data[__column]就是类似[NR, P, NN, AD, VV, PU, VV, VV, PN, PU, CD, M,...]的一维列表
				# return torch.LongTensor([list(map(lambda __pos_tag: STANFORD_POS_TAG_INDEX.get(__pos_tag, -1), __data[__column])) for __data in _batch_data])
				return torch.LongTensor([list(map(lambda __pos_tag: STANFORD_POS_TAG_INDEX.get(__pos_tag, -1) + 1, __data[__column])) for __data in _batch_data])
			
			def __collate_statement_pos_tags():			
				return __collate_pos_tags(__column='statement_pos_tags')
				
			def __collate_option_a_pos_tags():
				return __collate_pos_tags(__column='option_a_pos_tags')

			def __collate_option_b_pos_tags():
				return __collate_pos_tags(__column='option_b_pos_tags')
				
			def __collate_option_c_pos_tags():
				return __collate_pos_tags(__column='option_c_pos_tags')

			def __collate_option_d_pos_tags():
				return __collate_pos_tags(__column='option_d_pos_tags')
			
			# 2022/05/18 23:31:44 参考书目文档的词性标注处理
			if args.use_reference:
				
				def __collate_reference_pos_tags():
					# 2022/05/18 23:49:03 与__collate_pos_tags的区别在于这里的__data['reference_pos_tags']是二维列表, 第一维是参考段落的数量
					return torch.LongTensor([[list(map(lambda ___pos_tag: 0 if ___pos_tag == '' else STANFORD_POS_TAG_INDEX[___pos_tag] + 1, ___pos_tags)) for ___pos_tags in __data['reference_pos_tags']] for __data in _batch_data])
						
			# 2022/04/15 22:29:45 若使用句法树, 则必然使用词性标注, 因此嵌套在该循环中
			if args.use_parse_tree:
				
				def __collate_parse_tree(__column):
					# return [[parse_tree_to_graph(parse_tree=__parse_tree, display=False, return_type='dgl', ignore_text=True) for __parse_tree in __data[__column]] for __data in _batch_data]
					return [__data[__column] for __data in _batch_data]
					
				def __collate_statement_tree():
					return __collate_parse_tree(__column='statement_tree')
					
				def __collate_option_a_tree():
					return __collate_parse_tree(__column='option_a_tree')
					
				def __collate_option_b_tree():
					return __collate_parse_tree(__column='option_b_tree')
					
				def __collate_option_c_tree():
					return __collate_parse_tree(__column='option_c_tree')
					
				def __collate_option_d_tree():
					return __collate_parse_tree(__column='option_d_tree')
					
				if args.use_reference:
					
					def __collate_reference_tree():
						# return [[[parse_tree_to_graph(parse_tree=___parse_tree, display=False, return_type='dgl', ignore_text=True) for ___parse_tree in ___parse_trees] for ___parse_trees in __data['reference_tree']] for __data in _batch_data]
						return __collate_parse_tree(__column='reference_tree')
						
		_collate_data = {}
		for _column in column:
			_collate_data[_column] = eval(f'__collate_{_column}')()
		return _collate_data

	dataloader = DataLoader(dataset=dataset,
							batch_size=batch_size,
							num_workers=args.num_workers,
							collate_fn=_collate_fn,
							shuffle=shuffle)
	return dataloader


def generate_parse_tree_dataloader(args, mode='train', do_export=False, pipeline='choice', for_debug=False):
	dataset = ParseTreeDataset(args=args, 
							   mode=mode, 
						   	   do_export=do_export, 
						   	   pipeline=pipeline,
						   	   for_debug=for_debug)
						   	   
	if mode.startswith('train'):
		batch_size = args.train_batch_size
		shuffle = True
	if mode.startswith('valid'):
		batch_size = args.valid_batch_size
		shuffle = False
	if mode.startswith('test'):
		batch_size = args.test_batch_size
		shuffle = False


	dataloader = DataLoader(dataset=dataset,
							batch_size=batch_size,
							num_workers=args.num_workers,
							collate_fn=list,	# 其实简单作list变换即可
							shuffle=shuffle)
	return dataloader

class BasicDataset(Dataset):
	"""基础数据管道"""
	def __init__(self, 
				 args, 
				 mode='train', 
				 do_export=False, 
				 pipeline='choice', 
				 for_debug=False):
		"""参数说明:
		:param args			: DatasetConfig配置
		:param mode			: 数据集模式, 详见下面第一行的断言
		:param do_export	: 是否导出self.data
		:param pipeline		: 目前考虑judgment与choice两种模式
		:param for_debug	: 2021/12/30 19:21:46 调试模式, 只用少量数据集加快测试效率"""
		self.pipelines = {
			'choice'	: self.choice_pipeline,
			'judgment'	: self.judgment_pipeline,
			'display'	: self.display_pipeline,
		}
		
		assert mode in ['train', 'train_kd', 'train_ca', 'valid', 'valid_kd', 'valid_ca', 'test', 'test_kd', 'test_ca']
		assert pipeline in self.pipelines
		assert args.word_embedding is None or args.document_embedding is None	# 2021/12/27 14:03:07 词嵌入和文档嵌入只能使用一个
		
		# 构造变量转为成员变量
		self.args = deepcopy(args)
		self.mode = mode
		self.do_export = do_export
		self.pipeline = pipeline
		self.for_debug = for_debug
		
		# 根据配置生成对应的成员变量
		if self.args.filter_stopword:
			self.stopwords = load_stopwords(stopword_names=None)
		
		if self.args.use_reference:
			# 2021/12/27 21:24:39 使用参考书目文档必须调用检索模型: 目前只有gensim模块下的检索模型
			_args = load_args(Config=RetrievalModelConfig)
			# 2021/12/27 21:24:29 重置新参数值, 因为传入Dataset类的args参数中的一些值可能与默认值不同, 以传入值为准
			for key in vars(_args):
				if key in self.args:
					_args.__setattr__(key, self.args.__getattribute__(key))
			self.grm = GensimRetrievalModel(args=_args)		
		
		if self.args.word_embedding in GENSIM_EMBEDDING_MODEL_SUMMARY or self.args.document_embedding in GENSIM_EMBEDDING_MODEL_SUMMARY:
			# 2021/12/27 21:24:43 使用gensim模块中的词向量模型或文档向量模型
			_args = load_args(Config=EmbeddingModelConfig)
			# 2021/12/27 21:24:24 重置新参数值, 因为传入Dataset类的args参数中的一些值可能与默认值不同, 以传入值为准
			for key in vars(_args):
				if key in self.args:
					_args.__setattr__(key, self.args.__getattribute__(key))
			self.gem = GensimEmbeddingModel(args=_args)
		
		if self.args.word_embedding in BERT_MODEL_SUMMARY or self.args.document_embedding in BERT_MODEL_SUMMARY:
			# 2021/12/27 21:24:17 使用BERT相关模型
			_args = load_args(Config=EmbeddingModelConfig)
			# 2021/12/27 21:24:20 重置新参数值, 因为传入Dataset类的args参数中的一些值可能与默认值不同, 以传入值为准
			for key in vars(_args):
				if key in self.args:
					_args.__setattr__(key, self.args.__getattribute__(key))
			self.tem = TransformersEmbeddingModel(args=_args)
		
		# 生成数据表
		self.pipelines[pipeline]()
		
		# 导出数据表
		if self.do_export:
			logging.info('导出数据表...')
			self.data.to_csv(COMPLETE_REFERENCE_PATH, sep='\t', header=True, index=False)
	
	@timer
	def choice_pipeline(self):
		"""选择题形式的输入数据, 输出字段有:
		id			: 题目编号
		question	: 题目题干
		option		: 合并后的四个选项
		subject		: use_reference配置为True时生效, 包含num_top_subject个法律门类
		reference	: use_reference配置为True时生效, 包含相关的num_best个参考书目文档段落
		type		: 零一值表示概念题或情景题
		label_choice: train或valid模式时生效, 即题目答案"""
		if self.mode.startswith('train'):
			filepaths = TRAINSET_PATHs[:]
		elif self.mode.startswith('valid'):  # 20211101新增验证集处理逻辑
			filepaths = VALIDSET_PATHs[:]
		elif self.mode.startswith('test'):
			filepaths = TESTSET_PATHs[:]
		else:
			assert False
		max_option_length = self.args.max_option_length
		max_statement_length = self.args.max_statement_length
		max_reference_length = self.args.max_reference_length
		
		# 2022/04/05 23:24:48 代码优化: 记录最后数据的字段
		selected_columns = ['id', 'question', 'options', 'type']

		# 数据集字段预处理
		logging.info('预处理题目题干与选项...')
		start_time = time.time()
		
		# 合并概念题和情景题后的题库
		dataset_dataframe = pandas.concat([pandas.read_csv(filepath, sep='\t', header=0) for filepath in filepaths]).reset_index(drop=True)	
		
		if self.mode.endswith('_kd'):   
			dataset_dataframe = dataset_dataframe[dataset_dataframe['type'] == 0].reset_index(drop=True)	# 筛选概念题
		elif self.mode.endswith('_ca'): 
			dataset_dataframe = dataset_dataframe[dataset_dataframe['type'] == 1].reset_index(drop=True)	# 筛选情景分析题
		else:
			dataset_dataframe = dataset_dataframe.reset_index(drop=True)									# 无需筛选直接重索引
			
		# 2021/12/30 19:50:18 在调试情况下只选取少量数据运行以提高效率
		if self.for_debug:
			dataset_dataframe = dataset_dataframe.loc[:10, :]
			
		dataset_dataframe['id'] = dataset_dataframe['id'].astype(str)				# 字段id转为字符串
		dataset_dataframe['type'] = dataset_dataframe['type'].astype(int)			# 字段type转为整数
		dataset_dataframe['statement'] = dataset_dataframe['statement'].map(eval)	# 字段statement用eval函数转为分词列表
		dataset_dataframe['option_a'] = dataset_dataframe['option_a'].map(eval)		# 字段option_a用eval函数转为分词列表
		dataset_dataframe['option_b'] = dataset_dataframe['option_b'].map(eval)		# 字段option_b用eval函数转为分词列表
		dataset_dataframe['option_c'] = dataset_dataframe['option_c'].map(eval)		# 字段option_c用eval函数转为分词列表
		dataset_dataframe['option_d'] = dataset_dataframe['option_d'].map(eval)		# 字段option_d用eval函数转为分词列表
		
		# 2022/05/17 13:09:01 过滤特殊字符: \u3000与单空格
		if self.args.filter_stopword:
			# 2022/05/20 12:29:27 STANFORD_IGNORED_SYMBOL中目前只有\u3000与单空格(即' ')
			_filter_ignored_symbol = partial(filter_stopwords, stopwords=STANFORD_IGNORED_SYMBOL)
			dataset_dataframe['statement'] = dataset_dataframe['statement'].map(_filter_ignored_symbol)	# 字段statement去除停用词
			dataset_dataframe['option_a'] = dataset_dataframe['option_a'].map(_filter_ignored_symbol)	# 字段option_a去除停用词
			dataset_dataframe['option_b'] = dataset_dataframe['option_b'].map(_filter_ignored_symbol)	# 字段option_b去除停用词
			dataset_dataframe['option_c'] = dataset_dataframe['option_c'].map(_filter_ignored_symbol)	# 字段option_c去除停用词
			dataset_dataframe['option_d'] = dataset_dataframe['option_d'].map(_filter_ignored_symbol)	# 字段option_d去除停用词	

		# 2022/01/15 22:01:48 冗长的条件分支(用于进行词嵌入或句嵌入): 待优化
		if self.args.word_embedding is None and self.args.document_embedding is None:
			# 不使用任何嵌入, 即使用顺序编码值(token2id)进行词嵌入
			
			# token2id字典: 20211212后决定以参考书目文档的token2id为标准, 而非题库的token2id
			token2id_dataframe = pandas.read_csv(REFERENCE_TOKEN2ID_PATH, sep='\t', header=0)
			token2id = {token: _id for token, _id in zip(token2id_dataframe['token'], token2id_dataframe['id'])}			
			
			dataset_dataframe['question'] = dataset_dataframe['statement'].map(self.token_to_id(max_length=max_statement_length, token2id=token2id))																# 题目题干的分词列表转为编号列表
			dataset_dataframe['options'] = dataset_dataframe[['option_a', 'option_b', 'option_c', 'option_d']].apply(self.combine_option(max_length=max_option_length, token2emb=token2id, encode_as='id'), axis=1)	# 题目选项的分词列表转为编号列表并合并	
		
		elif self.args.word_embedding in GENSIM_EMBEDDING_MODEL_SUMMARY:
			# 使用gensim词向量模型进行训练: word2vec, fasttext
			embedding_model_class = eval(GENSIM_EMBEDDING_MODEL_SUMMARY[self.args.word_embedding]['class'])
			embedding_model_path = GENSIM_EMBEDDING_MODEL_SUMMARY[self.args.word_embedding]['model']
			embedding_model = embedding_model_class.load(embedding_model_path)
			token2vector = embedding_model.wv
			self.vector_size = embedding_model.wv.vector_size
			
			# 2021/12/27 21:26:45 这里可以直接删除模型, 直接用wv即可
			del embedding_model

			dataset_dataframe['question'] = dataset_dataframe['statement'].map(self.token_to_vector(max_length=max_statement_length, token2vector=token2vector))															# 题目题干的分词列表转为编号列表
			dataset_dataframe['options'] = dataset_dataframe[['option_a', 'option_b', 'option_c', 'option_d']].apply(self.combine_option(max_length=max_option_length, token2emb=token2vector, encode_as='vector'), axis=1)	# 题目选项的分词列表转为编号列表并合并
		
		elif self.args.word_embedding in BERT_MODEL_SUMMARY:
			# 目前不考虑用BERT模型生成词向量
			raise NotImplementedError
		
		elif self.args.document_embedding in GENSIM_EMBEDDING_MODEL_SUMMARY:
			# 使用gensim文档向量模型进行训练: 目前这里特指doc2vec模型, 代码目前比较硬
			embedding_model_class = eval(GENSIM_EMBEDDING_MODEL_SUMMARY[self.args.document_embedding]['class'])
			embedding_model_path = GENSIM_EMBEDDING_MODEL_SUMMARY[self.args.document_embedding]['model']
			embedding_model = embedding_model_class.load(embedding_model_path)
			
			# 相对来说词向量模型的调用要麻烦一些, 文档向量模型的调用要容易很多, 也不是很好和下面几个成员函数合并, 所以直接写lambda函数来解决了
			# 2021/12/27 22:25:08 注意torch.cat与torch.stack的区别, 前者不会增加维数, 后者则支持拼接得到更高维数的张量
			dataset_dataframe['question'] = dataset_dataframe['statement'].map(embedding_model.infer_vector)
			dataset_dataframe['options'] = dataset_dataframe[['option_a', 'option_b', 'option_c', 'option_d']].apply(lambda _dataframe: [embedding_model.infer_vector(_dataframe[0]),
																																		 embedding_model.infer_vector(_dataframe[1]),
																																		 embedding_model.infer_vector(_dataframe[2]),
																																		 embedding_model.infer_vector(_dataframe[3])], axis=1)

		elif self.args.document_embedding in BERT_MODEL_SUMMARY:	
			# 2022/03/23 10:48:44 之前选择在生成Dataset时调用BERT模型即时生成题干, 选项以及参考书目的嵌入的方法被证实是不可行的(OOM)
			# 2022/03/23 11:23:01 目前已经预先将题干, 选项以及参考书目的BERT嵌入生成好存放在本地
			# # 使用BERT模型生成文档向量: 注意只有BERT模型输出是torch.Tensor, 其他都是numpy.ndarray, 是可以比较容易处理的
			# bert_tokenizer, bert_model = TransformersEmbeddingModel.load_bert_model(model_name=self.args.document_embedding)
			# bert_config = tem.load_bert_config(model_name=self.args.document_embedding)
			
			# def _generate_bert_output(_tokens):
				# # BERT模型无需分词, 直接输入整个句子即可
				# _text = [''.join(_tokens)]
				# _output = self.tem.generate_bert_output(text=_text, tokenizer=bert_tokenizer, model=bert_model, max_length=bert_config['max_position_embeddings'])
				# return _output
				
			# dataset_dataframe['question'] = dataset_dataframe['statement'].map(_generate_bert_output)
			# dataset_dataframe['options'] = dataset_dataframe[['option_a', 'option_b', 'option_c', 'option_d']].apply(lambda _dataframe: [_generate_bert_output(_dataframe[0]),
																																		 # _generate_bert_output(_dataframe[1]),
																																		 # _generate_bert_output(_dataframe[2]),
																																		 # _generate_bert_output(_dataframe[3])], axis=1)
			# 2022/03/23 13:58:07 使用预先生成好的bert_output直接取得嵌入
			reference_bert_outputs = dill.load(open(REFERENCE_POOLER_OUTPUT_PATH, 'rb'))
			question_bert_outputs = dill.load(open(QUESTION_POOLER_OUTPUT_PATH, 'rb'))
			
			question_id_to_bert_output = {
				_id: {
					'statement'	: statement,
					'option_a'	: option_a,
					'option_b'	: option_b,
					'option_c'	: option_c,
					'option_d'	: option_d,
				} for _id, statement, option_a, option_b, option_c, option_d in zip(question_bert_outputs['id'], 
																					question_bert_outputs['statement'], 
																					question_bert_outputs['option_a'], 
																					question_bert_outputs['option_b'], 
																					question_bert_outputs['option_c'],
																					question_bert_outputs['option_d'])
			}
			
			# 2022/03/23 14:52:09 取[0]的原因是原先的形状是(1, 768), 希望是(768, )
			dataset_dataframe['question'] = dataset_dataframe['id'].map(lambda _id: question_id_to_bert_output[_id]['statement'][0])
			dataset_dataframe['options'] = dataset_dataframe['id'].map(lambda _id: [question_id_to_bert_output[_id]['option_a'][0],
																					question_id_to_bert_output[_id]['option_b'][0],
																					question_id_to_bert_output[_id]['option_c'][0],
																					question_id_to_bert_output[_id]['option_d'][0]])
			
		else:
			# 目前尚未完成其他词嵌入的使用
			raise NotImplementedError

		# 2022/04/05 23:54:24 参考文献相关字段预处理: 额外添加subject与reference字段
		if self.args.use_reference:
			# 2022/04/05 23:48:53 为selected_columns添加新字段: ['id', 'question', 'options', 'type', 'reference', 'subject']
			selected_columns.append('reference')
			selected_columns.append('subject')
			
			# 2022/05/20 13:19:29 为selected_columns添加新字段: ['id', 'question', 'options', 'type', 'reference', 'subject', 'reference_index_by_subject']
			# 2022/05/20 13:19:58 reference_index_by_subject并没有实际用到, 仅用于辅助调试
			selected_columns.append('reference_index_by_subject')
			
			# 加载文档检索模型相关内容
			dictionary_path = GENSIM_RETRIEVAL_MODEL_SUMMARY[self.args.retrieval_model_name]['dictionary']
			if dictionary_path is None:		
				# logentropy模型的dictionary字段是None
				dictionary_path = REFERENCE_DICTIONARY_PATH
			dictionary = Dictionary.load(dictionary_path)
			similarity = self.grm.build_similarity(model_name=self.args.retrieval_model_name, dictionary=None, corpus=None, num_best=None)
			sequence = GensimRetrievalModel.load_sequence(model_name=self.args.retrieval_model_name, subject=None)
			
			reference_dataframe = pandas.read_csv(REFERENCE_PATH, sep='\t', header=0)
			reference_dataframe['content'] = reference_dataframe['content'].map(eval)
			index2subject = {index: LAW2SUBJECT.get(law, law) for index, law in enumerate(reference_dataframe['law'])}		# 记录reference_dataframe中每一行对应的法律门类
			
			# 新生成的几个字段说明:
			# query_result		: 形如[(4, 0.8), (7, 0.1), (1, 0.1)], 列表长度为args.num_best
			# reference_index	: 将[4, 7, 1]给抽取出来
			# reference			: 将[4, 7, 1]对应的参考书目文档的段落的分词列表给抽取出来并转为编号列表
			# subject			: 题目对应的args.num_top_subject个候选法律门类
			logging.info('生成查询得分向量...')
			dataset_dataframe['query_result'] = dataset_dataframe[['statement', 'option_a', 'option_b', 'option_c', 'option_d']].apply(self.generate_query_result(dictionary=dictionary, 
																																								  similarity=similarity, 
																																								  sequence=sequence), axis=1)
			
			# 2022/04/15 23:33:26 这个reference_index是第一遍全文检索的index, 目前已经被弃用, 我们会在特定的几个subject中检索index
			dataset_dataframe['reference_index'] = dataset_dataframe['query_result'].map(lambda result: list(map(lambda x: x[0], result)))
			

			logging.info('检索参考书目文档段落...')
			
			
			# 2022/02/28 19:01:00 似乎使用文档全集进行检索得到的参考文档已经deprecated了, 后面都是用分门类的检索模型进行检索的
			# # 2022/01/15 22:01:48 冗长的条件分支: 待优化
			# if self.args.word_embedding is None and self.args.document_embedding is None:
				# dataset_dataframe['reference'] = dataset_dataframe['reference_index'].map(self.find_reference_by_index(max_length=max_reference_length, 
																													   # token2emb=token2id, 
																													   # reference_dataframe=reference_dataframe,
																													   # encode_as='id'))

			# elif self.args.word_embedding in GENSIM_EMBEDDING_MODEL_SUMMARY:
				# # logging.info(f'token2vector.vector_size: {token2vector.vector_size}')
				# # logging.info(f'self.vector_size: {self.vector_size}')
				# dataset_dataframe['reference'] = dataset_dataframe['reference_index'].map(self.find_reference_by_index(max_length=max_reference_length, 
																													   # token2emb=token2vector, 
																													   # reference_dataframe=reference_dataframe,
																													   # encode_as='vector'))		

			# elif self.args.word_embedding in BERT_MODEL_SUMMARY:
				# # 目前不考虑用BERT模型生成词向量
				# raise NotImplementedError
				
			# elif self.args.document_embedding in GENSIM_EMBEDDING_MODEL_SUMMARY:
				# # 2021/12/27 22:17:56 使用gensim文档向量模型进行训练: 目前这里特指doc2vec模型, 代码目前比较硬
				# # 2021/12/27 22:20:22 改自self.find_reference_by_index函数, 其实还是可以用lambda一行写完的, 可读性差了一些
				# dataset_dataframe['reference'] = dataset_dataframe['reference_index'].map(lambda _reference_index: numpy.stack([embedding_model.infer_vector(reference_dataframe.loc[_index, 'content']) for _index in _reference_index]))
			
			# elif self.args.document_embedding in BERT_MODEL_SUMMARY:	
				# # 2021/12/27 22:42:30 使用BERT模型生成文档向量: 注意只有BERT模型输出是torch.Tensor, 其他都是numpy.ndarray, 是可以比较容易处理的
				# def _generate_bert_output(_reference_index):
					# _reference_tensors = []
					# for _index in _reference_index:
						# # 2021/12/27 22:42:38 BERT模型无需分词, 直接输入整个句子即可
						# _text = [''.join(reference_dataframe.loc[_index, 'content'])]
						# _output = self.tem.generate_bert_output(text=_text, tokenizer=bert_tokenizer, model=bert_model, max_length=bert_config['max_position_embeddings'])
						# _reference_tensors.append(_output)
					# # 2021/12/27 22:42:42 不要输出为列表
					# return torch.stack(_reference_tensors)
				
				# # 2021/12/27 22:43:15 其实上面这个函数可以一行写完的, 只是太长了一些
				# # _generate_bert_output = lambda _reference_index: torch.stack([bert_model(**bert_tokenizer(''.join(reference_dataframe.loc[_index, 'content']), return_tensors='pt', padding=True)).get(self.args.tem.bert_output) for _index in _reference_index])

				# dataset_dataframe['reference'] = dataset_dataframe['reference_index'].map(_generate_bert_output)
				
			# else:
				# # 目前尚未完成其他词嵌入的使用
				# raise NotImplementedError	
			
			logging.info('填充subject字段的缺失值...')
			dataset_dataframe['subject'] = dataset_dataframe[['reference_index', 'subject']].apply(self.fill_subject(index2subject), axis=1)
			
			# 2022/01/14 23:35:20 根据填充好的subject字段进行精确检索, 即只在num_top_subject个subject中检索
			subject_summary = {}
			for subject in SUBJECT2INDEX:
				summary = {}
				dictionary_path = GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['summary'][self.args.retrieval_model_name]['dictionary']
				if dictionary_path is None:		
					# logentropy模型的dictionary字段是None
					dictionary_path = GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['dictionary']
				dictionary = Dictionary.load(dictionary_path)
				corpus = MmCorpus(GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['summary'][self.args.retrieval_model_name]['corpus'])
				# 2022/01/15 22:26:41 关于num_best为什么是self.args.num_best_per_subject * self.args.num_top_subject而非self.args.num_best_per_subject
				# 2022/01/15 22:28:15 因为我考虑可能subject字段凑不满self.args.num_top_subject的个数, 所以可能需要用前面的来递补
				# 2022/01/15 22:28:07 因此num_best设定为最坏情况的值, 即只有一个候选subject, 此时需要用到num_best_per_subject * num_top_subject的数值, 而非num_best_per_subject
				# 2022/02/01 16:18:46 今天发现即便设置num_best, 也无法确保查询结果有num_best个, 甚至会少于num_best_per_subject个, 这使得函数retrieve_references_by_subject中的相关逻辑可能报错
				similarity = self.grm.build_similarity(model_name=self.args.retrieval_model_name, 
													   dictionary=dictionary,
													   corpus=corpus,
													   num_best=self.args.num_best_per_subject * self.args.num_top_subject)
				sequence = GensimRetrievalModel.load_sequence(model_name=self.args.retrieval_model_name, subject=subject)
				
				# 2022/03/23 14:16:53 改为在reset_index时不丢弃原先的index
				# 2022/03/23 14:16:53 原因是这里得到的是分门类的reference, 但是预先生成的参考书目文档的BERT输出是按照原先reference的顺序排列的, 因此之前的index是需要保留的
				# summary['reference_dataframe'] = reference_dataframe[reference_dataframe['law'] == SUBJECT2LAW.get(subject, subject)].reset_index(drop=True)
				summary['reference_dataframe'] = reference_dataframe[reference_dataframe['law'] == SUBJECT2LAW.get(subject, subject)].reset_index(drop=False)
				summary['dictionary'] = Dictionary.load(GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['dictionary'])
				summary['corpus'] = corpus
				summary['similarity'] = similarity
				summary['sequence'] = sequence
				subject_summary[subject] = deepcopy(summary)
				del summary
			
			# 2022/04/15 23:38:09 先把按subject检索的index找出来: 这里的index就是原先reference的索引, 取值范围0-24717
			dataset_dataframe['reference_index_by_subject'] = dataset_dataframe[['statement', 'option_a', 'option_b', 'option_c', 'option_d', 'subject']].apply(self.retrieve_reference_index_by_subject(subject_summary=subject_summary), axis=1)
			
			# 2022/01/15 23:05:23 冗长的条件分支: 待优化
			if self.args.word_embedding is None and self.args.document_embedding is None:
				references = dataset_dataframe[['statement', 'option_a', 'option_b', 'option_c', 'option_d', 'subject']].apply(self.retrieve_references_by_subject(subject_summary=subject_summary), axis=1)
				_token_to_id = self.token_to_id(max_length=max_reference_length, token2id=token2id)
				dataset_dataframe['reference'] = references.map(lambda _references: numpy.stack([_token_to_id(_reference_tokens) for _reference_tokens in _references]))
						
			elif self.args.word_embedding in GENSIM_EMBEDDING_MODEL_SUMMARY:
				references = dataset_dataframe[['statement', 'option_a', 'option_b', 'option_c', 'option_d', 'subject']].apply(self.retrieve_references_by_subject(subject_summary=subject_summary), axis=1)
				_token_to_vector = self.token_to_vector(max_length=max_reference_length, token2vector=token2vector)
				dataset_dataframe['reference'] = references.map(lambda _references: numpy.stack([_token_to_vector(_reference_tokens) for _reference_tokens in _references]))
				
			elif self.args.word_embedding in BERT_MODEL_SUMMARY:
				# 目前不考虑用BERT模型生成词向量
				raise NotImplementedError
				
			elif self.args.document_embedding in GENSIM_EMBEDDING_MODEL_SUMMARY:
				# 使用文档编码模型
				references = dataset_dataframe[['statement', 'option_a', 'option_b', 'option_c', 'option_d', 'subject']].apply(self.retrieve_references_by_subject(subject_summary=subject_summary), axis=1)
				dataset_dataframe['reference'] = references.map(lambda _references: numpy.stack([embedding_model.infer_vector(_reference_tokens) for _reference_tokens in _references]))
				
			elif self.args.document_embedding in BERT_MODEL_SUMMARY:
				# 2022/03/23 10:48:44 之前选择在生成Dataset时调用BERT模型即时生成题干, 选项以及参考书目的嵌入的方法被证实是不可行的(OOM)
				# 2022/03/23 11:23:01 目前已经预先将题干, 选项以及参考书目的BERT嵌入生成好存放在本地
				# # 2022/01/15 23:01:33 仿照前面dataset_dataframe['reference']的写法
				# def _generate_bert_output(_references):
					# _reference_tensors = []
					# for _reference_tokens in _references:
						# # 2021/12/27 22:42:38 BERT模型无需分词, 直接输入整个句子即可
						# _text = [''.join(_reference_tokens)]
						# _output = self.tem.generate_bert_output(text=_text, tokenizer=bert_tokenizer, model=bert_model, max_length=bert_config['max_position_embeddings'])
						# _reference_tensors.append(_output)
					# # 2021/12/27 22:42:42 不要输出为列表
					# return torch.stack(_reference_tensors)
				# references = dataset_dataframe[['statement', 'option_a', 'option_b', 'option_c', 'option_d', 'subject']].apply(self.retrieve_references_by_subject(subject_summary=subject_summary), axis=1)
				# dataset_dataframe['reference'] = references.map(_generate_bert_output)
				
				# 2022/04/05 21:13:02 因为可能reference数量不足, index填充值为-1, 此时转成的张量是与BERT输出形状相同的零向量
				# 2022/05/07 23:50:39 reference_index_by_subject字段被调整到本条件分支前面的位置, 因为下面的reference_pos_tags和reference_parse_tree需要用到
				# dataset_dataframe['reference_index_by_subject'] = dataset_dataframe[['statement', 'option_a', 'option_b', 'option_c', 'option_d', 'subject']].apply(self.retrieve_reference_index_by_subject(subject_summary=subject_summary), axis=1)
				dataset_dataframe['reference'] = dataset_dataframe['reference_index_by_subject'].map(lambda index: torch.stack([reference_bert_outputs[indice][0] if indice >= 0 else torch.zeros((self.args.bert_hidden_size, )) for indice in index]))
			
			else:
				# 目前尚未完成其他词嵌入的使用
				raise NotImplementedError
	
		# 2022/04/06 00:18:52 做一个简单的断言: 若使用句法树, 则必然使用词性
		if self.args.use_parse_tree:
			assert self.args.use_pos_tags
		
		logging.info('生成词性特征与句法树...')
		
		# 2022/04/05 23:58:47 词性特征与句法树
		if self.args.use_pos_tags:
			# 2022/04/06 00:17:34 增加新字段: 选项与题干的词性标签
			selected_columns.append('statement_pos_tags')
			selected_columns.append('option_a_pos_tags')
			selected_columns.append('option_b_pos_tags')
			selected_columns.append('option_c_pos_tags')
			selected_columns.append('option_d_pos_tags')
				
			# 2022/04/06 00:17:40 读取题干句法树并预处理
			question_parse_tree_dataframe = pandas.read_csv(QUESTION_PARSE_TREE_PATH, sep='\t')
			question_parse_tree_dataframe = question_parse_tree_dataframe[['id', 'statement', 'option_a', 'option_b', 'option_c', 'option_d']]
			question_parse_tree_dataframe.columns = ['id', 'statement_tree', 'option_a_tree', 'option_b_tree', 'option_c_tree', 'option_d_tree']	
			question_parse_tree_dataframe['statement_tree'] = question_parse_tree_dataframe['statement_tree'].map(eval)
			question_parse_tree_dataframe['option_a_tree'] = question_parse_tree_dataframe['option_a_tree'].map(eval)
			question_parse_tree_dataframe['option_b_tree'] = question_parse_tree_dataframe['option_b_tree'].map(eval)
			question_parse_tree_dataframe['option_c_tree'] = question_parse_tree_dataframe['option_c_tree'].map(eval)
			question_parse_tree_dataframe['option_d_tree'] = question_parse_tree_dataframe['option_d_tree'].map(eval)
			
			# 2022/05/07 23:59:48 与数据表进行左连接: 得到题干与选项的句法树
			n_rows_before = dataset_dataframe.shape[0]
			dataset_dataframe = dataset_dataframe.merge(question_parse_tree_dataframe, on='id', how='left')
			n_rows_after = dataset_dataframe.shape[0]
			assert n_rows_before == n_rows_after, f'数据表连接不合法: 长度发生变化{n_rows_before}->{n_rows_after}'
			
			def _generate_pos_tags_from_parse_trees(_max_length, _padding_tag=''):
				# 2022/04/10 12:00:51 graph_tools.py中的generate_pos_tags_from_parse_tree函数只能应对一棵树的情况
				# 2022/04/10 12:01:37 事实上因为句子都很长, 因此通常会生成多棵树
				# 2022/05/08 19:13:22 发现需要作填补, 添加_max_length与_padding_tag两个参数				
				def __generate_pos_tags_from_parse_trees(__parse_trees):
					__pos_tags = []
					for __parse_tree in __parse_trees:
						__pos_tags_and_tokens = generate_pos_tags_from_parse_tree(parse_tree=__parse_tree)
						__pos_tags.extend(list(map(lambda __pos_tag_and_token: __pos_tag_and_token[0], __pos_tags_and_tokens)))
					# 填补或截断
					if len(__pos_tags) >= _max_length:
						return __pos_tags[: _max_length]
					else:
						return __pos_tags + [''] * (_max_length - len(__pos_tags))
				return __generate_pos_tags_from_parse_trees
			
			# 2022/05/08 00:03:25 生成需要新增的字段
			dataset_dataframe['statement_pos_tags'] = dataset_dataframe['statement_tree'].map(_generate_pos_tags_from_parse_trees(_max_length=self.args.max_statement_length, _padding_tag=''))
			dataset_dataframe['option_a_pos_tags'] = dataset_dataframe['option_a_tree'].map(_generate_pos_tags_from_parse_trees(_max_length=self.args.max_option_length, _padding_tag=''))
			dataset_dataframe['option_b_pos_tags'] = dataset_dataframe['option_b_tree'].map(_generate_pos_tags_from_parse_trees(_max_length=self.args.max_option_length, _padding_tag=''))
			dataset_dataframe['option_c_pos_tags'] = dataset_dataframe['option_c_tree'].map(_generate_pos_tags_from_parse_trees(_max_length=self.args.max_option_length, _padding_tag=''))
			dataset_dataframe['option_d_pos_tags'] = dataset_dataframe['option_d_tree'].map(_generate_pos_tags_from_parse_trees(_max_length=self.args.max_option_length, _padding_tag='')) 

			# 2022/04/29 10:27:48 参考文献部分的句法树和词性处理部分放在这里, 如果放在use_reference的分支里显得那个分支太冗长了
			if self.args.use_reference:
				# 2022/05/06 22:17:34 增加新字段: 参考文献的词性标注
				selected_columns.append('reference_pos_tags')
				
				# 2022/05/08 00:02:39 读取参考文献句法树并生成对应字典
				# 2022/05/08 00:06:13 便于后续根据主表dataset_dataframe的reference_index_by_subject字段来生成对应的若干句法树和词性标注
				reference_parse_tree_dataframe = pandas.read_csv(REFERENCE_PARSE_TREE_PATH, sep='\t')
				reference_parse_tree_dataframe = reference_parse_tree_dataframe[['id', 'content']]	
				reference_id2trees = {int(_id): eval(trees) for _id, trees in zip(reference_parse_tree_dataframe['id'], reference_parse_tree_dataframe['content'])}
				
				# 2022/05/08 00:18:07 根据每个问题检索得到的参考文档索引生成对应的词性标注
				# 2022/05/08 00:25:08 注意先filter是因为index中存在一些-1值表示为空
				# 2022/05/19 23:54:08 dataset_dataframe['reference_tree']整体形如三维列表, 三个维度分别为数据表长度, 参数段落数量, 每个参考段落的解析树数量(数量不定), 即[[[tree1, tree2, ...], [...], ..., [...]], [[tree1, tree2, ...], [...], ..., [...]], ..., [[tree1, tree2, ...], [...], ..., [...]]]
				# 2022/05/20 17:39:37 如果过滤掉值为-1的reference_index_by_subject, 可能会导致长度不足args.num_top_subject * num_best_per_subject
				# 2022/05/20 17:39:37 关于这一点详见retrieve_reference_index_by_subject函数中的注释, 关于gensim.Similarity的num_best参数的实际意义, 即数量可能不足
				# 2022/05/20 17:39:37 目前的做法是给值为-1的reference_index_by_subject赋予空列表作为其句法树
				# dataset_dataframe['reference_tree'] = dataset_dataframe['reference_index_by_subject'].map(lambda index: [reference_id2trees[true_indice] for true_indice in filter(lambda x: not x == -1, index)])
				dataset_dataframe['reference_tree'] = dataset_dataframe['reference_index_by_subject'].map(lambda index: [[] if indice == -1 else reference_id2trees[indice] for indice in index])
				_reference_tree_to_pos_tags = _generate_pos_tags_from_parse_trees(_max_length=self.args.max_reference_length, _padding_tag='')
			
				# 2022/05/18 23:59:19 特大坑!!! 参数名竟然是_BasicDataset__parse_trees而非自己写的__parse_trees, 还是通过help发现的, 详见博客https://blog.csdn.net/CY19980216/article/details/124789475
				# 2022/05/19 14:28:35 虽然可以不写参数名, 但是感觉这样似乎并不好, 我还是用_BasicDataset__parse_trees作为参数名相对能说的通
				# dataset_dataframe['reference_pos_tags'] = dataset_dataframe['reference_tree'].map(lambda _parse_trees_list: [_reference_tree_to_pos_tags(_BasicDataset__parse_trees=tree) for _parse_trees in _parse_trees_list])
				dataset_dataframe['reference_pos_tags'] = dataset_dataframe['reference_tree'].map(lambda _parse_trees_list: [_reference_tree_to_pos_tags(_parse_trees) for _parse_trees in _parse_trees_list])
				
			# 2022/04/06 00:18:52 若使用句法树, 则必然使用词性, 因此嵌套其中, 直接添加字段即可, 具体字段处理放在collate_fn中, 避免直接处理导致内存溢出
			if self.args.use_parse_tree:
				# 2022/05/06 22:17:34 增加新字段: 选项与题干的句法树, 这些字段已经是现成的了
				selected_columns.append('statement_tree')
				selected_columns.append('option_a_tree')
				selected_columns.append('option_b_tree')
				selected_columns.append('option_c_tree')
				selected_columns.append('option_d_tree')	
				
				# 2022/05/20 13:17:10 我后来想了如果将parse_tree_to_graph的运算放到collate_fn中显得很不经济(因为会在dataloader迭代时重复调用), 还是在生成dataset的时候直接处理完比较好
				_parse_trees_to_graph = lambda __parse_trees: [parse_tree_to_graph(parse_tree=__parse_tree, display=False, return_type='dgl', ignore_text=True) for __parse_tree in __parse_trees]
				
				logging.info('生成词性特征与句法树...')

				question_parse_tree_dataframe['statement_tree'] = question_parse_tree_dataframe['statement_tree'].map(_parse_trees_to_graph)
				question_parse_tree_dataframe['option_a_tree'] = question_parse_tree_dataframe['option_a_tree'].map(_parse_trees_to_graph)
				question_parse_tree_dataframe['option_b_tree'] = question_parse_tree_dataframe['option_b_tree'].map(_parse_trees_to_graph)
				question_parse_tree_dataframe['option_c_tree'] = question_parse_tree_dataframe['option_c_tree'].map(_parse_trees_to_graph)
				question_parse_tree_dataframe['option_d_tree'] = question_parse_tree_dataframe['option_d_tree'].map(_parse_trees_to_graph)
				
				dataset_dataframe = dataset_dataframe.drop(labels=['statement_tree', 'option_a_tree', 'option_b_tree', 'option_c_tree', 'option_d_tree'], axis=1)
				n_rows_before = dataset_dataframe.shape[0]
				dataset_dataframe = dataset_dataframe.merge(question_parse_tree_dataframe, on='id', how='left')
				n_rows_after = dataset_dataframe.shape[0]
				assert n_rows_before == n_rows_after, f'数据表连接不合法: 长度发生变化{n_rows_before}->{n_rows_after}'
				
				# 2022/04/29 10:27:48 参考文献部分的句法树和词性处理部分放在这里, 如果放在use_reference的分支里显得那个分支太冗长了
				if self.args.use_reference:
					# 2022/05/07 00:15:02 增加新字段: 参考文献的句法树
					selected_columns.append('reference_tree')
					
					# 2022/05/20 13:17:10 我后来想了如果将parse_tree_to_graph的运算放到collate_fn中显得很不经济(因为会在dataloader迭代时重复调用), 还是在生成dataset的时候直接处理完比较好
					reference_id2trees = {int(_id): _parse_trees_to_graph(_BasicDataset__parse_trees=eval(trees)) for _id, trees in zip(reference_parse_tree_dataframe['id'], reference_parse_tree_dataframe['content'])}
					dataset_dataframe['reference_tree'] = dataset_dataframe['reference_index_by_subject'].map(lambda index: [reference_id2trees[true_indice] for true_indice in filter(lambda x: not x == -1, index)])

		
		# 2022/04/05 23:53:52 训练数据集或验证数据集额外添加label_choice字段
		if self.mode.startswith('train') or self.mode.startswith('valid'):
			selected_columns.append('label_choice')
			dataset_dataframe['label_choice'] = dataset_dataframe['answer'].astype(int)
		
		# 2022/04/05 23:53:17 最终生成数据	
		self.data = dataset_dataframe[selected_columns].reset_index(drop=True)
	
	@timer
	def judgment_pipeline(self):
		"""
		20211121更新: 判断题形式的输入数据, 输出字段有:
		id				: 题目编号
		question		: 题目题干
		option			: 每个选项
		subject			: use_reference配置为True时生效, 包含num_top_subject个法律门类
		reference		: use_reference配置为True时生效, 包含相关的num_best个参考书目文档段落
		type			: 零一值表示概念题或情景题
		label_judgment	: train或valid模式时生效, 零一值表示判断题的答案
		option_id		: 20211216更新, 记录判断题对应的原选择题编号(ABCD)"""		
		self.choice_pipeline()
		self.data = self.choice_to_judgment(choice_dataframe=self.data, id_column='id', choice_column='options', answer_column='label_choice')
	
	@timer
	def display_pipeline(self):
		"""2022/02/23 14:04:47 
		用于展示数据的管道, 用于可视化的处理, 修改自choice_pipeline"""
		if self.mode.startswith('train'):
			filepaths = TRAINSET_PATHs[:]
		elif self.mode.startswith('valid'): 
			filepaths = VALIDSET_PATHs[:]
		elif self.mode.startswith('test'):
			filepaths = TESTSET_PATHs[:]
		else:
			assert False

		# 数据集字段预处理
		logging.info('预处理题目题干与选项...')
		start_time = time.time()
	
		# 合并概念题和情景题后的题库
		dataset_dataframe = pandas.concat([pandas.read_csv(filepath, sep='\t', header=0) for filepath in filepaths]).reset_index(drop=True)	
		
		if self.mode.endswith('_kd'):   
			dataset_dataframe = dataset_dataframe[dataset_dataframe['type'] == 0].reset_index(drop=True)	# 筛选概念题
		elif self.mode.endswith('_ca'): 
			dataset_dataframe = dataset_dataframe[dataset_dataframe['type'] == 1].reset_index(drop=True)	# 筛选情景分析题
		else:
			dataset_dataframe = dataset_dataframe.reset_index(drop=True)									# 无需筛选直接重索引
			
		# 2021/12/30 19:50:18 在调试情况下只选取少量数据运行以提高效率
		if self.for_debug:
			dataset_dataframe = dataset_dataframe.loc[:100, :]
			
		dataset_dataframe['id'] = dataset_dataframe['id'].astype(str)				# 字段id转为字符串
		dataset_dataframe['type'] = dataset_dataframe['type'].astype(int)			# 字段type转为整数
		dataset_dataframe['statement'] = dataset_dataframe['statement'].map(eval)	# 字段statement用eval函数转为分词列表
		dataset_dataframe['option_a'] = dataset_dataframe['option_a'].map(eval)		# 字段option_a用eval函数转为分词列表
		dataset_dataframe['option_b'] = dataset_dataframe['option_b'].map(eval)		# 字段option_b用eval函数转为分词列表
		dataset_dataframe['option_c'] = dataset_dataframe['option_c'].map(eval)		# 字段option_c用eval函数转为分词列表
		dataset_dataframe['option_d'] = dataset_dataframe['option_d'].map(eval)		# 字段option_d用eval函数转为分词列表
		
		# 参考文献相关字段预处理
		if self.args.use_reference:
			# 加载文档检索模型相关内容
			dictionary_path = GENSIM_RETRIEVAL_MODEL_SUMMARY[self.args.retrieval_model_name]['dictionary']
			if dictionary_path is None:		
				# logentropy模型的dictionary字段是None
				dictionary_path = REFERENCE_DICTIONARY_PATH
			dictionary = Dictionary.load(dictionary_path)
			similarity = self.grm.build_similarity(model_name=self.args.retrieval_model_name, dictionary=None, corpus=None, num_best=None)
			sequence = GensimRetrievalModel.load_sequence(model_name=self.args.retrieval_model_name, subject=None)
			
			reference_dataframe = pandas.read_csv(REFERENCE_PATH, sep='\t', header=0)
			index2subject = {index: LAW2SUBJECT.get(law, law) for index, law in enumerate(reference_dataframe['law'])}		# 记录reference_dataframe中每一行对应的法律门类
			
			# 新生成的几个字段说明:
			# query_result		: 形如[(4, 0.8), (7, 0.1), (1, 0.1)], 列表长度为args.num_best
			# reference_index	: 将[4, 7, 1]给抽取出来
			# reference			: 将[4, 7, 1]对应的参考书目文档的段落的分词列表给抽取出来并转为编号列表
			# subject			: 题目对应的args.num_top_subject个候选法律门类
			logging.info('生成查询得分向量...')
			dataset_dataframe['query_result'] = dataset_dataframe[['statement', 'option_a', 'option_b', 'option_c', 'option_d']].apply(self.generate_query_result(dictionary=dictionary, 
																																								  similarity=similarity, 
																																								  sequence=sequence), axis=1)
			dataset_dataframe['reference_index'] = dataset_dataframe['query_result'].map(lambda result: list(map(lambda x: x[0], result)))

			logging.info('填充subject字段的缺失值...')
			dataset_dataframe['subject'] = dataset_dataframe[['reference_index', 'subject']].apply(self.fill_subject(index2subject=index2subject), axis=1)

			# 2022/01/14 23:35:20 根据填充好的subject字段进行精确检索, 即只在num_top_subject个subject中检索
			subject_summary = {}
			for subject in SUBJECT2INDEX:
				summary = {}
				dictionary_path = GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['summary'][self.args.retrieval_model_name]['dictionary']
				if dictionary_path is None:		
					# logentropy模型的dictionary字段是None
					dictionary_path = GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['dictionary']
				dictionary = Dictionary.load(dictionary_path)
				corpus = MmCorpus(GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['summary'][self.args.retrieval_model_name]['corpus'])
				# 2022/01/15 22:26:41 关于num_best为什么是self.args.num_best_per_subject * self.args.num_top_subject而非self.args.num_best_per_subject
				# 2022/01/15 22:28:15 因为我考虑可能subject字段凑不满self.args.num_top_subject的个数, 所以可能需要用前面的来递补
				# 2022/01/15 22:28:07 因此num_best设定为最坏情况的值, 即只有一个候选subject, 此时需要用到num_best_per_subject * num_top_subject的数值, 而非num_best_per_subject
				# 2022/02/01 16:18:46 今天发现即便设置num_best, 也无法确保查询结果有num_best个, 甚至会少于num_best_per_subject个, 这使得函数retrieve_references_by_subject中的相关逻辑可能报错
				similarity = self.grm.build_similarity(model_name=self.args.retrieval_model_name, 
													   dictionary=dictionary,
													   corpus=corpus,
													   num_best=self.args.num_best_per_subject * self.args.num_top_subject)
				sequence = GensimRetrievalModel.load_sequence(model_name=self.args.retrieval_model_name, subject=subject)
				
				summary['reference_dataframe'] = reference_dataframe[reference_dataframe['law'] == SUBJECT2LAW.get(subject, subject)].reset_index(drop=True)
				summary['dictionary'] = Dictionary.load(GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['dictionary'])
				summary['corpus'] = corpus
				summary['similarity'] = similarity
				summary['sequence'] = sequence
				subject_summary[subject] = deepcopy(summary)
				del summary
			
			dataset_dataframe['reference'] = dataset_dataframe[['statement', 'option_a', 'option_b', 'option_c', 'option_d', 'subject']].apply(self.retrieve_references_by_subject(subject_summary=subject_summary), axis=1)	
			
			if self.mode.startswith('train') or self.mode.startswith('valid'):
				dataset_dataframe['label_choice'] = dataset_dataframe['answer'].astype(int)
				self.data = dataset_dataframe[['id', 'statement', 'option_a', 'option_b', 'option_c', 'option_d', 'subject', 'reference', 'reference_index', 'type', 'label_choice']].reset_index(drop=True)
			elif self.mode.startswith('test'):
				self.data = dataset_dataframe[['id', 'statement', 'option_a', 'option_b', 'option_c', 'option_d', 'subject', 'reference', 'reference_index', 'type']].reset_index(drop=True)		
		else:
			# 2022/02/03 10:50:54 生成最终的数据
			if self.mode.startswith('train') or self.mode.startswith('valid'):
				dataset_dataframe['label_choice'] = dataset_dataframe['answer'].astype(int)
				self.data = dataset_dataframe[['id', 'statement', 'option_a', 'option_b', 'option_c', 'option_d', 'type', 'label_choice']].reset_index(drop=True)
			elif self.mode.startswith('test'):
				self.data = dataset_dataframe[['id', 'statement', 'option_a', 'option_b', 'option_c', 'option_d', 'type']].reset_index(drop=True)

	def token_to_id(self, max_length, token2id):
		"""题目题干分词列表转编号"""
		def _token_to_id(_tokens):
			_ids = list(map(lambda _token: token2id.get(_token, token2id['UNK']), _tokens))
			if len(_ids) >= max_length:
				return _ids[: max_length]
			else:
				return _ids + [token2id['PAD']] * (max_length - len(_ids))
		return _token_to_id
		
	def token_to_vector(self, max_length, token2vector):
		"""引入词向量对分词进行编码"""
		pad_vector = numpy.zeros((self.vector_size, ))
		unk_vector = numpy.zeros((self.vector_size, ))
		def _token_to_vector(_tokens):
			_vectors = list(map(lambda _token: token2vector[_token] if _token in token2vector else unk_vector[:], _tokens))
			if len(_vectors) >= max_length:
				_vectors = _vectors[: max_length]
			else:
				_vectors = _vectors + [pad_vector[:] for _ in range(max_length - len(_vectors))]	# 注意这里复制列表就别用乘号了, 老老实实用循环免得出问题
			return numpy.stack(_vectors)															# 2021/12/27 23:00:44 修正为stack拼接
		return _token_to_vector
	
	def combine_option(self, max_length, token2emb, encode_as='id'):
		"""题目选项分词列表转编号并合并"""		
		if encode_as == 'id':
			def _combine_option(_dataframe):
				__token_to_id = self.token_to_id(max_length=max_length, token2id=token2emb)
				_option_a, _option_b, _option_c, _option_d = _dataframe
				return [__token_to_id(_option_a),
						__token_to_id(_option_b),
						__token_to_id(_option_c),
						__token_to_id(_option_d)]
		elif encode_as == 'vector':
			def _combine_option(_dataframe):
				__token_to_vector = self.token_to_vector(max_length=max_length, token2vector=token2emb)
				_option_a, _option_b, _option_c, _option_d = _dataframe
				return [__token_to_vector(_option_a),
						__token_to_vector(_option_b),
						__token_to_vector(_option_c),
						__token_to_vector(_option_d)]		
		else:
			raise NotImplementedError

		return _combine_option

	def generate_query_result(self, dictionary, similarity, sequence):
		"""生成查询得分向量"""
		def _generate_query_result(_dataframe):
			_statement, _option_a, _option_b, _option_c, _option_d = _dataframe						# 提取用于查询文档的关键词: 题干与四个选项
			_query_tokens = _statement + _option_a + _option_b + _option_c + _option_d				# 拼接题目和四个选项的分词
			if self.args.filter_stopword:											
				_query_tokens = filter_stopwords(tokens=_query_tokens, stopwords=self.stopwords)	# 筛除停用词
			return self.grm.query(query_tokens=_query_tokens, 
								  dictionary=dictionary, 
								  similarity=similarity, 
								  sequence=sequence)
		return _generate_query_result

	def find_reference_by_index(self, max_length, token2emb, reference_dataframe, encode_as='id'):
		"""根据新生成的reference_index寻找对应的参考段落, 并转为编号形式"""
		if encode_as == 'id':
			
			def _find_reference_by_index(_reference_index):
				_reference = []
				__token_to_id = self.token_to_id(max_length=max_length, token2id=token2emb)	
				for _index in _reference_index:
					_tokens = reference_dataframe.loc[_index, 'content']	# reference_index对应在reference_dataframe中的分词列表
					_reference.append(__token_to_id(_tokens))
				if len(_reference) < self.args.num_best:					# 2021/12/19 11:18:24 竟然Similarity可能返回的结果不足num_best, 也不是很能理解, 只能手动填补了
					for _ in range(self.args.num_best - len(_reference)):
						_reference.append([token2emb['UNK']] * max_length)	# 2021/12/19 11:25:16 填补为UNK
				return numpy.stack(_reference)								# 2021/12/27 22:07:18 要转为数组形式
				
		elif encode_as == 'vector':
			
			def _find_reference_by_index(_reference_index):
				_reference = []
				__token_to_vector = self.token_to_vector(max_length=max_length, token2vector=token2emb)	
				for _index in _reference_index:
					_tokens = reference_dataframe.loc[_index, 'content']				# reference_index对应在reference_dataframe中的分词列表
					_reference.append(__token_to_vector(_tokens))
				if len(_reference) < self.args.num_best:								# 2021/12/19 11:18:24 竟然Similarity可能返回的结果不足num_best, 也不是很能理解, 只能手动填补了
					for _ in range(self.args.num_best - len(_reference)):
						_reference.append(numpy.zeros((max_length, self.vector_size)))	# 2021/12/19 11:25:26 填补为零向量
				return numpy.stack(_reference)											# 2021/12/27 22:07:18 要转为数组形式
		else:
			raise NotImplementedError
		return _find_reference_by_index

	def fill_subject(self, index2subject):
		"""填充缺失的subject字段, 这里拟填充args.top_subject个候选subject"""
		def _fill_subject(_dataframe):
			_reference_index, _subject = _dataframe
			if _subject == _subject:														
				# 不缺失的情况无需填充, 直接填充到长度为args.top_subject即可
				# 注意这里都会多填充一位
				return [SUBJECT2INDEX[_subject] + 1] + [0] * (self.args.num_top_subject - 1)
			_candidate_subjects = [index2subject[_index] for _index in _reference_index]	# 根据reference_index生成候选的法律门类: index2subject记录了参考书目文档里每一行对应的法律门类
			_weighted_count = {}															# 记录每个候选的法律门类的加权累和数
			for _rank, _candidate_subject in enumerate(_candidate_subjects):
				if _candidate_subject in _weighted_count: 
					_weighted_count[_candidate_subject] += 1 / (_rank + 1)					# 这个加权的方式就是MRR
				else:
					_weighted_count[_candidate_subject] = 1 / (_rank + 1)
			_counter = Counter(_weighted_count).most_common(self.args.num_top_subject)		# 提取前args.top_subject个候选subject, 注意虽然是提取self.args.top_subject个, 但是可能会不足, 所以return的时候还需要填充
			_predicted_subjects = list(map(lambda x: x[0], _counter))						# 真实预测得到的至多args.top_subject给法律门类
			
			# 转为真实的法律门类字符串并填充到长度为args.top_subject
			# 注意这里索引加1是为将索引值等于0的情况视为subject缺失
			return [SUBJECT2INDEX[_candidate_subject] + 1 for _candidate_subject in _predicted_subjects] + [0] * (self.args.num_top_subject - len(_predicted_subjects))	
		return _fill_subject
	
	def retrieve_reference_index_by_subject(self, subject_summary):
		"""2022/03/23 14:04:01 
		根据填充后的subject字段进行新一轮的检索
		修改自`retrieve_references_by_subject`
		区别在于这里只返回index, 用于预先生成好的BERT输出结果上, 可以直接检索到嵌入结果
		用于padding的index表示为-1"""
		def _retrieve_reference_index_by_subject(_dataframe):
			# 2022/01/15 22:09:10 这里的_subject_index是形如[1, 18, 5]这样的列表
			# 2022/01/15 22:10:40 注意INDEX2SUBJECT中index的范围是0-17, 因此这里需要先减1
			_statement, _option_a, _option_b, _option_c, _option_d, _subject_index = _dataframe	# 提取用于查询文档的关键词: 题干与四个选项
			_subjects = list(map(lambda x: INDEX2SUBJECT.get(x - 1), _subject_index))
			_query_tokens = _statement + _option_a + _option_b + _option_c + _option_d
			# 2022/01/15 22:13:18 查询逻辑仿照generate_query_result函数
			if self.args.filter_stopword:											
				# 筛除停用词
				_query_tokens = filter_stopwords(tokens=_query_tokens, stopwords=self.stopwords)
			
			# 2022/01/15 22:42:06 生成查询检索
			_query_results = {}
			for _subject in _subjects:
				if _subject is not None:
					_query_result = self.grm.query(query_tokens=_query_tokens, 
												   dictionary=subject_summary[_subject]['dictionary'], 
												   similarity=subject_summary[_subject]['similarity'], 
												   sequence=subject_summary[_subject]['sequence'])
					_query_results[_subject] = _query_result
			
			# 2022/01/15 22:42:03 生成若干reference段落的分词列表
			_num_subjects = len(_query_results)
			_reference_index = []
			for _subject, _query_result in _query_results.items():
				# 2022/02/01 16:21:18 绝大多数情况下这是正确的(这是为什么设定num_best=self.args.num_best_per_subject * self.args.num_top_subject的原因)
				# 2022/02/01 16:23:55 但是会有例外, 即query的结果低于设定的参数num_best值, 因此注释掉下面这行断言, 改为警告, 并对下面内部循环次数作修正
				# assert len(_query_result) > self.args.num_best_per_subject
				if len(_query_result) > self.args.num_best_per_subject:
					warnings.warn('文档检索查询数量低于num_best')
				for _i in range(min(self.args.num_best_per_subject, len(_query_result))):
					# 2022/03/23 14:23:55 index指reset_index后保留的原index
					_reference_index.append(subject_summary[_subject]['reference_dataframe'].loc[_query_result[_i][0], 'index'])
			
			# 2022/01/15 22:41:58 可能数量不足(候选subject数量不足self.args.num_top_subject), 需要填补
			# 2022/01/15 22:42:45 目前填补策略就是从排名第一的subject(即_subjects[0])中继续选取检索结果递补
			# 2022/02/03 11:00:20 正如上面2022/02/01 16:23:55中所述查询结果可能少于num_best值, 因此这里可能数量还是不够, 尽管这种情况很少, 但是依然会出现
			for _i in range(self.args.num_best_per_subject * self.args.num_top_subject - len(_reference_index)):
				# 2022/02/13 16:57:56 关于gensim.similarites.Similarity的num_best参数已经有结论: 
				# 2022/02/13 16:57:56 如果是None则返回文档总数长度的向量, 该向量中的非零元数量是最大的num_best可能取值
				# 2022/02/13 16:57:56 如果是一个确定的数值, 则返回至多非零元数量的结果, 因此可能会不足num_best的长度
				# 2022/02/13 17:02:09 因此这里做一些条件分支处理来确保不会报错
				if self.args.num_best_per_subject + _i < len(_query_results[_subjects[0]]):
					# 2022/02/13 17:02:55 从排在第一位的subject里挑选参考文档段落
					# 2022/03/23 14:23:55 index指reset_index后保留的原index
					_reference_index.append(subject_summary[_subjects[0]]['reference_dataframe'].loc[_query_results[_subjects[0]][self.args.num_best_per_subject + _i][0], 'index'])
				else:
					# 2022/03/23 14:23:26 否则添加空文档(以-1表示)
					_reference_index.append(-1)
			return _reference_index
			
		return _retrieve_reference_index_by_subject

	def retrieve_references_by_subject(self, subject_summary):
		"""2022/01/15 19:42:42 
		根据填充后的subject字段进行新一轮的检索"""
		def _retrieve_references_by_subject(_dataframe):
			# 2022/01/15 22:09:10 这里的_subject_index是形如[1, 18, 5]这样的列表
			# 2022/01/15 22:10:40 注意INDEX2SUBJECT中index的范围是0-17, 因此这里需要先减1
			_statement, _option_a, _option_b, _option_c, _option_d, _subject_index = _dataframe	# 提取用于查询文档的关键词: 题干与四个选项
			_subjects = list(map(lambda x: INDEX2SUBJECT.get(x - 1), _subject_index))
			_query_tokens = _statement + _option_a + _option_b + _option_c + _option_d
			# 2022/01/15 22:13:18 查询逻辑仿照generate_query_result函数
			if self.args.filter_stopword:											
				# 筛除停用词
				_query_tokens = filter_stopwords(tokens=_query_tokens, stopwords=self.stopwords)
			
			# 2022/01/15 22:42:06 生成查询检索
			_query_results = {}
			for _subject in _subjects:
				if _subject is not None:
					_query_result = self.grm.query(query_tokens=_query_tokens, 
												   dictionary=subject_summary[_subject]['dictionary'], 
												   similarity=subject_summary[_subject]['similarity'], 
												   sequence=subject_summary[_subject]['sequence'])
					_query_results[_subject] = _query_result
			
			# 2022/01/15 22:42:03 生成若干reference段落的分词列表
			_num_subjects = len(_query_results)
			_reference_index = {_subject: [_x[0] for _x in _query_result] for _subject, _query_result in _query_results.items()}
			
			_references = []
			for _subject, _query_result in _query_results.items():
				# 2022/02/01 16:21:18 绝大多数情况下这是正确的(这是为什么设定num_best=self.args.num_best_per_subject * self.args.num_top_subject的原因)
				# 2022/02/01 16:23:55 但是会有例外, 即query的结果低于设定的参数num_best值, 因此注释掉下面这行断言, 改为警告, 并对下面内部循环次数作修正
				# assert len(_query_result) > self.args.num_best_per_subject
				if len(_query_result) > self.args.num_best_per_subject:
					warnings.warn('文档检索查询数量低于num_best')
				for _i in range(min(self.args.num_best_per_subject, len(_query_result))):
					_references.append(subject_summary[_subject]['reference_dataframe'].loc[_query_result[_i][0], 'content'])
			
			# 2022/01/15 22:41:58 可能数量不足(候选subject数量不足self.args.num_top_subject), 需要填补
			# 2022/01/15 22:42:45 目前填补策略就是从排名第一的subject(即_subjects[0])中继续选取检索结果递补
			# 2022/02/03 11:00:20 正如上面2022/02/01 16:23:55中所述查询结果可能少于num_best值, 因此这里可能数量还是不够
			for _i in range(self.args.num_best_per_subject * self.args.num_top_subject - len(_references)):
				# 2022/02/13 16:57:56 关于gensim.similarites.Similarity的num_best参数已经有结论, 如果是None则返回文档总数长度的向量, 该向量中的非零元数量是最大的num_best可能取值, 如果是一个确定的数值, 则返回至多非零元数量的结果, 因此可能会不足num_best的长度
				# 2022/02/13 17:02:09 因此这里做一些条件分支处理来确保不会报错
				if self.args.num_best_per_subject + _i < len(_query_results[_subjects[0]]):
					# 2022/02/13 17:02:55 从排在第一位的subject里挑选参考文档段落
					_references.append(subject_summary[_subjects[0]]['reference_dataframe'].loc[_query_results[_subjects[0]][self.args.num_best_per_subject + _i][0], 'content'])
				else:
					# 2022/02/13 17:03:43 否则添加空文档(空分词列表)
					_references.append([])
			return _references
			
		return _retrieve_references_by_subject

	def choice_to_judgment(self, choice_dataframe, id_column='id', choice_column='options', answer_column='label_choice'):
		"""
		20211124更新: 将选择题形式的dataframe转为判断题形式的dataframe
		2022/02/03 10:33:21 此处使用了dataframe的apply高阶用法, 使用result_type='expand'的模式来简化代码格式
		2022/02/03 10:33:31 注意
		:param choice_dataframe		: 选择题形式的dataframe
		:param id_column			: 题目编号所在的字段名, 用于表的连接
		:param choice_column		: 题目选项所在的字段名, 要求是一个长度为4的可迭代对象, 用于拆分
		:param answer_column		: 题目答案所在的字段名, 要求是0-15的编码值
		:return judgment_dataframe	: 判断题形式的dataframe
		"""
		# 左表是去除选项和答案的原数据表
		left_dataframe_columns = choice_dataframe.columns.tolist()
		left_dataframe_columns.remove(choice_column)
		if self.mode.startswith('train') or self.mode.startswith('valid'):
			left_dataframe_columns.remove(answer_column)
		left_dataframe = choice_dataframe[left_dataframe_columns]

		# 右表是由问题编号、单选项、判断真伪三个字段构成的数据表
		if self.mode.startswith('train') or self.mode.startswith('valid'):
			right_dataframe_columns = [id_column, 'option', 'label_judgment']
			right_dataframe = pandas.concat([choice_dataframe[[id_column]].apply(lambda x: [x[0]] * 4, axis=1, result_type='expand').stack().reset_index(drop=True),									# 通过expand将一道题的id扩充为四道题
											 choice_dataframe[[choice_column]].apply(lambda x: [x[0][0], x[0][1], x[0][2], x[0][3]], axis=1, result_type='expand').stack().reset_index(drop=True),		# 通过expand将一道题的四个选项扩充为四道题
											 choice_dataframe[[answer_column]].apply(lambda x: decode_answer(x[0], result_type=int), axis=1, result_type='expand').stack().reset_index(drop=True)], 	# 使用result_type为int的decode_answer方法获得每个选项的对错值, 并扩充成四道题
											 axis=1)
		else:
			right_dataframe_columns = [id_column, 'option']
			right_dataframe = pandas.concat([choice_dataframe[[id_column]].apply(lambda x: [x[0]] * 4, axis=1, result_type='expand').stack().reset_index(drop=True),									# 通过expand将一道题的id扩充为四道题
											 choice_dataframe[[choice_column]].apply(lambda x: [x[0][0], x[0][1], x[0][2], x[0][3]], axis=1, result_type='expand').stack().reset_index(drop=True)],		# 通过expand将一道题的四个选项扩充为四道题
											 axis=1)

		right_dataframe.columns = right_dataframe_columns
		right_dataframe['option_id'] = ['A', 'B', 'C', 'D'] * (right_dataframe.shape[0] // 4)	# 记录每个option的选项号ABCD
		
		judgment_dataframe = left_dataframe.merge(right_dataframe, how='left', on=id_column).reset_index(drop=True)
		return judgment_dataframe

	def __getitem__(self, item):
		return self.data.loc[item, :]

	def __len__(self):
		return len(self.data)


class ParseTreeDataset(BasicDataset):
	"""2022/09/08 18:21:48 
	使用句法树的数据管道
	之所以没有沿用BasicDataset, 原因是这个数据管道与上面有几个重要的区别:
	- 需要额外使用QUESTION_PARSE_TREE_PATH与REFERENCE_PARSE_TREE_PATH两个文件的数据
	- 不使用padding, 因为TreeRNNEncoder不需要分词序列等长, 而且也不希望分词序列会被max_options_length给切分掉
	- 仅支持词向量模式"""
	def __init__(self, 
				 args, 
				 mode='train', 
				 do_export=False, 
				 pipeline='choice', 
				 for_debug=False):
		"""参数说明:
		:param args			: DatasetConfig配置
		:param mode			: 数据集模式, 详见下面第一行的断言
		:param do_export	: 是否导出self.data
		:param pipeline		: 目前考虑judgment与choice两种模式
		:param for_debug	: 2021/12/30 19:21:46 调试模式, 只用少量数据集加快测试效率"""
		self.pipelines = {
			'choice'	: self.choice_pipeline,
		}
		assert mode in ['train', 'train_kd', 'train_ca', 'valid', 'valid_kd', 'valid_ca', 'test', 'test_kd', 'test_ca']
		assert pipeline in self.pipelines
		assert args.document_embedding is None	# 2022/09/08 18:15:46 使用的句法树仅支持使用词嵌入
		
		# 构造变量转为成员变量
		self.args = deepcopy(args)
		self.mode = mode
		self.do_export = do_export
		self.pipeline = pipeline
		self.for_debug = for_debug
		
		# 根据配置生成对应的成员变量
		if self.args.filter_stopword:
			self.stopwords = load_stopwords(stopword_names=None)
		
		if self.args.use_reference:
			# 2021/12/27 21:24:39 使用参考书目文档必须调用检索模型: 目前只有gensim模块下的检索模型
			_args = load_args(Config=RetrievalModelConfig)
			# 2021/12/27 21:24:29 重置新参数值, 因为传入Dataset类的args参数中的一些值可能与默认值不同, 以传入值为准
			for key in vars(_args):
				if key in self.args:
					_args.__setattr__(key, self.args.__getattribute__(key))
			self.grm = GensimRetrievalModel(args=_args)		
		
		if self.args.word_embedding in GENSIM_EMBEDDING_MODEL_SUMMARY:
			# 2021/12/27 21:24:43 使用gensim模块中的词向量模型或文档向量模型
			_args = load_args(Config=EmbeddingModelConfig)
			# 2021/12/27 21:24:24 重置新参数值, 因为传入Dataset类的args参数中的一些值可能与默认值不同, 以传入值为准
			for key in vars(_args):
				if key in self.args:
					_args.__setattr__(key, self.args.__getattribute__(key))
			self.gem = GensimEmbeddingModel(args=_args)
		elif self.args.word_embedding is not None:
			# 2022/09/08 19:59:04 None的情况不需要编写逻辑, 为了省事就这样写条件分支了
			raise Exception(f'Unknown word_embedding: {self.args.word_embedding}')
		
		# 生成数据表
		self.pipelines[pipeline]()
		
		# 导出数据表
		if self.do_export:
			logging.info('导出数据表...')
			self.data.to_csv(COMPLETE_REFERENCE_PATH, sep='\t', header=True, index=False)
 
	@timer
	def choice_pipeline(self):
		"""2022/09/19 13:03:56 
		最终输出字段: 
		[id                     1_6055
		type                    0
		question               	[3119, 2268, 674, 13, 1361, 8, 676, 45, 13, 24...
		option_a				[811, 14976, 674, 6088, 5334, 654, 19633, 674,...
		option_b       		 	[674, 7988, 4619, 13, 149, 1026, 4594, 5, 2258...
		option_c       			[2426, 2258, 2635, 277, 1640, 1362, 1278, 7990...
		option_d        		[674, 676, 8, 676, 13, 24023, 811, 1371, 2172]
		question_parse_tree    	[(ROOT (FRAG (DNP (NP (NP (ADJP (JJ 下列)) (DNP ...
		option_a_parse_tree    	[(ROOT (IP (VP (PP (P 由) (NP (NN 每届) (NN 全国人民代...
		option_b_parse_tree    	[(ROOT (NP (CP (IP (NP (NN 全国人民代表大会) (NN 任期)) ...
		option_c_parse_tree    	[(ROOT (IP (VP (ADVP (CS 如果)) (VP (VV 全国人民代表大会...
		option_d_parse_tree    	[(ROOT (IP (NP (DNP (NP (NP (NR 全国人民代表大会)) (NP...
		label_choice            15
		Name: 5, dtype: object]
		若use_reference为True, 则输出字段额外增加'reference', 'reference_parse_tree', 'subject', 'reference_index_by_subject'
		"""
		if self.mode.startswith('train'):
			filepaths = TRAINSET_PATHs[:]
		elif self.mode.startswith('valid'):  # 20211101新增验证集处理逻辑
			filepaths = VALIDSET_PATHs[:]
		elif self.mode.startswith('test'):
			filepaths = TESTSET_PATHs[:]
		else:
			assert False
		max_option_length = self.args.max_option_length
		max_statement_length = self.args.max_statement_length
		max_reference_length = self.args.max_reference_length

		# 2022/09/09 09:37:19 初始的各个字段, 与BasicDataset的不同
		selected_columns = ['id', 'type', 'question_vector', 'option_a_vector', 'option_b_vector', 'option_c_vector', 'option_d_vector']

		# 数据集字段预处理
		logging.info('预处理题目题干与选项...')
		start_time = time.time()
		
		# 合并概念题和情景题后的题库
		dataset_dataframe = pandas.concat([pandas.read_csv(filepath, sep='\t', header=0) for filepath in filepaths]).reset_index(drop=True)	
		
		if self.mode.endswith('_kd'):   
			dataset_dataframe = dataset_dataframe[dataset_dataframe['type'] == 0].reset_index(drop=True)	# 筛选概念题
		elif self.mode.endswith('_ca'): 
			dataset_dataframe = dataset_dataframe[dataset_dataframe['type'] == 1].reset_index(drop=True)	# 筛选情景分析题
		else:
			dataset_dataframe = dataset_dataframe.reset_index(drop=True)									# 无需筛选直接重索引
			
		# 2021/12/30 19:50:18 在调试情况下只选取少量数据运行以提高效率
		if self.for_debug:
			dataset_dataframe = dataset_dataframe.loc[:10, :]
			
		dataset_dataframe['id'] = dataset_dataframe['id'].astype(str)				# 字段id转为字符串
		dataset_dataframe['type'] = dataset_dataframe['type'].astype(int)			# 字段type转为整数
		dataset_dataframe['statement'] = dataset_dataframe['statement'].map(eval)	# 字段statement用eval函数转为分词列表
		dataset_dataframe['option_a'] = dataset_dataframe['option_a'].map(eval)		# 字段option_a用eval函数转为分词列表
		dataset_dataframe['option_b'] = dataset_dataframe['option_b'].map(eval)		# 字段option_b用eval函数转为分词列表
		dataset_dataframe['option_c'] = dataset_dataframe['option_c'].map(eval)		# 字段option_c用eval函数转为分词列表
		dataset_dataframe['option_d'] = dataset_dataframe['option_d'].map(eval)		# 字段option_d用eval函数转为分词列表
		
		# 2022/05/17 13:09:01 过滤特殊字符: \u3000与单空格
		if self.args.filter_stopword:
			# 2022/05/20 12:29:27 STANFORD_IGNORED_SYMBOL中目前只有\u3000与单空格(即' ')
			_filter_ignored_symbol = partial(filter_stopwords, stopwords=STANFORD_IGNORED_SYMBOL)
			dataset_dataframe['statement'] = dataset_dataframe['statement'].map(_filter_ignored_symbol)	# 字段statement去除停用词
			dataset_dataframe['option_a'] = dataset_dataframe['option_a'].map(_filter_ignored_symbol)	# 字段option_a去除停用词
			dataset_dataframe['option_b'] = dataset_dataframe['option_b'].map(_filter_ignored_symbol)	# 字段option_b去除停用词
			dataset_dataframe['option_c'] = dataset_dataframe['option_c'].map(_filter_ignored_symbol)	# 字段option_c去除停用词
			dataset_dataframe['option_d'] = dataset_dataframe['option_d'].map(_filter_ignored_symbol)	# 字段option_d去除停用词	

		if self.args.word_embedding is None:
			# 不使用任何嵌入, 即使用顺序编码值(token2id)进行词嵌入
			
			# token2id字典: 20211212后决定以参考书目文档的token2id为标准, 而非题库的token2id
			token2id_dataframe = pandas.read_csv(REFERENCE_TOKEN2ID_PATH, sep='\t', header=0)
			token2id = {token: _id for token, _id in zip(token2id_dataframe['token'], token2id_dataframe['id'])}			
			
			# 题目题干的分词列表转为编号列表
			dataset_dataframe['question_vector'] = dataset_dataframe['statement'].map(self.token_to_id(token2id=token2id))
			dataset_dataframe['option_a_vector'] = dataset_dataframe['option_a'].map(self.token_to_id(token2id=token2id))
			dataset_dataframe['option_b_vector'] = dataset_dataframe['option_b'].map(self.token_to_id(token2id=token2id))
			dataset_dataframe['option_c_vector'] = dataset_dataframe['option_c'].map(self.token_to_id(token2id=token2id))
			dataset_dataframe['option_d_vector'] = dataset_dataframe['option_d'].map(self.token_to_id(token2id=token2id))
																			
		elif self.args.word_embedding in GENSIM_EMBEDDING_MODEL_SUMMARY:
			# 使用gensim词向量模型进行训练: word2vec, fasttext
			embedding_model_class = eval(GENSIM_EMBEDDING_MODEL_SUMMARY[self.args.word_embedding]['class'])
			embedding_model_path = GENSIM_EMBEDDING_MODEL_SUMMARY[self.args.word_embedding]['model']
			embedding_model = embedding_model_class.load(embedding_model_path)
			token2vector = embedding_model.wv
			self.vector_size = embedding_model.wv.vector_size
			
			# 2021/12/27 21:26:45 这里可以直接删除模型, 直接用wv即可
			del embedding_model
															# 题目题干的分词列表转为编号列表
			dataset_dataframe['question_vector'] = dataset_dataframe['statement'].map(self.token_to_vector(token2vector=token2vector))
			dataset_dataframe['option_a_vector'] = dataset_dataframe['option_a'].map(self.token_to_vector(token2vector=token2vector))
			dataset_dataframe['option_b_vector'] = dataset_dataframe['option_b'].map(self.token_to_vector(token2vector=token2vector))
			dataset_dataframe['option_c_vector'] = dataset_dataframe['option_c'].map(self.token_to_vector(token2vector=token2vector))
			dataset_dataframe['option_d_vector'] = dataset_dataframe['option_d'].map(self.token_to_vector(token2vector=token2vector))	
		else:
			raise Exception(f'Unknown word_embedding: {self.args.word_embedding}')

		# 读取题库句法树文件
		logging.info('读取题库句法树文件...')
		selected_columns.extend(['question_parse_tree', 
								 'option_a_parse_tree', 
								 'option_b_parse_tree', 
								 'option_c_parse_tree', 
								 'option_d_parse_tree'])
		question_parse_tree_dataframe = pandas.read_csv(QUESTION_PARSE_TREE_PATH, sep='\t', header=0)
		question_id2parse_tree = {
			question_parse_tree_dataframe.loc[i, 'id']: {
				'statement'	: eval(question_parse_tree_dataframe.loc[i, 'statement']),
				'option_a'	: eval(question_parse_tree_dataframe.loc[i, 'option_a']),
				'option_b'	: eval(question_parse_tree_dataframe.loc[i, 'option_b']),
				'option_c'	: eval(question_parse_tree_dataframe.loc[i, 'option_c']),
				'option_d'	: eval(question_parse_tree_dataframe.loc[i, 'option_d']),
			} for i in range(question_parse_tree_dataframe.shape[0])
		}
		del question_parse_tree_dataframe	# 生成完整索引后可以释放内存
		
		dataset_dataframe['question_parse_tree'] = dataset_dataframe['id'].map(lambda x: question_id2parse_tree[x]['statement'])
		dataset_dataframe['option_a_parse_tree'] = dataset_dataframe['id'].map(lambda x: question_id2parse_tree[x]['option_a'])
		dataset_dataframe['option_b_parse_tree'] = dataset_dataframe['id'].map(lambda x: question_id2parse_tree[x]['option_b'])
		dataset_dataframe['option_c_parse_tree'] = dataset_dataframe['id'].map(lambda x: question_id2parse_tree[x]['option_c'])
		dataset_dataframe['option_d_parse_tree'] = dataset_dataframe['id'].map(lambda x: question_id2parse_tree[x]['option_d'])
		
		# 2022/09/19 16:12:44 在数据加载时顺带遍历句法树, 省得每个epoch都要重新遍历一次
		selected_columns.extend(['question_parse_tree_traverse', 
								 'option_a_parse_tree_traverse', 
								 'option_b_parse_tree_traverse', 
								 'option_c_parse_tree_traverse', 
								 'option_d_parse_tree_traverse'])
		_traverse_parse_tree_list = lambda _parse_tree_list: [traverse_parse_tree(parse_tree=_parse_tree) for _parse_tree in _parse_tree_list]
		dataset_dataframe['question_parse_tree_traverse'] = dataset_dataframe['question_parse_tree'].map(_traverse_parse_tree_list)
		dataset_dataframe['option_a_parse_tree_traverse'] = dataset_dataframe['question_parse_tree'].map(_traverse_parse_tree_list)
		dataset_dataframe['option_b_parse_tree_traverse'] = dataset_dataframe['question_parse_tree'].map(_traverse_parse_tree_list)
		dataset_dataframe['option_c_parse_tree_traverse'] = dataset_dataframe['question_parse_tree'].map(_traverse_parse_tree_list)
		dataset_dataframe['option_d_parse_tree_traverse'] = dataset_dataframe['question_parse_tree'].map(_traverse_parse_tree_list)

		# 2022/04/05 23:54:24 参考文献相关字段预处理: 额外添加subject与reference字段
		if self.args.use_reference:
			# 2022/05/20 13:19:58 reference_index_by_subject并没有实际用到, 仅用于辅助调试
			selected_columns.extend(['reference', 'reference_parse_tree', 'subject', 'reference_index_by_subject'])
			
			# 加载文档检索模型相关内容
			dictionary_path = GENSIM_RETRIEVAL_MODEL_SUMMARY[self.args.retrieval_model_name]['dictionary']
			if dictionary_path is None:		
				# logentropy模型的dictionary字段是None, 使用默认值REFERENCE_DICTIONARY_PATH
				dictionary_path = REFERENCE_DICTIONARY_PATH
			dictionary = Dictionary.load(dictionary_path)
			similarity = self.grm.build_similarity(model_name=self.args.retrieval_model_name, dictionary=None, corpus=None, num_best=None)
			sequence = GensimRetrievalModel.load_sequence(model_name=self.args.retrieval_model_name, subject=None)
			
			reference_dataframe = pandas.read_csv(REFERENCE_PATH, sep='\t', header=0)
			reference_dataframe['content'] = reference_dataframe['content'].map(eval)
			index2subject = {index: LAW2SUBJECT.get(law, law) for index, law in enumerate(reference_dataframe['law'])}		# 记录reference_dataframe中每一行对应的法律门类
			
			# 新生成的几个字段说明:
			# query_result		: 形如[(4, 0.8), (7, 0.1), (1, 0.1)], 列表长度为args.num_best
			# reference_index	: 将[4, 7, 1]给抽取出来
			# reference			: 将[4, 7, 1]对应的参考书目文档的段落的分词列表给抽取出来并转为编号列表
			# subject			: 题目对应的args.num_top_subject个候选法律门类
			logging.info('生成查询得分向量...')
			dataset_dataframe['query_result'] = dataset_dataframe[['statement', 'option_a', 'option_b', 'option_c', 'option_d']].apply(self.generate_query_result(dictionary=dictionary, 
																																								  similarity=similarity, 
																																								  sequence=sequence), axis=1)
			
			# 2022/04/15 23:33:26 这个reference_index是第一遍全文检索的index, 目前已经被弃用, 我们会在特定的几个subject中检索index
			dataset_dataframe['reference_index'] = dataset_dataframe['query_result'].map(lambda result: list(map(lambda x: x[0], result)))
			
			logging.info('填充subject字段的缺失值...')
			dataset_dataframe['subject'] = dataset_dataframe[['reference_index', 'subject']].apply(self.fill_subject(index2subject=index2subject), axis=1)
			
			# 2022/01/14 23:35:20 根据填充好的subject字段进行精确检索, 即只在num_top_subject个subject中检索
			subject_summary = {}
			for subject in SUBJECT2INDEX:
				summary = {}
				dictionary_path = GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['summary'][self.args.retrieval_model_name]['dictionary']
				if dictionary_path is None:		
					# logentropy模型的dictionary字段是None
					dictionary_path = GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['dictionary']
				dictionary = Dictionary.load(dictionary_path)
				corpus = MmCorpus(GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['summary'][self.args.retrieval_model_name]['corpus'])
				# 2022/01/15 22:26:41 关于num_best为什么是self.args.num_best_per_subject * self.args.num_top_subject而非self.args.num_best_per_subject
				# 2022/01/15 22:28:15 因为我考虑可能subject字段凑不满self.args.num_top_subject的个数, 所以可能需要用前面的来递补
				# 2022/01/15 22:28:07 因此num_best设定为最坏情况的值, 即只有一个候选subject, 此时需要用到num_best_per_subject * num_top_subject的数值, 而非num_best_per_subject
				# 2022/02/01 16:18:46 今天发现即便设置num_best, 也无法确保查询结果有num_best个, 甚至会少于num_best_per_subject个, 这使得函数retrieve_references_by_subject中的相关逻辑可能报错
				similarity = self.grm.build_similarity(model_name=self.args.retrieval_model_name, 
													   dictionary=dictionary,
													   corpus=corpus,
													   num_best=self.args.num_best_per_subject * self.args.num_top_subject)
				sequence = GensimRetrievalModel.load_sequence(model_name=self.args.retrieval_model_name, subject=subject)
				
				# 2022/03/23 14:16:53 改为在reset_index时不丢弃原先的index
				# 2022/03/23 14:16:53 原因是这里得到的是分门类的reference, 但是预先生成的参考书目文档的BERT输出是按照原先reference的顺序排列的, 因此之前的index是需要保留的
				# summary['reference_dataframe'] = reference_dataframe[reference_dataframe['law'] == SUBJECT2LAW.get(subject, subject)].reset_index(drop=True)
				summary['reference_dataframe'] = reference_dataframe[reference_dataframe['law'] == SUBJECT2LAW.get(subject, subject)].reset_index(drop=False)
				summary['dictionary'] = Dictionary.load(GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['dictionary'])
				summary['corpus'] = corpus
				summary['similarity'] = similarity
				summary['sequence'] = sequence
				subject_summary[subject] = deepcopy(summary)
				del summary
			
			# 2022/04/15 23:38:09 先把按subject检索的index找出来: 这里的index就是原先reference的索引, 取值范围0-24717
			dataset_dataframe['reference_index_by_subject'] = dataset_dataframe[['statement', 'option_a', 'option_b', 'option_c', 'option_d', 'subject']].apply(self.retrieve_reference_index_by_subject(subject_summary=subject_summary), axis=1)
			if self.args.word_embedding is None:
				references = dataset_dataframe[['statement', 'option_a', 'option_b', 'option_c', 'option_d', 'subject']].apply(self.retrieve_references_by_subject(subject_summary=subject_summary), axis=1)
				_token_to_id = self.token_to_id(token2id=token2id)
				dataset_dataframe['reference'] = references.map(lambda _references: [_token_to_id(_reference_tokens) for _reference_tokens in _references])
	
			elif self.args.word_embedding in GENSIM_EMBEDDING_MODEL_SUMMARY:
				references = dataset_dataframe[['statement', 'option_a', 'option_b', 'option_c', 'option_d', 'subject']].apply(self.retrieve_references_by_subject(subject_summary=subject_summary), axis=1)
				_token_to_vector = self.token_to_vector(token2vector=token2vector)
				dataset_dataframe['reference'] = references.map(lambda _references: [_token_to_vector(_reference_tokens) for _reference_tokens in _references])
			else:
				raise NotImplementedError
			
			# 2022/09/09 20:11:23 匹配对应的参考文档句法树
			selected_columns.extend(['reference_parse_tree', 'reference_parse_tree_traverse'])
			reference_parse_tree_dataframe = pandas.read_csv(REFERENCE_PARSE_TREE_PATH, sep='\t')
			reference_parse_tree_dataframe = reference_parse_tree_dataframe[['id', 'content']]	
			reference_id2trees = {int(_id): eval(trees) for _id, trees in zip(reference_parse_tree_dataframe['id'], reference_parse_tree_dataframe['content'])}
			dataset_dataframe['reference_parse_tree'] = dataset_dataframe['reference_index_by_subject'].map(lambda index: [[] if indice == -1 else reference_id2trees[indice] for indice in index])
			
			# 2022/09/19 16:15:06 遍历参考文档句法树	
			dataset_dataframe['reference_parse_tree_traverse'] = dataset_dataframe['reference_index_by_subject'].map(lambda index: [(None, None) if indice == -1 else _traverse_parse_tree_list(_parse_tree_list=reference_id2trees[indice]) for indice in index])
			
		# 2022/04/05 23:53:52 训练数据集或验证数据集额外添加label_choice字段
		if self.mode.startswith('train') or self.mode.startswith('valid'):
			selected_columns.append('label_choice')
			dataset_dataframe['label_choice'] = dataset_dataframe['answer'].astype(int)
		
		# 2022/04/05 23:53:17 最终生成数据	
		self.data = dataset_dataframe[selected_columns].reset_index(drop=True)
	

	def token_to_id(self, token2id):
		"""题目题干分词列表转编号: 与BasicDataset的不同之处在于不做padding也不作去尾"""
		return lambda _tokens: list(map(lambda _token: token2id.get(_token, token2id['UNK']), _tokens))
		
	def token_to_vector(self, token2vector):
		"""引入词向量对分词进行编码: 与BasicDataset的不同之处在于不做padding也不作去尾"""
		unk_vector = numpy.zeros((self.vector_size, ))
		return lambda _tokens: numpy.stack(list(map(lambda _token: token2vector[_token] if _token in token2vector else unk_vector[:], _tokens)))
