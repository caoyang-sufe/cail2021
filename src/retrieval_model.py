# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
# 用于检索题目对应文档的模型

if __name__ == '__main__':
	import sys
	sys.path.append('../')

import time
import numpy
import jieba
import pandas
import gensim
import pickle
import warnings

from copy import deepcopy
from gensim.corpora import MmCorpus, Dictionary
from gensim.similarities import Similarity

from setting import *

from src.data_tools import load_stopwords, filter_stopwords
from src.utils import timer

class GensimRetrievalModel:
	"""gensim模块下的文档检索模型"""
	def __init__(self, args, **kwargs):
		"""
		:param args	: EmbeddingModelConfig配置
		"""
		self.args = deepcopy(args)
		
		if self.args.filter_stopword:
			self.stopwords = load_stopwords(stopword_names=None)
	
	@timer
	def build_reference_corpus(self,
							   reference_path=REFERENCE_PATH,
							   dictionary_export_path=REFERENCE_DICTIONARY_PATH, 
							   corpus_export_path=REFERENCE_CORPUS_PATH,
							   document_export_path=REFERENCE_DOCUMENT_PATH,
							   subject=None):
		"""
		构建参考书目语料(corpus), 在gensim模块下指词频矩阵与字典
		:param reference_path			: 预处理得到的参考书目CSV文件
		:param dictionary_export_path	: gensim字典导出路径
		:param corpus_export_path		: gensim语料导出路径
		:param stopwords				: 停用词列表
		:param subject					: 2022/01/14 21:34:59 只用一个subject的参考书目
		
		:return corpus					: gensim模块下的语料, 即为分词词频矩阵
		:return dictionary				: gensim模块下的字典, 即为分词索引
		"""													
		reference_dataframe = pandas.read_csv(reference_path, sep='\t', header=0, dtype=str)	# 读取处理后的参考书目CSV文件
		reference_dataframe = reference_dataframe.fillna('')									# 参考书目的section字段存在缺失, 可使用空字符串填充
		
		if subject is not None:
			# 筛选某一法律门类的参考文档
			reference_dataframe = reference_dataframe[reference_dataframe['law'] == SUBJECT2LAW.get(subject, subject)].reset_index(drop=True)
		
		# 将参考书目文档中的每一行段落的分词列表都存入document中
		document = []
		for i in range(reference_dataframe.shape[0]):
			section = reference_dataframe.loc[i, 'section']
			content = eval(reference_dataframe.loc[i, 'content'])
			paragraph = jieba.lcut(section) + content 						# 要把小节名称作为文档段落内容: 因为数据预处理时把小节名称从文档段落中分离出来了
			if self.args.filter_stopword:									# 过滤停用词一定程度上可以提升模型性能
				paragraph = filter_stopwords(tokens=paragraph, stopwords=self.stopwords)
			document.append(paragraph)										# 向文档终添加段落
		dictionary = Dictionary(document)									# 生成字典: 分词索引
		corpus = [dictionary.doc2bow(paragraph) for paragraph in document]	# 生成语料: 分词词频矩阵
		if dictionary is not None:
			dictionary.save(dictionary_export_path)							# 保存生成的字典
		if corpus_export_path is not None:
			MmCorpus.serialize(corpus_export_path, corpus)					# 保存生成的语料
		if document_export_path is not None:
			pickle.dump(document, open(document_export_path, 'wb'))
		return corpus, dictionary, document


	@timer
	def build_tfidf_model(self, 
						  corpus_import_path=REFERENCE_CORPUS_PATH,
						  model_export_path=REFERENCE_TFIDF_MODEL_PATH,
						  corpus_export_path=REFERENCE_CORPUS_TFIDF_PATH):
		"""构建TFIDF模型并利用TFIDF模型生成新的TFIDF指数权重矩阵"""
		kwargs = {
			'dictionary': None,
			'normalize'	: True,
			'smartirs'	: self.args.smartirs_tfidf,
			'pivot'		: self.args.pivot_tfidf,
			'slope'		: self.args.slope_tfidf,
		}
		return GensimRetrievalModel.easy_build_model(model_name='tfidf',
													corpus_import_path=corpus_import_path,
													model_export_path=model_export_path,
													corpus_export_path=corpus_export_path,
													**kwargs)
	
	@timer
	def build_lsi_model(self, 
						corpus_import_path=REFERENCE_CORPUS_TFIDF_PATH,
						model_export_path=REFERENCE_LSI_MODEL_PATH,
						corpus_export_path=REFERENCE_CORPUS_LSI_PATH):
		"""构建LSI模型并利用LSI模型生成新的LSI指数权重矩阵"""

		kwargs = {
			'num_topics'	: self.args.num_topics_lsi,
			'chunksize'		: 20000,
			'decay'			: self.args.decay_lsi,						# 旧样本的衰减
			'distributed'	: False,									# 分布式计算
			'onepass'		: True,										# 设为False则使用每次传递多个的随机算法
			'power_iters'	: self.args.power_iters_lsi,				# 提升power_iters会增加模型精确性, 但是性能会下降
			'extra_samples'	: self.args.extra_samples_lsi,				# 除秩k外还可使用的样本数量, 可以提升精确性
			'dtype'			: numpy.float64,
		}
		return GensimRetrievalModel.easy_build_model(model_name='lsi',
													corpus_import_path=corpus_import_path,
													model_export_path=model_export_path,
													corpus_export_path=corpus_export_path,
													**kwargs)
	
	@timer
	def build_lda_model(self, 
						corpus_import_path=REFERENCE_CORPUS_TFIDF_PATH,
						model_export_path=REFERENCE_LDA_MODEL_PATH,
						corpus_export_path=REFERENCE_CORPUS_LDA_PATH):
		"""构建LDA模型并利用LDA模型生成新的LDA指数权重矩阵"""
		kwargs = {
			'num_topics'			: self.args.num_topics_lda,
			'distributed'			: False,							# 分布式计算
			'chunksize'				: 2000,
			'passes'				: 1,								# 训练时每次传递的样本, 可能类似批训练的batchsize
			'update_every'			: 1,								# Number of documents to be iterated through for each update. Set to 0 for batch learning, > 1 for online iterative learning.
			'alpha'					: 'symmetric',						# topics的优先级, 默认symmetric是等权, asymmetric是降序, 也可以传入num_topics长度的优先级向量
			'eta'					: None,								# 类似alpha的一个参数
			'decay'					: self.args.decay_lda,				# 范围是0.5-1.0
			'offset'				: 1.,
			'eval_every'			: 10,
			'iterations'			: self.args.iterations_lda,			# 最大迭代次数
			'gamma_threshold'		: self.args.gamma_threshold_lda,	# gamma参数的最小变化值, 类似某种步长
			'minimum_probability'	: self.args.minimum_probability_lda,# 小于该数值的topic会被过滤掉
			'random_state'			: None,
			'ns_conf'				: None,								# 只有在distributed为True时生效
			'minimum_phi_value'		: .01,								# 只有在per_word_topics为True时生效
			'per_word_topics'		: False,							# If True, the model also computes a list of topics, sorted in descending order of most likely topics for each word, along with their phi values multiplied by the feature length (i.e. word count).
			'callbacks'				: None,								# 评估指标
			'dtype'					: numpy.float32
		}
		return GensimRetrievalModel.easy_build_model(model_name='lda',
													corpus_import_path=corpus_import_path,
													model_export_path=model_export_path,
													corpus_export_path=corpus_export_path,
													**kwargs)
	
	@timer
	def build_hdp_model(self, 
						corpus_import_path=REFERENCE_CORPUS_TFIDF_PATH,
						model_export_path=REFERENCE_HDP_MODEL_PATH,
						corpus_export_path=REFERENCE_CORPUS_HDP_PATH):
		"""构建HDP模型并利用HDP模型生成新的HDP指数权重矩阵"""
		kwargs = {
			'max_chunks'	: None,
			'max_time'		: None,
			'chunksize'		: 256,
			'kappa'			: self.args.kappa_hdp,						# Learning parameter which acts as exponential decay factor to influence extent of learning from each batch.
			'tau'			: self.args.tau_hdp,						# Learning parameter which down-weights early iterations of documents.
			'K'				: self.args.K_hdp,							# Second level truncation level
			'T'				: self.args.T_hdp,							# Top level truncation level
			'alpha'			: 1,										# Second level concentration, 这个参数是跟topic优先度有关的, 默认平权
			'gamma'			: 1,										# First level concentration, 这个参数也是跟topic优先度有关的, 默认平权
			'eta'			: .01,										# The topic Dirichlet
			'scale'			: 1.,										# Weights information from the mini-chunk of corpus to calculate rhot.
			'var_converge'	: .0001,									# Lower bound on the right side of convergence. Used when updating variational parameters for a single document.
			'outputdir'		: None,
			'random_state'	: None,
		}
		return GensimRetrievalModel.easy_build_model(model_name='hdp',
													corpus_import_path=corpus_import_path,
													model_export_path=model_export_path,
													corpus_export_path=corpus_export_path,
													**kwargs)
	
	@timer
	def build_logentropy_model(self,
							   corpus_import_path=REFERENCE_CORPUS_TFIDF_PATH, 
							   model_export_path=REFERENCE_LOGENTROPY_MODEL_PATH,
							   corpus_export_path=REFERENCE_CORPUS_LOGENTROPY_PATH):
		"""构建LogEntropy模型并利用ogEntropy模型生成新的LDA指数权重矩阵"""
		kwargs = {
			'normalize': True,
		}
		return GensimRetrievalModel.easy_build_model(model_name='logentropy',
													corpus_import_path=corpus_import_path,
													model_export_path=model_export_path,
													corpus_export_path=corpus_export_path,
													**kwargs)

	@classmethod
	def validate_corpus_import_path(cls, corpus_import_path, model_name):
		"""检查模型构建函数的参数corpus_import_path的合法性"""
		sequence_length = len(GENSIM_RETRIEVAL_MODEL_SUMMARY[model_name]['sequence'])
		if sequence_length == 1:
			if not corpus_import_path == REFERENCE_CORPUS_PATH:
				warnings.warn('参数corpus_import_path不合规')
		elif sequence_length == 2:
			if not corpus_import_path == GENSIM_RETRIEVAL_MODEL_SUMMARY[GENSIM_RETRIEVAL_MODEL_SUMMARY[model_name]['sequence'][0]]['corpus']:
				warnings.warn('参数corpus_import_path不合规')
		else:
			raise NotImplementedError('目前不考虑模型序列长度超过2的情形')
	
	@classmethod
	def easy_build_model(cls,
						 model_name,
						 corpus_import_path,
						 model_export_path=None,
						 corpus_export_path=None,
						 *args, **kwargs):
		"""
		20211210更新: 统一的模型构建方法
		:param model_name	: 模型名称
		:param corpus_import_path		: gensim语料导入路径
		:param dictionary_import_path	: gensim字典导入路径
		:param model_export_path		: 模型保存路径
		"""
		GensimRetrievalModel.validate_corpus_import_path(corpus_import_path=corpus_import_path, model_name=model_name)
		corpus = MmCorpus(corpus_import_path)
		
		# 20211212更新: 意外发现LogEntropy模型的构造参数里没有id2word, 因此将setting.py中GENSIM_MODEL_SUMMARY里对应的dictionary字段修正为None
		dictionary_import_path = GENSIM_RETRIEVAL_MODEL_SUMMARY[model_name]['dictionary']
		if GENSIM_RETRIEVAL_MODEL_SUMMARY[model_name]['dictionary'] is None:
			model = eval(GENSIM_RETRIEVAL_MODEL_SUMMARY[model_name]['class'])(corpus, **kwargs)	
		else:
			dictionary = Dictionary.load(dictionary_import_path)
			model = eval(GENSIM_RETRIEVAL_MODEL_SUMMARY[model_name]['class'])(corpus, id2word=dictionary, **kwargs)			
		corpus = [model[doc] for doc in corpus]																# 转换为模型指数矩阵语料
		if model_export_path is not None:
			model.save(model_export_path)																	# 保存模型
		if corpus_export_path is not None:
			MmCorpus.serialize(corpus_export_path, corpus)													# 保存模型生成的新语料
		return model, corpus

	@classmethod
	def load_sequence(cls, model_name, subject=None):
		"""加载模型序列"""
		# 2022/01/14 23:59:36 添加subject字段以迎合分subject的检索模型
		sequence = []	
		summary = GENSIM_RETRIEVAL_MODEL_SUMMARY.copy() if subject is None else GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['summary']
		for _model_name in summary[model_name]['sequence']:
			load_function = eval(summary[model_name]['class']).load
			model_path = summary[model_name]['model']
			model = load_function(model_path)
			sequence.append(model)
		return sequence

	@timer
	def build_similarity(self, model_name, dictionary=None, corpus=None, num_best=None):
		"""构建模型的gensim相似度(Similarity)"""
		# 2022/01/14 23:48:56 之前是只给GENSIM_RETRIEVAL_MODEL_SUMMARY下的模型生成相似度, 现在改进为可以为分subject的检索模型生成相似度, 因而新添加几个参数
		if dictionary is None:
			dictionary_import_path = GENSIM_RETRIEVAL_MODEL_SUMMARY[model_name]['dictionary']
			if dictionary_import_path is None:
				dictionary_import_path = REFERENCE_DICTIONARY_PATH
			dictionary = Dictionary.load(dictionary_import_path)
		if corpus is None:
			corpus_import_path = GENSIM_RETRIEVAL_MODEL_SUMMARY[model_name]['corpus']
			corpus = MmCorpus(corpus_import_path)
		if num_best is None:
			num_best = self.args.num_best
		
		temp_similarity_saved_path = os.path.join(TEMP_SIMILARITY_DIR, f'gensim_similarity_{model_name}_{time.strftime("%Y%m%d%H%M%S")}')
		similarity = Similarity(temp_similarity_saved_path, corpus, num_features=len(dictionary), num_best=num_best)
		return similarity

	def query(self, query_tokens, dictionary, similarity, sequence):
		"""
		给定查询分词列表返回相似度匹配向量
		:param query_tokens	: 需要查询的关键词分词列表
		:param dictionary	: gensim字典
		:param similarity	: gensim相似度
		:param sequence		: 模型序列
		:return result		: 文档中每个段落的匹配分值
		"""
		if self.args.filter_stopword:
			filter_stopwords(tokens=query_tokens, stopwords=self.stopwords)
		query_corpus = dictionary.doc2bow(query_tokens)
		for model in sequence:
			query_corpus = model[query_corpus]
		result = similarity[query_corpus]
		return result
		

class NeuralRetrieveModel:
	"""基于神经网络模型的文档检索"""
	pass

