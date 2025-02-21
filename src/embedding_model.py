# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
# 词嵌入模型

if __name__ == '__main__':
	import sys
	sys.path.append('../')

import os
import gc
import time
import json
import dill
import torch
import gensim
import pandas
import pickle
import logging


from copy import deepcopy
from gensim.corpora import MmCorpus, Dictionary
from gensim.models import Word2Vec, FastText, Doc2Vec, WordEmbeddingSimilarityIndex
from gensim.models.doc2vec import TaggedDocument
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix

from setting import *

from transformers import BertTokenizer, BertModel

from src.data_tools import load_stopwords, filter_stopwords
from src.utils import timer

class GensimEmbeddingModel:
	"""gensim模块下的词嵌入模型"""
	def __init__(self, args):
		"""
		:param args	: EmbeddingModelConfig配置
		"""
		self.args = deepcopy(args)
		
		if self.args.filter_stopword:
			self.stopwords = load_stopwords(stopword_names=None)
	
	@timer
	def build_word2vec_model(self, 
							 corpus_import_path=REFERENCE_CORPUS_PATH,
							 document_import_path=REFERENCE_DOCUMENT_PATH,
							 model_export_path=REFERENCE_WORD2VEC_MODEL_PATH):
		kwargs = {
			'size'		: self.args.size_word2vec,
			'min_count'	: self.args.min_count_word2vec,
			'window'	: self.args.window_word2vec,
			'workers'	: self.args.workers_word2vec,
		}
		return GensimEmbeddingModel.easy_build_model(model_name='word2vec',
													 corpus_import_path=corpus_import_path,
													 document_import_path=document_import_path,
													 model_export_path=model_export_path,
													 **kwargs)
	@timer
	def build_fasttext_model(self, 
							 corpus_import_path=REFERENCE_CORPUS_PATH,
							 document_import_path=REFERENCE_DOCUMENT_PATH,
							 model_export_path=REFERENCE_WORD2VEC_MODEL_PATH):
		kwargs = {
			'size'		: self.args.size_fasttext,
			'min_count'	: self.args.min_count_fasttext,
			'window'	: self.args.window_fasttext,
			'workers'	: self.args.workers_fasttext,
		}
		return GensimEmbeddingModel.easy_build_model(model_name='fasttext',
													 corpus_import_path=corpus_import_path,
													 document_import_path=document_import_path,
													 model_export_path=model_export_path,
													 **kwargs)
	
	@classmethod
	def easy_build_model(cls,
						 model_name,
						 corpus_import_path,
						 document_import_path,
						 model_export_path,
						 **kwargs):
		"""
		20211218更新: 
		最近发现用corpus_file参数训练得到的模型词汇表全是索引而非分词
		而且观察下来跟dictionary的索引还对不上, 非常的恼火, 只能改用sentences参数的写法了
		"""
		# model = eval(GENSIM_EMBEDDING_MODEL_SUMMARY[model_name]['class'])(corpus_file=corpus_import_path, **kwargs)
		model = eval(GENSIM_EMBEDDING_MODEL_SUMMARY[model_name]['class'])(sentences=pickle.load(open(document_import_path, 'rb')), **kwargs)
		if model_export_path is not None:
			model.save(model_export_path)
		return model
	
	@timer
	def build_similarity(self, model_name):
		"""构建模型的gensim相似度(Similarity)"""
		model = eval(GENSIM_EMBEDDING_MODEL_SUMMARY[model_name]['class']).load(GENSIM_EMBEDDING_MODEL_SUMMARY[model_name]['model'])
		dictionary = Dictionary.load(REFERENCE_DICTIONARY_PATH)
		corpus = MmCorpus(REFERENCE_CORPUS_PATH)
		similarity_index = WordEmbeddingSimilarityIndex(model.wv)
		similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary)
		similarity = SoftCosineSimilarity(corpus, similarity_matrix, num_best=self.args.num_best)
		return similarity

	def query(self, query_tokens, dictionary, similarity):
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
		result = similarity[query_corpus]
		return result

	@timer
	def build_doc2vec_model(self, 
							corpus_import_path=REFERENCE_CORPUS_PATH, 
							document_import_path=REFERENCE_DOCUMENT_PATH, 
							model_export_path=REFERENCE_DOC2VEC_MODEL_PATH):
		"""2021/12/27 14:21:10 构建Doc2Vec模型"""		
		kwargs = {
			'vector_size': self.args.size_doc2vec,
			'min_count': self.args.min_count_doc2vec,
			'window': self.args.window_doc2vec,
			'workers': self.args.workers_doc2vec,
		}
		from gensim.test.utils import common_texts
		documents = pickle.load(open(document_import_path, 'rb'))
		tagged_documents = [TaggedDocument(document, [tag]) for tag, document in enumerate(documents)]
		model = Doc2Vec(documents=tagged_documents, **kwargs)
		# model = Doc2Vec(corpus_file=corpus_import_path, **kwargs)
		if model_export_path is not None:
			model.save(model_export_path)
		return model

	
class TransformersEmbeddingModel:
	"""transformers模块下的词嵌入模型"""
	def __init__(self, args):
		"""
		:param args	: EmbeddingModelConfig配置
		"""
		self.args = deepcopy(args)
	
	@classmethod
	@timer
	def load_bert_model(cls, model_name='bert-base-chinese'):
		assert model_name in BERT_MODEL_SUMMARY		
		tokenizer = BertTokenizer(BERT_MODEL_SUMMARY[model_name]['vocab'])
		model = BertModel.from_pretrained(BERT_MODEL_SUMMARY[model_name]['root'])
		model.eval()
		return tokenizer, model
	
	def load_bert_config(self, model_name='bert-base-chinese'):
		assert model_name in BERT_MODEL_SUMMARY		
		config = json.load(open(BERT_MODEL_SUMMARY[model_name]['config'], 'r', encoding='utf8'))
		config['hidden_size'] = self.args.bert_hidden_size
		return config
	
	def generate_bert_output(self, text, tokenizer, model, max_length=512):
		encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
		output = model(**encoded_input)
		return output.get(self.args.bert_output)
		
	def build_reference_bert_output(self, 
									export_path=REFERENCE_POOLER_OUTPUT_PATH, 
									batch_save_path=REFERENCE_POOLER_OUTPUT_DIR, 
									batch_size=None):
		"""2022/03/19 16:01:51
		生成参考书目文档的BERT输出结果, 以dill文件形式保存到本地
		:param export_path		: 最终的dill文件存储路径
		:param batch_save_path	: 若`batch_size`不为None, 则生效, 会按batch_size的规模分批存放在`batch_save_path`下, 命名规则为`batch_size`.dill
		:param batch_size		: 如果是None, 则全部生成后进行一次性存储到外部文件, 否则每隔`batch_size`次存储一次输出结果(防止OOM)
		"""
		# 读取预处理后的参考书目文档
		reference_dataframe = pandas.read_csv(REFERENCE_PATH, sep='\t', header=0, dtype=str)
		reference_dataframe = reference_dataframe.fillna('')								
		
		# 加载BERT模型
		tokenizer, model = TransformersEmbeddingModel.load_bert_model(model_name='bert-base-chinese')
		
		# 生成BERT输出并存储
		bert_outputs = []
		for i in range(reference_dataframe.shape[0]):
			logging.info(i)
			content_tokens = eval(reference_dataframe.loc[i, 'content'])
			content = ''.join(content_tokens)
			# 可能会因为分句过长而无法解析的情况, 报错原因通常是OOM, 采取异常抛出
			try:
				bert_output = self.generate_bert_output(text=content, tokenizer=tokenizer, model=model, max_length=512)
			except Exception as exception:
				bert_output = exception
			logging.info(type(bert_output))
			bert_outputs.append(bert_output)
			
			# 临时存储, 防止过大
			if batch_size is not None and (i + 1) % batch_size == 0:
				save_path = os.path.join(batch_save_path, f'{i + 1}.dill')
				with open(save_path, 'wb') as f:
					dill.dump(bert_outputs, f)
					logging.info(f'Save to {save_path}')
				bert_outputs = []										# 重置bert_outputs
				gc.collect()
		
		# 最后一次存储
		save_path = export_path if batch_size is None else os.path.join(batch_save_path, f'{i + 1}.dill')
		with open(save_path, 'wb') as f:
			dill.dump(bert_outputs, f)
			logging.info(f'Final output is saved to {save_path}')
		gc.collect()
			
		
	def build_question_bert_output(self, 
								   export_path=QUESTION_POOLER_OUTPUT_PATH, 
								   batch_save_path=QUESTION_POOLER_OUTPUT_DIR, 
								   batch_size=None):
		"""202022/03/20 11:05:1822/03/19 16:01:51
		生成参考书目文档的BERT输出结果, 以dill文件形式保存到本地
		:param export_path		: 最终的dill文件存储路径
		:param batch_save_path	: 若`batch_size`不为None, 则生效, 会按batch_size的规模分批存放在`batch_save_path`下, 命名规则为`batch_size`.dill
		:param batch_size		: 如果是None, 则全部生成后进行一次性存储到外部文件, 否则每隔`batch_size`次存储一次输出结果(防止OOM)
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
		
		# 加载BERT模型
		tokenizer, model = TransformersEmbeddingModel.load_bert_model(model_name='bert-base-chinese')
		
		bert_outputs = {column: [] for column in selected_columns}		# 以DataFrame的数据格式进行存储
		for i in range(question_dataframe.shape[0]):
			logging.info(i)
			_id, statement_tokens, option_a_tokens, option_b_tokens, option_c_tokens, option_d_tokens, source = question_dataframe.loc[i, selected_columns]
			statement, option_a, option_b, option_c, option_d = list(map(lambda tokens: ''.join(tokens), [statement_tokens, option_a_tokens, option_b_tokens, option_c_tokens, option_d_tokens]))
			
			bert_outputs['id'].append(_id)
			bert_outputs['source'].append(source)
			for text, field_name in zip([statement, option_a, option_b, option_c, option_d], selected_columns):
				if field_name in ['id', 'source']: 
					continue
				try:
					bert_output = self.generate_bert_output(text=text, tokenizer=tokenizer, model=model, max_length=512)
				except Exception as exception:
					bert_output = exception
				logging.info(f'{_id} - {field_name} - {type(bert_output)}')
				bert_outputs[field_name].append(bert_output)

			# 临时存储, 防止过大
			if batch_size is not None and (i + 1) % batch_size == 0:
				save_path = os.path.join(batch_save_path, f'{i + 1}.dill')
				with open(save_path, 'wb') as f:
					dill.dump(bert_outputs, f)
					logging.info(f'Save to {save_path}')
				bert_outputs = {column: [] for column in selected_columns}	# 重置bert_outputs
				gc.collect()
		
		# 最后一次存储
		save_path = export_path if batch_size is None else os.path.join(batch_save_path, f'{i + 1}.dill')
		with open(save_path, 'wb') as f:
			dill.dump(bert_outputs, f)
			logging.info(f'Final output is saved to {save_path}')
		gc.collect()

	@classmethod
	def combine_batch_dill_files(cls, 
								 category='reference', 
								 export_path=REFERENCE_POOLER_OUTPUT_PATH, 
								 batch_save_path=REFERENCE_POOLER_OUTPUT_DIR):
		"""
		合并由`build_reference_bert_output`与`build_question_bert_output`生成的批量dill文件
		"""
		assert category in ['reference', 'question']
		
		filenames = os.listdir(batch_save_path)
		batch_counts = list(map(lambda filename: int(filename.split('.')[0]), filenames))
		sorted_batch_counts = sorted(batch_counts)
		
		if category == 'reference':
			bert_outputs = []
			for batch_count in sorted_batch_counts:
				print(batch_count)
				with open(os.path.join(batch_save_path, f'{batch_count}.dill'), 'rb') as f:
					bert_output = dill.load(f)	
				bert_outputs.extend(bert_output)		

		elif category == 'question':
			with open(os.path.join(batch_save_path, filenames[0]), 'rb') as f:
				bert_output = dill.load(f)
			bert_outputs = {column: [] for column in bert_output}
			for batch_count in sorted_batch_counts:
				print(batch_count)
				with open(os.path.join(batch_save_path, f'{batch_count}.dill'), 'rb') as f:
					bert_output = dill.load(f)		
				for column in bert_outputs:
					bert_outputs[column].extend(bert_output[column])					
		else:
			raise Exception(f'Unknown param `category`: {category}')
		
		with open(export_path, 'wb') as f:
			dill.dump(bert_outputs, f)
		
