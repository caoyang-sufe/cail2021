# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# 数据预处理

import os
import re
import time
import gensim

from setting import *
from config import BaseConfig, RetrievalModelConfig, EmbeddingModelConfig, GraphConfig
from src.data_tools import json_to_csv, split_validset, token2frequency_to_csv, token2id_to_csv, reference_to_csv, load_stopwords, filter_stopwords, load_userdict_jieba, generate_userdict
from src.retrieval_model import GensimRetrievalModel
from src.embedding_model import GensimEmbeddingModel, TransformersEmbeddingModel
from src.graph import Graph
from src.graph_tools import generate_parse_tree
from src.utils import initialize_logger, load_args, save_args, timer

# 新建所有文件夹
@timer
def makedirs():
	os.makedirs(NEWDATA_DIR, exist_ok=True)
	os.makedirs(LOGGING_DIR, exist_ok=True)
	os.makedirs(TEMP_DIR, exist_ok=True)
	os.makedirs(MODEL_DIR, exist_ok=True)
	os.makedirs(CHECKPOINT_DIR, exist_ok=True)
	os.makedirs(RETRIEVAL_MODEL_DIR, exist_ok=True)
	os.makedirs(GENSIM_RETRIEVAL_MODEL_DIR, exist_ok=True)
	os.makedirs(GENSIM_RETRIEVAL_MODEL_SUBJECT_DIR, exist_ok=True)
	os.makedirs(EMBEDDING_MODEL_DIR, exist_ok=True)
	os.makedirs(GENSIM_EMBEDDING_MODEL_DIR, exist_ok=True)
	os.makedirs(TEMP_SIMILARITY_DIR, exist_ok=True)						# 2022/02/18 11:01:29 存放gensim文档检索模型相似度临时文件的目录
	
	os.makedirs(BERT_OUTPUT_DIR, exist_ok=True)							# 2022/03/20 10:39:02 存放BERT模型输出的文件夹
	
	os.makedirs(REFERENCE_BERT_OUTPUT_DIR, exist_ok=True)				# 2022/03/22 19:48:40 存放BERT模型输出子文件的临时目录(参考书目)
	os.makedirs(QUESTION_BERT_OUTPUT_DIR, exist_ok=True)				# 2022/03/22 19:48:40 存放BERT模型输出子文件的临时目录(题库)
	
	os.makedirs(REFERENCE_POOLER_OUTPUT_DIR, exist_ok=True)				# 2022/03/24 20:48:35 存放BERT模型输出子文件的临时目录pooler_output(参考书目)
	os.makedirs(QUESTION_POOLER_OUTPUT_DIR, exist_ok=True)				# 2022/03/24 20:48:35 存放BERT模型输出子文件的临时目录pooler_output(题库)
	os.makedirs(REFERENCE_LAST_HIDDEN_STATE_DIR, exist_ok=True)			# 2022/03/24 20:48:35 存放BERT模型输出子文件的临时目录last_hidden_state(参考书目)
	os.makedirs(QUESTION_LAST_HIDDEN_STATE_DIR, exist_ok=True)			# 2022/03/24 20:48:35 存放BERT模型输出子文件的临时目录last_hidden_state(题库)
	
	
	for subject in SUBJECT2INDEX:
		os.makedirs(os.path.join(GENSIM_RETRIEVAL_MODEL_SUBJECT_DIR, subject), exist_ok=True)

# 题库训练集与测试集的预处理
@timer
def preprocess_trainsets_and_testsets():
	token2frequency = {}
	for raw_trainset_path, trainset_path, validset_path in zip(RAW_TRAINSET_PATHs, TRAINSET_PATHs, VALIDSET_PATHs):
		dataframe, token2frequency = json_to_csv(json_path=raw_trainset_path,
												 csv_path=None,
												 token2frequency=token2frequency,
												 mode='train')
		split_validset(dataframe, train_export_path=trainset_path, valid_export_path=validset_path)

	for raw_testset_path, new_testset_path in zip(RAW_TESTSET_PATHs, TESTSET_PATHs):
		_, token2frequency = json_to_csv(json_path=raw_testset_path,
										 csv_path=new_testset_path,
										 token2frequency=token2frequency,
										 mode='test')

	token2frequency_to_csv(export_path=TOKEN2FREQUENCY_PATH, token2frequency=token2frequency)
	token2id_to_csv(export_path=TOKEN2ID_PATH, token2frequency=token2frequency)

# 参考书目的预处理
@timer
def preprocess_reference_book():
	# 参考书目的预处理
	_, token2frequency = reference_to_csv(export_path=REFERENCE_PATH)
	token2frequency_to_csv(export_path=REFERENCE_TOKEN2FREQUENCY_PATH, token2frequency=token2frequency)
	token2id_to_csv(export_path=REFERENCE_TOKEN2ID_PATH, token2frequency=token2frequency)

# gensim文档检索模型预构建
# 2022/01/14 22:53:10 添加build_by_subject参数, 指是否为每一个法律门类构建模型
@timer
def build_gensim_retrieval_models(args=None, model_names=None, update_reference_corpus=True, build_by_subject=True):
	if args is None:
		args = load_args(Config=RetrievalModelConfig)

	if model_names is None:
		model_names = list(GENSIM_RETRIEVAL_MODEL_SUMMARY.keys())
			
	
	
	if update_reference_corpus:
		grm = GensimRetrievalModel(args=args)
		grm.build_reference_corpus(reference_path=REFERENCE_PATH, 
								   dictionary_export_path=REFERENCE_DICTIONARY_PATH, 
								   corpus_export_path=REFERENCE_CORPUS_PATH,
								   document_export_path=REFERENCE_DOCUMENT_PATH,
								   subject=None)
		
		# 2022/01/14 22:58:08 给每一个法律门类创建文档, 语料, 字典
		if build_by_subject:
			for subject in SUBJECT2INDEX:
				grm.build_reference_corpus(reference_path=REFERENCE_PATH,
										   dictionary_export_path=GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['dictionary'], 
										   corpus_export_path=GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['corpus'],
										   document_export_path=GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['document'],
										   subject=subject)				
		

	if 'tfidf' in model_names:
		# 20211214更新: 默认参数是(None, None, .25), 测试下来这一组参数的hit@3精确率有87.8%
		# ('atu', 1., .5)参数的hit@3能到91.6%
		# ('atu', .5, .5)参数的hit@3还是87.8%
		# ('ann', None, .25)参数的hit@3还是91.7%
		# 详细调参结果见文件夹temp/tfidf调参/下的结果
		# 2022/01/12 13:27:07 使用法律领域的字典后, 最高的hit@3还是('ann', None, .25), 达到了92.1%
		args.smartirs = 'ann'
		args.pivot = None
		args.slope = .25
		grm = GensimRetrievalModel(args=args)
		grm.build_tfidf_model(corpus_import_path=REFERENCE_CORPUS_PATH,
							  model_export_path=REFERENCE_TFIDF_MODEL_PATH,
							  corpus_export_path=REFERENCE_CORPUS_TFIDF_PATH)
							  
		# 2022/01/14 22:58:08 给每一个法律门类创建检索模型
		if build_by_subject:
			for subject in SUBJECT2INDEX:
				grm.build_tfidf_model(corpus_import_path=GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['corpus'],
									  model_export_path=GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['summary']['tfidf']['model'],
									  corpus_export_path=GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['summary']['tfidf']['corpus'])	
		
	if 'lsi' in model_names:
		args.num_topics_lsi = 256
		args.power_iters_lsi = 3
		args.extra_samples_lsi = 256
		grm = GensimRetrievalModel(args=args)
		grm.build_lsi_model(corpus_import_path=REFERENCE_CORPUS_TFIDF_PATH, 
							model_export_path=REFERENCE_LSI_MODEL_PATH,
							corpus_export_path=REFERENCE_CORPUS_LSI_PATH)
		
		# 2022/01/14 22:58:08 给每一个法律门类创建检索模型
		if build_by_subject:
			for subject in SUBJECT2INDEX:
				grm.build_lsi_model(corpus_import_path=GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['summary']['tfidf']['corpus'],
									model_export_path=GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['summary']['lsi']['model'],
									corpus_export_path=GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['summary']['lsi']['corpus'])
							
	if 'lda' in model_names:
		args.num_topics_lsi = 256
		args.decay_lda = 1.
		args.iterations_lda = 500
		args.gamma_threshold_lda = .0001
		args.minimum_probability_lda = 0.
		grm = GensimRetrievalModel(args=args)
		grm.build_lda_model(corpus_import_path=REFERENCE_CORPUS_TFIDF_PATH, 
							model_export_path=REFERENCE_LDA_MODEL_PATH,
							corpus_export_path=REFERENCE_CORPUS_LDA_PATH)

		# 2022/01/14 22:58:08 给每一个法律门类创建检索模型
		if build_by_subject:
			for subject in SUBJECT2INDEX:
				grm.build_lda_model(corpus_import_path=GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['summary']['tfidf']['corpus'],
									model_export_path=GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['summary']['lda']['model'],
									corpus_export_path=GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['summary']['lda']['corpus'])


	if 'hdp' in model_names:
		args.kappa_hdp = 0.8
		args.tau_hdp = 32.
		args.K_hdp = 16
		args.T_hdp = 256
		grm = GensimRetrievalModel(args=args)
		grm.build_hdp_model(corpus_import_path=REFERENCE_CORPUS_PATH,
							model_export_path=REFERENCE_HDP_MODEL_PATH,
							corpus_export_path=REFERENCE_CORPUS_HDP_PATH)

		# 2022/01/14 22:58:08 给每一个法律门类创建检索模型
		if build_by_subject:
			for subject in SUBJECT2INDEX:
				grm.build_hdp_model(corpus_import_path=GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['corpus'],
									model_export_path=GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['summary']['hdp']['model'],
									corpus_export_path=GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['summary']['hdp']['corpus'])
							
	if 'logentropy' in model_names:			
		grm = GensimRetrievalModel(args=args)	
		grm.build_logentropy_model(corpus_import_path=REFERENCE_CORPUS_PATH, 
								   model_export_path=REFERENCE_LOGENTROPY_MODEL_PATH,
								   corpus_export_path=REFERENCE_CORPUS_LOGENTROPY_PATH)

		# 2022/01/14 22:58:08 给每一个法律门类创建检索模型
		if build_by_subject:
			for subject in SUBJECT2INDEX:
				grm.build_logentropy_model(corpus_import_path=GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['corpus'],
										   model_export_path=GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['summary']['logentropy']['model'],
										   corpus_export_path=GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY[subject]['summary']['logentropy']['corpus'])
	
	save_args(args=args, save_path=os.path.join(TEMP_DIR, 'RetrievalModelConfig.json'))

# gensim词嵌入模型预构建
@timer
def build_gensim_embedding_models(args=None, model_names=None):
	if args is None:
		args = load_args(Config=EmbeddingModelConfig)
		args.size_word2vec = 256
		args.min_count_word2vec = 1
		args.window_word2vec = 5
		args.workers_word2vec = 16
		
		args.size_fasttext = 256			 
		args.min_count_fasttext = 1	
		args.window_fasttext = 5
		args.workers_fasttext = 16	
		
		args.size_doc2vec = 512			 
		args.min_count_doc2vec = 1	
		args.window_doc2vec = 5
		args.workers_doc2vec = 16	
		
	if model_names is None:
		model_names = list(GENSIM_EMBEDDING_MODEL_SUMMARY.keys())

	gem = GensimEmbeddingModel(args=args)
	
	if 'word2vec' in model_names:
		gem.build_word2vec_model(corpus_import_path=REFERENCE_CORPUS_PATH, 
								 document_import_path=REFERENCE_DOCUMENT_PATH,
								 model_export_path=REFERENCE_WORD2VEC_MODEL_PATH)
	if 'fasttext' in model_names:
		gem.build_fasttext_model(corpus_import_path=REFERENCE_CORPUS_PATH, 
								 document_import_path=REFERENCE_DOCUMENT_PATH,
								 model_export_path=REFERENCE_FASTTEXT_MODEL_PATH)
	if 'doc2vec' in model_names:
		gem.build_doc2vec_model(corpus_import_path=REFERENCE_CORPUS_PATH, 
								document_import_path=REFERENCE_DOCUMENT_PATH,
								model_export_path=REFERENCE_DOC2VEC_MODEL_PATH)
	
	save_args(args=args, save_path=os.path.join(TEMP_DIR, 'EmbeddingModelConfig.json'))

# 2022/03/19 11:10:47 简易预处理: 必须进行的预处理步骤, 通常几分钟内可以完成
def easy_preprocess():
	args = load_args(Config=BaseConfig)
	
	# 2022/02/18 13:08:38 目前服务器上跑出的模型都没有使用
	# 2022/02/19 09:41:32 开始使用
	args.use_userdict = True	
	
	# 2022/03/10 15:52:15 生成自定义的用户字典
	generate_userdict(export_path=USERDICT_PATHs['diy'])

	if args.use_userdict:
		load_userdict_jieba()
	
	makedirs()
	preprocess_trainsets_and_testsets()
	preprocess_reference_book()
	build_gensim_retrieval_models(args=None, model_names=[args.retrieval_model_name], update_reference_corpus=True, build_by_subject=True)
	build_gensim_embedding_models(args=None, model_names=['word2vec', 'fasttext', 'doc2vec'])

# 2022/03/19 11:11:23 高阶预处理: 耗时较长的预处理
def advance_preprocess():
	# # 生成句法解析树: 参考书目文档需要两天, 题库需要近三天
	# args = load_args(Config=GraphConfig)
	# graph = Graph(args=args)
	# graph.build_reference_parse_tree(export_path=REFERENCE_PARSE_TREE_PATH)
	# graph.build_question_parse_tree(export_path=QUESTION_PARSE_TREE_PATH)
	
	# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
	
	# 生成依存关系图
	args = load_args(Config=GraphConfig)
	graph = Graph(args=args)
	graph.build_reference_dependency(export_path=REFERENCE_DEPENDENCY_PATH)
	# graph.build_question_dependency(export_path=QUESTION_DEPENDENCY_PATH)
	
	# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
	
	# # 生成BERT模型输出结果
	# args = load_args(Config=EmbeddingModelConfig)
	
	# args.bert_output = 'pooler_output'
	# tem = TransformersEmbeddingModel(args=args)
	# tem.build_reference_bert_output(export_path=REFERENCE_POOLER_OUTPUT_PATH, 
									# batch_save_path=REFERENCE_POOLER_OUTPUT_DIR,
									# batch_size=100)
									
	# tem.build_question_bert_output(export_path=QUESTION_POOLER_OUTPUT_PATH, 
								   # batch_save_path=QUESTION_POOLER_OUTPUT_DIR,
								   # batch_size=25)
								   
	# args.bert_output = 'last_hidden_state'
	# tem = TransformersEmbeddingModel(args=args)
	# tem.build_reference_bert_output(export_path=REFERENCE_LAST_HIDDEN_STATE_PATH, 
									# batch_save_path=REFERENCE_LAST_HIDDEN_STATE_DIR,
									# batch_size=25)
									
	# tem.build_question_bert_output(export_path=QUESTION_LAST_HIDDEN_STATE_PATH, 
								   # batch_save_path=QUESTION_LAST_HIDDEN_STATE_DIR,
								   # batch_size=5)
								   
	# # 合并上一步生成的批量dill文件
	# TransformersEmbeddingModel.combine_batch_dill_files(category='reference', export_path=REFERENCE_BERT_OUTPUT_PATH, batch_save_path=REFERENCE_BERT_OUTPUT_DIR)
	# TransformersEmbeddingModel.combine_batch_dill_files(category='question', export_path=QUESTION_BERT_OUTPUT_PATH, batch_save_path=QUESTION_BERT_OUTPUT_DIR)

# 个别处理的
def special_prerprocess():
	# 参考书目文档第19594行(索引为19592)解析错误
	# 原分词序列为['我国', '现行', '由', '全国人民代表大会常务委员会', '审议', '通过', '并', '颁布', '的', '环境保护', '单行', '法律', '主要', '有', ':', '1982', '年', '《', '海洋环境保护法', '》', '(', '1999', '年', '.', '2013', '年', '.', '2016', '年', '.', '2017', '年', '修订', ')', '.', '1984', '年', '《', '水污染防治法', '》', '(', '1996', '年', '.', '2008', '年', '.', '2017', '年', '修订', ')', '.', '1987', '年', '《', '大气污染防治法', '》', '(', '1995', '年', '.', '2000', '年', '.', '2015', '年', '修订', ')', '.', '1988', '年', '《', '野生动物保护法', '》', '(', '2004', '年', '.', '2009', '年', '.', '2016', '年', '修订', ')', '.', '1991', '年', '《', '水土保持法', '》', '(', '2010', '年', '修订', ')', '.', '1995', '年', '《', '固体废物污染环境防治法', '》', '(', '2004', '年', '.', '2013', '年', '.', '2015', '年', '.', '2016', '年', '修订', ')', '.', '1996', '年', '《', '环境噪声污染防治法', '》', '.', '2001', '年', '《', '防沙治沙法', '》', '.', '2002', '年', '《', '清洁生产促进法', '》', '(', '2012', '年', '修订', ')', '.', '2002', '年', '《', '环境影响评价法', '》', '(', '2016', '年', '修正', ')', '.', '2005', '年', '《', '可再生能源法', '》', '(', '2009', '年', '修正', ')', '.', '2008', '年', '《', '循环经济促进法', '》', '，', '等等', '。', '此外', '，', '关于', '土地', '.', '矿产', '.', '海洋', '.', '水资源', '.', '森林', '.', '草原', '.', '农业', '.', '渔业', '.', '港口', '.', '种子', '.', '煤炭', '.', '城乡规划', '.', '防洪', '.', '防灾', '减灾', '.', '突发事件', '应对', '等', '领域', '的', '法律', '，', '也', '有', '关于', '环境保护', '的', '规定', '。']
	# 句法树解析错误原因: 句子太长, 包含太多法条
	# 解决方案: 手动分句
	tokens_list = [
		['我国', '现行', '由', '全国人民代表大会常务委员会', '审议', '通过', '并', '颁布', '的', '环境保护', '单行', '法律', '主要', '有', ':'],
		['1982', '年', '《', '海洋环境保护法', '》', '(', '1999', '年', '.', '2013', '年', '.', '2016', '年', '.', '2017', '年', '修订', ')', '.'],
		['1984', '年', '《', '水污染防治法', '》', '(', '1996', '年', '.', '2008', '年', '.', '2017', '年', '修订', ')', '.'],
		['1987', '年', '《', '大气污染防治法', '》', '(', '1995', '年', '.', '2000', '年', '.', '2015', '年', '修订', ')', '.'],
		['1988', '年', '《', '野生动物保护法', '》', '(', '2004', '年', '.', '2009', '年', '.', '2016', '年', '修订', ')', '.'],
		['1991', '年', '《', '水土保持法', '》', '(', '2010', '年', '修订', ')', '.'],
		['1995', '年', '《', '固体废物污染环境防治法', '》', '(', '2004', '年', '.', '2013', '年', '.', '2015', '年', '.', '2016', '年', '修订', ')', '.'],
		['1996', '年', '《', '环境噪声污染防治法', '》', '.'],
		['2001', '年', '《', '防沙治沙法', '》', '.'],
		['2002', '年', '《', '清洁生产促进法', '》', '(', '2012', '年', '修订', ')', '.'],
		['2002', '年', '《', '环境影响评价法', '》', '(', '2016', '年', '修正', ')', '.'],
		['2005', '年', '《', '可再生能源法', '》', '(', '2009', '年', '修正', ')', '.'],
		['2008', '年', '《', '循环经济促进法', '》', '，', '等等', '。'],
		['此外', '，', '关于', '土地', '.', '矿产', '.', '海洋', '.', '水资源', '.', '森林', '.', '草原', '.', '农业', '.', '渔业', '.', '港口', '.', '种子', '.', '煤炭', '.', '城乡规划', '.', '防洪', '.', '防灾', '减灾', '.', '突发事件', '应对', '等', '领域', '的', '法律', '，', '也', '有', '关于', '环境保护', '的', '规定', '。'],
	]
	parse_trees = []
	regex_compiler = re.compile(r'\s+', re.I)	# 用于匹配连续空格的正则
	for tokens in tokens_list:
		parse_tree = generate_parse_tree(tokens=tokens)
		parse_tree = list(map(lambda tree: regex_compiler.sub(' ', str(tree)), parse_tree))
		if len(parse_tree) > 1:
			raise Exception('More than one tree !')
		parse_trees.append(str(parse_tree[0]))
	
	with open(REFERENCE_PARSE_TREE_PATH, 'r', encoding='utf8') as f:
		lines = f.read().splitlines()
	
	lines[19593] = lines[19593].split('\t')[0] + '\t' + str(parse_trees)
	
	with open(REFERENCE_PARSE_TREE_PATH, 'w', encoding='utf8') as f:
		f.write('\n'.join(lines))
	

if __name__ == '__main__':
	initialize_logger(filename=os.path.join(LOGGING_DIR, 'advance_preprocess.log'))
	# easy_preprocess()
	advance_preprocess()
	# special_prerprocess()
