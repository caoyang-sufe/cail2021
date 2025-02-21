# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
# 全局变量设定

import os
import torch
import platform

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PLATFORM = platform.system()

# Linux系统使用相对路径读取文件时需要添加前缀
# 2021/12/27 13:32:39 似乎Linux系统中也不完全都需要在相对路径前添加斜杠
# DIR_SUFFIX = '' if PLATFORM == 'Windows' else '/'
DIR_SUFFIX = ''

# 一些零散的全局变量
OPTION2INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}									# 选项对应的索引
INDEX2OPTION = {_INDEX: _OPTION for _OPTION, _INDEX in OPTION2INDEX.items()}	# 索引对应的选项
TOTAL_OPTIONS = len(OPTION2INDEX)												# 每道题固定的选项数
VALID_RATIO = .1																# 从训练数据中划分验证集的比例(用于本地测试)
TOKEN2ID = {'PAD': 0, 'UNK': 1}													# 预设的特殊分词符号
FREQUENCY_THRESHOLD = 1															# 统计分词的最小频次
LAW2SUBJECT = {'目录和中国法律史': '法制史'}										# JEC-QA数据集中的subject字段与参考书目文档下的文件名基本是相同的, 目前只有'目录与中国法律史'与'法制史'不同, 特列出
SUBJECT2LAW = {_SUBJECT: _LAW for _LAW, _SUBJECT in LAW2SUBJECT.items()}

# -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*-
# 1. data文件夹及其结构设定
DATA_DIR = DIR_SUFFIX + 'data'

RAWDATA_DIR		= os.path.join(DATA_DIR, 'JEC-QA')						# [JEC-QA.zip](https://jecqa.thunlp.org/readme)压缩包解压到当前文件夹可得RAWDATA_DIR中的若干内容
REFERENCE_DIR	= os.path.join(RAWDATA_DIR, 'reference_book')			# 存放JEC-QA数据集中的原始参考书目文档，包含在[JEC-QA.zip](https://jecqa.thunlp.org/readme)中

SUBJECT2INDEX = {LAW2SUBJECT.get(_LAW, _LAW): _INDEX for _INDEX, _LAW in enumerate(os.listdir(REFERENCE_DIR))}	# 法律门类对应索引
INDEX2SUBJECT = {_INDEX: _SUBJECT for _SUBJECT, _INDEX in SUBJECT2INDEX.items()}								# 索引对应法律门类

NEWDATA_DIR					= os.path.join(DATA_DIR, 'JEC-QA-preprocessed')						# 存放预处理得到的新数据

BERT_OUTPUT_DIR				= os.path.join(NEWDATA_DIR, 'bert_output')							# 2022/03/20 10:22:22 存放BERT模型输出的文件夹, 之所以用单独文件夹存放, 是因为无法单独存储, 实在是太大了

REFERENCE_BERT_OUTPUT_DIR	= os.path.join(BERT_OUTPUT_DIR, 'reference_bert_output')			# 2022/03/22 19:47:31 BERT输出存储到本地文件的临时目录(因为会生成很多子文件, 先存放在此, 然后合并到REFERENCE_BERT_OUTPUT_PATH)
QUESTION_BERT_OUTPUT_DIR	= os.path.join(BERT_OUTPUT_DIR, 'question_bert_output')				# 2022/03/20 00:25:21 BERT输出存储到本地文件的临时目录(因为会生成很多子文件, 先存放在此, 然后合并到QUESTION_BERT_OUTPUT_PATH)

REFERENCE_POOLER_OUTPUT_DIR		= os.path.join(REFERENCE_BERT_OUTPUT_DIR, 'pooler_output')		# 2022/03/24 20:39:11 pooler_output的结果存放目录, 若干形状为(1, 768)的张量dill文件, 经证实效果不是很好(参考书目)
REFERENCE_LAST_HIDDEN_STATE_DIR	= os.path.join(REFERENCE_BERT_OUTPUT_DIR, 'last_hidden_state')	# 2022/03/24 20:39:11 last_hidden_state的结果存放目录, 若干形状为(1, 42, 768)的张量dill文件, 待测试(参考书目)
QUESTION_POOLER_OUTPUT_DIR		= os.path.join(QUESTION_BERT_OUTPUT_DIR, 'pooler_output')		# 2022/03/24 20:39:11 pooler_output的结果存放目录, 若干形状为(1, 768)的张量dill文件, 经证实效果不是很好(题库)
QUESTION_LAST_HIDDEN_STATE_DIR	= os.path.join(QUESTION_BERT_OUTPUT_DIR, 'last_hidden_state')	# 2022/03/24 20:39:11 last_hidden_state的结果存放目录, 若干形状为(1, 42, 768)的张量dill文件, 待测试(题库)

STOPWORDS_DIR	= os.path.join(DATA_DIR, 'stopwords-master')			# [stopwords-master.zip](https://github.com/goto456/stopwords)压缩包解压到当前文件夹可得

USERDICT_DIR	= os.path.join(DATA_DIR, 'userdict')					# 2022/01/10 11:51:52 用于分词的字典

RAW_TRAINSET_PATHs	= [os.path.join(RAWDATA_DIR, '0_train.json'), os.path.join(RAWDATA_DIR, '1_train.json')]# 原始训练集包含两个JSON文件: 第一个是概念题, 第二个是情境题
RAW_TESTSET_PATHs	= [os.path.join(RAWDATA_DIR, '0_test.json'), os.path.join(RAWDATA_DIR, '1_test.json')]	# 原始测试集包含两个JSON文件: 第一个是概念题, 第二个是情境题
TRAINSET_PATHs	= [os.path.join(NEWDATA_DIR, '0_train.csv'), os.path.join(NEWDATA_DIR, '1_train.csv')]		# 预处理后的训练集包含两个JSON文件: 第一个是概念题, 第二个是情境题
VALIDSET_PATHs	= [os.path.join(NEWDATA_DIR, '0_valid.csv'), os.path.join(NEWDATA_DIR, '1_valid.csv')]		# 预处理后的验证集包含两个JSON文件: 第一个是概念题, 第二个是情境题
TESTSET_PATHs	= [os.path.join(NEWDATA_DIR, '0_test.csv'), os.path.join(NEWDATA_DIR, '1_test.csv')]		# 预处理后的测试集包含两个JSON文件: 第一个是概念题, 第二个是情境题

TOKEN2ID_PATH					= os.path.join(NEWDATA_DIR, 'token2id.csv')						# 预处理得到的分词编号文件(题库)
TOKEN2FREQUENCY_PATH			= os.path.join(NEWDATA_DIR, 'token2frequency.csv')				# 预处理得到的分词词频文件(题库)
REFERENCE_PATH					= os.path.join(NEWDATA_DIR, 'reference_book.csv')				# 预处理得到的参考书目文件
REFERENCE_TOKEN2ID_PATH			= os.path.join(NEWDATA_DIR, 'reference_token2id.csv')			# 预处理得到的分词编号文件(参考书目)
REFERENCE_TOKEN2FREQUENCY_PATH	= os.path.join(NEWDATA_DIR, 'reference_token2frequency.csv')	# 预处理得到的分词词频文件(参考书目)

REFERENCE_PARSE_TREE_PATH		= os.path.join(NEWDATA_DIR, 'reference_parse_tree.csv')			# 2022/03/10 10:15:32 预处理得到的句法解析树文件(参考书目)
REFERENCE_DEPENDENCY_PATH		= os.path.join(NEWDATA_DIR, 'reference_dependency.csv')			# 2022/06/12 10:42:12 预处理得到的依存关系文件(参考书目)
QUESTION_PARSE_TREE_PATH		= os.path.join(NEWDATA_DIR, 'question_parse_tree.csv')			# 2022/03/15 19:59:41 预处理得到的句法解析树文件(题库)
QUESTION_DEPENDENCY_PATH		= os.path.join(NEWDATA_DIR, 'question_dependency.csv')			# 2022/06/12 10:42:12 预处理得到的句法解析树文件(题库)

REFERENCE_POOLER_OUTPUT_PATH		= os.path.join(REFERENCE_BERT_OUTPUT_DIR, 'reference_pooler_output.dill')		# 2022/03/20 00:25:21 BERT输出pooler_output本地合并文件(参考书目)
REFERENCE_LAST_HIDDEN_STATE_PATH	= os.path.join(REFERENCE_BERT_OUTPUT_DIR, 'reference_last_hidden_state.dill')	# 2022/03/24 20:46:11 BERT输出last_hidden_state本地合并文件(参考书目)
QUESTION_POOLER_OUTPUT_PATH			= os.path.join(QUESTION_BERT_OUTPUT_DIR, 'question_pooler_output.dill')			# 2022/03/20 00:25:21 BERT输出pooler_output本地合并文件(题库)
QUESTION_LAST_HIDDEN_STATE_PATH		= os.path.join(QUESTION_BERT_OUTPUT_DIR, 'question_last_hidden_state.dill')		# 2022/03/24 20:46:11 BERT输出last_hidden_state本地合并文件(题库)

STOPWORD_PATHs = {
    'baidu'		: os.path.join(STOPWORDS_DIR, 'baidu_stopwords.txt'),	# 百度停用词表
    'cn'		: os.path.join(STOPWORDS_DIR, 'cn_stopwords.txt'),		# 中文停用词表
    'hit'		: os.path.join(STOPWORDS_DIR, 'hit_stopwords.txt'),		# 哈工大停用词表
    'scu'		: os.path.join(STOPWORDS_DIR, 'scu_stopwords.txt'),		# 四川大学机器智能实验室停用词库
}

USERDICT_PATHs = {
	'crime'		: os.path.join(USERDICT_DIR, 'crime.txt'),		# 罪名分词
	'THUOCL_law': os.path.join(USERDICT_DIR, 'THUOCL_law.txt'),	# 清华法律分词
	'diy'		: os.path.join(USERDICT_DIR, 'diy.txt'),		# 2022/03/10 12:23:38 根据参考书目文档的自定义分词
}

# -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*-
# 2. logging文件夹及其结构设定
LOGGING_DIR = DIR_SUFFIX + 'logging'

# -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*-
# 3. temp文件夹及其结构设定
TEMP_DIR = DIR_SUFFIX + 'temp'

TEMP_SIMILARITY_DIR = os.path.join(TEMP_DIR, 'similarity')		# 2022/02/18 10:01:24 存放gensim文档检索模型的相似度临时文件

# -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*-
# 4. checkpoint文件夹及其结构设定
CHECKPOINT_DIR = DIR_SUFFIX + 'checkpoint'


# -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*-
# 5. image文件夹及其结构设定
IMAGE_DIR = DIR_SUFFIX + 'image'

# -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*-
# 6. model文件夹及其结构设定
MODEL_DIR = DIR_SUFFIX + 'model'
RETRIEVAL_MODEL_DIR = os.path.join(MODEL_DIR, 'retrieval_model')
GENSIM_RETRIEVAL_MODEL_DIR = os.path.join(RETRIEVAL_MODEL_DIR, 'gensim')

REFERENCE_DOCUMENT_PATH				= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_document.pk')				# 参考书目字典
REFERENCE_DICTIONARY_PATH			= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_dictionary.dtn')			# 参考书目字典
REFERENCE_CORPUS_PATH				= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_corpus.cps')				# 参考书目分词权重(原始词频)
REFERENCE_CORPUS_TFIDF_PATH			= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_corpus_tfidf.cps')		# 参考书目分词权重(TFIDF处理后)
REFERENCE_CORPUS_LSI_PATH			= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_corpus_lsi.cps')			# 参考书目分词权重(LSI处理后)
REFERENCE_CORPUS_LDA_PATH			= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_corpus_lda.cps')			# 参考书目分词权重(LDA处理后)
REFERENCE_CORPUS_HDP_PATH			= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_corpus_hdp.cps')			# 参考书目分词权重(HDP处理后)
REFERENCE_CORPUS_LOGENTROPY_PATH	= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_corpus_logentropy.cps')	# 参考书目分词权重(LOGENTROPY处理后)
REFERENCE_TFIDF_MODEL_PATH			= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_tfidf.m')					# 参考书目TFIDF模型
REFERENCE_LSI_MODEL_PATH			= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_lsi.m')					# 参考书目LSI模型
REFERENCE_LDA_MODEL_PATH			= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_lda.m')					# 参考书目LDA模型
REFERENCE_HDP_MODEL_PATH			= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_hdp.m')					# 参考书目HDP模型
REFERENCE_LOGENTROPY_MODEL_PATH		= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_logentropy.m')			# 参考书目LogEntropy模型

# -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*-
# 7. 文档检索模型相关
# 类似注册表的字典, 便于相关代码简化
# build_function	: 在src.retrieval_model中对应的模型构建方法
# class				: 在gensim中对应的模型类
# sequence			: 该模型依次需要调用的模型序列, 如LSI模型需要先调用TFIDF生成词权矩阵后再进行奇异值分解
GENSIM_RETRIEVAL_MODEL_SUMMARY = {
	'tfidf': {
		'corpus'		: REFERENCE_CORPUS_TFIDF_PATH,
		'model'			: REFERENCE_TFIDF_MODEL_PATH,
		'dictionary'	: REFERENCE_DICTIONARY_PATH,
		'build_function': 'GensimRetrievalModel.build_tfidf_model',
		'class'			: 'gensim.models.TfidfModel',
		'sequence'		: ['tfidf'],					
	},
	'lsi': {
		'corpus'		: REFERENCE_CORPUS_LSI_PATH,
		'model'			: REFERENCE_LSI_MODEL_PATH,
		'dictionary'	: REFERENCE_DICTIONARY_PATH,
		'build_function': 'GensimRetrievalModel.build_lsi_model',		
		'class'			: 'gensim.models.LsiModel',
		'sequence'		: ['tfidf', 'lsi'],
	},
	'lda': {
		'corpus'		: REFERENCE_CORPUS_LDA_PATH,
		'model'			: REFERENCE_LDA_MODEL_PATH,
		'dictionary'	: REFERENCE_DICTIONARY_PATH,
		'build_function': 'GensimRetrievalModel.build_lda_model',
		'class'			: 'gensim.models.LdaModel',
		'sequence'		: ['tfidf', 'lda'],
	},
	
	# 20211210新增HDP模型
	'hdp': {
		'corpus'		: REFERENCE_CORPUS_HDP_PATH,
		'model'			: REFERENCE_HDP_MODEL_PATH,
		'dictionary'	: REFERENCE_DICTIONARY_PATH,
		'build_function': 'GensimRetrievalModel.build_hdp_model',
		'class'			: 'gensim.models.HdpModel',
		'sequence'		: ['hdp'],
	},	
	
	# 20211210新增LogEntropy模型
	'logentropy': {
		'corpus'		: REFERENCE_CORPUS_LOGENTROPY_PATH,
		'model'			: REFERENCE_LOGENTROPY_MODEL_PATH,
		'dictionary'	: None,											# 不知为何gensim.models.LogEntropyModel的构造参数里竟然没有id2word
		'build_function': 'GensimRetrievalModel.build_logentropy_model',
		'class'			: 'gensim.models.LogEntropyModel',
		'sequence'		: ['logentropy'],
	},
}

GENSIM_RETRIEVAL_MODEL_SUBJECT_DIR = os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'subject')
GENSIM_RETRIEVAL_MODEL_SUBJECT_SUMMARY = {
	_subject: {
		'document'	: os.path.join(GENSIM_RETRIEVAL_MODEL_SUBJECT_DIR, _subject, 'reference_document.pk'),
		'dictionary': os.path.join(GENSIM_RETRIEVAL_MODEL_SUBJECT_DIR, _subject, 'reference_dictionary.dtn'),
		'corpus'	: os.path.join(GENSIM_RETRIEVAL_MODEL_SUBJECT_DIR, _subject, 'reference_corpus.cps'),
		'summary'	: {
			'tfidf': {
				'corpus'		: os.path.join(GENSIM_RETRIEVAL_MODEL_SUBJECT_DIR, _subject, 'reference_tfidf_corpus.cps'),
				'model'			: os.path.join(GENSIM_RETRIEVAL_MODEL_SUBJECT_DIR, _subject, 'reference_tfidf.m'),
				'dictionary'	: os.path.join(GENSIM_RETRIEVAL_MODEL_SUBJECT_DIR, _subject, 'reference_dictionary.dtn'),
				'build_function': 'GensimRetrievalModel.build_tfidf_model',
				'class'			: 'gensim.models.TfidfModel',
				'sequence'		: ['tfidf'],					
			},
			'lsi': {
				'corpus'		: os.path.join(GENSIM_RETRIEVAL_MODEL_SUBJECT_DIR, _subject, 'reference_lsi_corpus.cps'),
				'model'			: os.path.join(GENSIM_RETRIEVAL_MODEL_SUBJECT_DIR, _subject, 'reference_lsi.m'),
				'dictionary'	: os.path.join(GENSIM_RETRIEVAL_MODEL_SUBJECT_DIR, _subject, 'reference_dictionary.dtn'),
				'build_function': 'GensimRetrievalModel.build_lsi_model',		
				'class'			: 'gensim.models.LsiModel',
				'sequence'		: ['tfidf', 'lsi'],
			},
			'lda': {
				'corpus'		: os.path.join(GENSIM_RETRIEVAL_MODEL_SUBJECT_DIR, _subject, 'reference_lda_corpus.cps'),
				'model'			: os.path.join(GENSIM_RETRIEVAL_MODEL_SUBJECT_DIR, _subject, 'reference_lda.m'),
				'dictionary'	: os.path.join(GENSIM_RETRIEVAL_MODEL_SUBJECT_DIR, _subject, 'reference_dictionary.dtn'),
				'build_function': 'GensimRetrievalModel.build_lda_model',
				'class'			: 'gensim.models.LdaModel',
				'sequence'		: ['tfidf', 'lda'],
			},
			'hdp': {
				'corpus'		: os.path.join(GENSIM_RETRIEVAL_MODEL_SUBJECT_DIR, _subject, 'reference_hdp_corpus.cps'),
				'model'			: os.path.join(GENSIM_RETRIEVAL_MODEL_SUBJECT_DIR, _subject, 'reference_hdp.m'),
				'dictionary'	: os.path.join(GENSIM_RETRIEVAL_MODEL_SUBJECT_DIR, _subject, 'reference_dictionary.dtn'),
				'build_function': 'GensimRetrievalModel.build_hdp_model',
				'class'			: 'gensim.models.HdpModel',
				'sequence'		: ['hdp'],
			},	
			'logentropy': {
				'corpus'		: os.path.join(GENSIM_RETRIEVAL_MODEL_SUBJECT_DIR, _subject, 'reference_logentropy_corpus.cps'),
				'model'			: os.path.join(GENSIM_RETRIEVAL_MODEL_SUBJECT_DIR, _subject, 'reference_logentropy.m'),
				'dictionary'	: None,
				'build_function': 'GensimRetrievalModel.build_logentropy_model',
				'class'			: 'gensim.models.LogEntropyModel',
				'sequence'		: ['logentropy'],
			},
		} 
	} for _subject in SUBJECT2INDEX
}

# -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*-
# 8. 嵌入模型相关
EMBEDDING_MODEL_DIR = os.path.join(MODEL_DIR, 'embedding_model')		
GENSIM_EMBEDDING_MODEL_DIR = os.path.join(EMBEDDING_MODEL_DIR, 'gensim')

REFERENCE_WORD2VEC_MODEL_PATH = os.path.join(GENSIM_EMBEDDING_MODEL_DIR, 'reference_word2vec.m')	# 参考书目文档训练得到的word2vec模型
REFERENCE_FASTTEXT_MODEL_PATH = os.path.join(GENSIM_EMBEDDING_MODEL_DIR, 'reference_fasttext.m')	# 参考书目文档训练得到的fasttext模型
REFERENCE_DOC2VEC_MODEL_PATH = os.path.join(GENSIM_EMBEDDING_MODEL_DIR, 'reference_doc2vec.m')		# 参考书目文档训练得到的doc2vec模型: 该模型不用于测试检索

# 类似注册表的字典, 便于相关代码简化
# build_function	: 在src.embedding_model中对应的模型构建方法
# class				: 在gensim中对应的模型类
GENSIM_EMBEDDING_MODEL_SUMMARY = {
	'word2vec': {
		'model': REFERENCE_WORD2VEC_MODEL_PATH,
		'class': 'gensim.models.Word2Vec',
	},
	'fasttext': {
		'model': REFERENCE_FASTTEXT_MODEL_PATH,
		'class': 'gensim.models.FastText',
	},
	'doc2vec': {
		'model': REFERENCE_DOC2VEC_MODEL_PATH,
		'class': 'gensim.models.Doc2Vec',
	}
}

TRANSFORMERS_EMBEDDING_MODEL_DIR = os.path.join(EMBEDDING_MODEL_DIR, 'transformers')	# Transformers库中调用的HuggingFace模型目录
BERT_MODEL_DIR = os.path.join(TRANSFORMERS_EMBEDDING_MODEL_DIR, 'bert')					# BERT模型目录

BERT_MODEL_SUMMARY = {
	'bert-base-chinese': {
		'root': os.path.join(BERT_MODEL_DIR, 'bert-base-chinese'),		
		'config': os.path.join(BERT_MODEL_DIR, 'bert-base-chinese', 'config.json'),	
		'vocab': os.path.join(BERT_MODEL_DIR, 'bert-base-chinese', 'vocab.txt'),	
	}
}

# -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*-
# 9. 句法解析树相关
# 句法解析模型存储目录
PARSER_MODEL_DIR = os.path.join(MODEL_DIR, 'parser_model')
STANFORD_PARSER_MODEL_DIR = os.path.join(PARSER_MODEL_DIR, 'stanford_parser')
STANFORD_PARSER_MODEL_SUMMARY = {
	'jar': os.path.join(STANFORD_PARSER_MODEL_DIR, 'stanford-parser.jar'),
	'models': os.path.join(STANFORD_PARSER_MODEL_DIR, 'stanford-parser-4.2.0-models.jar'),
	'pcfg': {
		'chinese': os.path.join(STANFORD_PARSER_MODEL_DIR, 'chinesePCFG.ser.gz'),
		'english': os.path.join(STANFORD_PARSER_MODEL_DIR, 'englishPCFG.ser.gz'),
		'english.caseless': os.path.join(STANFORD_PARSER_MODEL_DIR, 'englishPCFG.caseless.ser.gz'),
	},
}

# Stanford句法树标注集的一些配置
# 2022/03/29 20:57:45 句法解析树会把原分词序列中的一些特殊符号分词改写为其他形式
# 2022/03/29 20:57:54 如-LRB-表示左小括号, -RRB-表示右小括号, 单空格会被\xa0替换
STANFORD_SPECIAL_SYMBOL_MAPPING = {'(': '-LRB-', ')': '-RRB-', '\xa0': ''}	# jiaba分词序列转为句法树后被替换的分词
STANFORD_IGNORED_SYMBOL = {'\u3000', ' '}									# jieba分词序列转为句法树后被删除的分词

# Stanford中文词性标注集(对应句法树叶子节点上的标注, 共33个), 具体说明见https://blog.csdn.net/weixin_30642561/article/details/97772970
STANFORD_POS_TAG = [
	'AD', 'AS', 'BA', 'CC', 'CD', 'CS', 'DEC', 'DEG', 'DER', 'DEV', 'DT', 
	'ETC', 'FW', 'IJ', 'JJ', 'LB', 'LC', 'M', 'MSP', 'NN', 'NR', 'NT', 
	'OD', 'P', 'PN', 'PU', 'SB', 'SP', 'URL', 'VA', 'VC', 'VE', 'VV', 
]

STANFORD_POS_TAG_INDEX = {_POS_TAG: _INDEX for _INDEX, _POS_TAG in enumerate(STANFORD_POS_TAG)}

# Stanford中文句法依存分析标注集(共33+28个, 前33个来自STANFORD_POS_TAG, 后28个对应句法树非叶节点上的标注)
STANFORD_SYNTACTIC_TAG = STANFORD_POS_TAG + [
	'ADJP', 'ADVP', 'CLP', 'CP', 'DFL', 'DNP', 'DP', 'DVP', 'FLR', 
	'FRAG', 'INC', 'INTJ', 'IP', 'LCP', 'LST', 'NP', 'PP', 'PRN', 'QP', 
	'ROOT', 'UCP', 'VCD', 'VCP', 'VNV', 'VP', 'VPT', 'VRD', 'VSB',
]

STANFORD_SYNTACTIC_TAG_INDEX = {_SYNTACTIC_TAG: _INDEX for _INDEX, _SYNTACTIC_TAG in enumerate(STANFORD_SYNTACTIC_TAG)}
