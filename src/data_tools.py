# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# 数据相关工具

if __name__ == '__main__':
	import sys
	sys.path.append('../')

import os
import re
import time
import json
import jieba
import torch
import pandas
import pickle
import logging

from copy import deepcopy
from sklearn.model_selection import train_test_split
from collections import Counter

from setting import *

# 分词: 顺带统计分词词频
def tokenize(sentence, token2frequency):
	tokens = []
	for token in jieba.cut(sentence):	# jieba分词
		tokens.append(token)
		if not token in token2frequency:
			token2frequency[token] = 0
		token2frequency[token] += 1
	return tokens, token2frequency

# 编码选择题的答案: 用{1, 2, 4, 8}分别表示{A, B, C, D}四个选项, 加和得到编码值(本质为二进制编码)
def encode_answer(decoded_answer):
	return sum(list(map(lambda x: 2 ** OPTION2INDEX[x], filter(lambda x: x in OPTION2INDEX, decoded_answer))))

# 解码选择题的答案: encode_answer的反函数
def decode_answer(encoded_answer, result_type=str):
	# 20211124更新: 
	# 1. 之前返回如['A', 'B', 'D']的字符串列表, 为使用判断题形式, 修正为返回如[1, 1, 0, 1]的零一值列表, 通过修改参数result_type为int即可实现
	# 2. 有部分题目的答案是空, 0_train.json中有四道题是这种情况, id为4_772, 1_1379, 1_1365, 1_1206, 此时encoded_answer为0
	assert 0 <= encoded_answer <= 15
	decoded_answer = []
	for index, char in enumerate(bin(encoded_answer)[: 1: -1]):			# [: 1: -1]这种写法就是从最后一个位置开始反向遍历, 直到第1个位置停止, 可以跳过前2个字符ob 
		if char == '1':
			decoded_answer.append(INDEX2OPTION[index])
	if result_type == str:
		return decoded_answer
	elif result_type == int:
		return [1 if option in decoded_answer else 0 for option in OPTION2INDEX]
	raise NotImplementedError(f'Unknown param `result_type`: {result_type}')

# JEC-QA数据集中的题库JSON文件转为CSV文件: 顺带统计分词词频
def json_to_csv(json_path, csv_path, token2frequency=None, mode='train'):
	assert mode in ['train', 'test'], f'Unknown param `mode`: {mode}'
	data_dict = {
		'id'		: [],	# 题目编号
		'statement'	: [],	# 题干分词列表
		'option_a'	: [],	# 选项A分词列表
		'option_b'	: [],	# 选项B分词列表
		'option_c'	: [],	# 选项C分词列表
		'option_d'	: [],	# 选项D分词列表
		'type'		: [],	# 0或1分别表示概念题与情景题
		'subject'	: [],	# 所属参考书目中的18钟法律类型之一(该字段存在62.9%的缺失, 需要使用语言模型进行预测)
	}
	_token2frequency = {} if token2frequency is None else token2frequency.copy()
	if mode == 'train':
		data_dict['answer'] = []	# 训练集比测试集多一个answer字段, 即题目答案
	with open(json_path, 'r', encoding='utf8') as f:
		while True:
			line = f.readline()
			if not line:
				break
			data = json.loads(line)
			assert len(data['option_list']) == TOTAL_OPTIONS	# 确保每道题都是4个选项
			
			# 获取每道题的字段信息
			_id = data['id']
			_type = data['type']
			subject = data.get('subject')	# 该字段可能存在缺失, 可能不支持subscriptable, 因此改用get方法
			statement, _token2frequency = tokenize(data['statement'], _token2frequency)
			option_a, _token2frequency = tokenize(data['option_list']['A'], _token2frequency)
			option_b, _token2frequency = tokenize(data['option_list']['B'], _token2frequency)
			option_c, _token2frequency = tokenize(data['option_list']['C'], _token2frequency)
			option_d, _token2frequency = tokenize(data['option_list']['D'], _token2frequency)
			
			# 记录每道题的字段信息
			data_dict['id'].append(_id)
			data_dict['statement'].append(statement)
			data_dict['option_a'].append(option_a)
			data_dict['option_b'].append(option_b)
			data_dict['option_c'].append(option_c)
			data_dict['option_d'].append(option_d)
			data_dict['type'].append(_type)
			data_dict['subject'].append(subject)
			if mode == 'train':
				answer = encode_answer(data['answer'])
				data_dict['answer'].append(answer)
	
	# 字典转为DataFrame并导出为CSV文件
	dataframe = pandas.DataFrame(data_dict, columns=list(data_dict.keys()))
	if csv_path is not None:
		dataframe.to_csv(csv_path, sep='\t', index=False, header=True)
	return dataframe, _token2frequency

# 20211101更新: 划分0_train.csv和1_train.csv文件得到验证集, 用于本地测试
def split_validset(dataframe, train_export_path, valid_export_path, valid_ratio=VALID_RATIO):
	dataframe_train, dataframe_valid = train_test_split(dataframe, test_size=valid_ratio)
	if train_export_path is not None:
		dataframe_train.to_csv(train_export_path, sep='\t', index=False, header=True)
	if valid_export_path is not None:
		dataframe_valid.to_csv(valid_export_path, sep='\t', index=False, header=True)
	return dataframe_train, dataframe_valid

# token2id字典转为CSV文件
def token2id_to_csv(export_path, token2frequency):
	token2id = TOKEN2ID.copy()
	_id = len(token2id)
	for token, frequency in token2frequency.items():
		if frequency >= FREQUENCY_THRESHOLD:
			token2id[token] = _id
			_id += 1
	pandas.DataFrame({
		'id'	: list(token2id.values()),
		'token'	: list(token2id.keys()),
	}).sort_values(by=['id'], ascending=True).to_csv(export_path, sep='\t', index=False, header=True)

# token2frequency字典转为CSV文件: 按照分词频次降序排列
def token2frequency_to_csv(export_path, token2frequency):
	pandas.DataFrame({
		'token'		: list(token2frequency.keys()),
		'frequency'	: list(token2frequency.values()),
	}).sort_values(by=['frequency'], ascending=False).to_csv(export_path, sep='\t', index=False, header=True)

# 中文数词转数字: 用于识别参考书目的章节信息, 可处理至多两位数的中文写法
def chinese_to_number(string):
	easy_mapper = {
		'一': '1', '二': '2', '两': '2',
		'三': '3', '四': '4', '五': '5',
		'六': '6', '七': '7', '八': '8',
		'九': '9', '十': '0',
	}
	number_string = ''.join(list(map(easy_mapper.get, string)))
	if number_string[0] == '0':
		number = int(number_string) + 10
	elif number_string[-1] == '0':
		number = int(number_string)
	else:
		number = int(number_string.replace('0', ''))
	return number

# JEC-QA数据集中的参考书目TXT文件转为CSV格式文件: 顺带统计分词词频
def reference_to_csv(export_path, token2frequency=None):
	reference_dict = {
		'law'			: [],	# 法律门类: 即reference_book下18门法律
		'chapter_number': [],	# 章节编号: 该字段直接从文件名中抽取即可
		'chapter_name'	: [],	# 章节名称: 有的文件名中有章节名称, 有的则没有, 需要从文件内容中自动识别, 一般在开头部分
		'section'		: [],	# 小节名称: TXT文件中每一行除最后一个分块外的所有内容的拼接
		'content'		: [],	# 实际内容分词列表: TXT文件中每一行最后一个分块
	}
	_token2frequency = {} if token2frequency is None else token2frequency.copy()
	# 遍历所有法律门类
	for law in os.listdir(REFERENCE_DIR):
		# 遍历所有章节
		for filename in os.listdir(os.path.join(REFERENCE_DIR, law)):
			if filename.endswith('.txt'):								# 存在一些无用的特殊文件, 如刑事诉讼法中有一个.swp文件)
				_filename = filename.replace(' ', '').replace('.txt', '')
				start_index = _filename.find('第') + 1
				end_index = _filename.find('章')
				if start_index == 0 or end_index == -1:					# 跳过文件名中没有章节信息的文件, 如目录和中国法律史中有一个目录.txt文件是无用的
					continue
				chapter_number_1 = _filename[start_index: end_index]	# 文件名中的章节编号: 经过检验这个字段要比从文件内容中抽取的章节编号更准确
				chapter_name_1 = _filename[end_index + 1: ]				# 文件名中的章节名称: 可能缺失
				filepath = os.path.join(REFERENCE_DIR, law, filename)	# 文件路径
				with open(filepath, 'r', encoding='utf8') as f:
					paragraphs = eval(f.read())							# 每个文件中的文档内容由字符串段落组成的列表样式数据组成
				total_paragraphs = len(paragraphs)						# 统计总段落数
				
				# 该循环是为了从文件内容二次识别章节编号与章节名称, 因为文件名中的章节名称可能缺失
				for i in range(total_paragraphs):
					paragraph_string = paragraphs[i].replace(' ', '')
					start_index = paragraph_string.find('第') + 1
					end_index = paragraph_string.find('章')
					chapter_number_2 = paragraph_string[start_index: end_index]
					if start_index != 0 and end_index != -1:
						chapter_name_2 = paragraphs[i + 1].replace(' ', '') if paragraph_string[-1] == '章' else paragraph_string[paragraph_string.find('章') + 1:]
						break
				
				# 章节编号与章节名称最终确定
				chapter_number = chinese_to_number(chapter_number_1)				# 目前认为直接使用文件名中的章节编号即可
				chapter_name = chapter_name_1 if chapter_name_1 else chapter_name_2	# 优先使用文件名中的章节名称, 若缺失则使用文件内容中识别的章节名称
				
				# 该循环是记录文件内容中每个段落的小节名称与实际内容
				for i in range(total_paragraphs):
					blocks = paragraphs[i].strip().split(' ')
					section = ' '.join(blocks[: -1])	# 小节名称为每一行除最后一个分块外的所有内容的拼接
					content, _token2frequency = tokenize(blocks[-1], _token2frequency)
					reference_dict['law'].append(law)
					reference_dict['chapter_number'].append(chapter_number)
					reference_dict['chapter_name'].append(chapter_name)
					reference_dict['section'].append(section)
					reference_dict['content'].append(content)
	
	# 字典转为DataFrame并导出为CSV文件
	reference_dataframe = pandas.DataFrame(reference_dict, columns=list(reference_dict.keys()))
	if export_path is not None:
		reference_dataframe.to_csv(export_path, sep='\t', header=True, index=False)
	return reference_dataframe, _token2frequency

# 加载停用词: 默认加载stopwords-master中所有的停用词
def load_stopwords(stopword_names=None):
	if stopword_names is None:
		stopword_names = STOPWORD_PATHs.keys()
	stopwords = []
	for stopword_name in stopword_names:
		with open(STOPWORD_PATHs[stopword_name], 'r', encoding='utf8') as f:
			stopwords.extend(f.read().splitlines())
	return list(set(stopwords))

# 过滤停用词: 默认使用stopwords-master中所有的停用词
def filter_stopwords(tokens, stopwords=None):
	if stopwords is None:
		stopwords = load_stopwords(stopword_names=None)
	return list(filter(lambda x: not x in stopwords, tokens))
	
# 2022/01/10 11:50:13 加载本地字典
def load_userdict_jieba(ignored_filenames=[]):
	for filename in os.listdir(USERDICT_DIR):
		if not filename in ignored_filenames:
			filepath = os.path.join(USERDICT_DIR, filename)
			jieba.load_userdict(filepath)

# 2022/03/10 11:56:08 根据一些规则生成用户字典的词条, 具体逻辑见函数内注释, 可能会进一步进行划分
def generate_userdict(export_path=USERDICT_PATHs['diy']):
	userdict = []
	
	# 2022/03/10 15:35:33 第一部分: 匹配参考书目文档中被书名号夹住的内容
	regex = '《[^》]+》'
	max_length = 64
	compiler = re.compile(regex, re.I)
	for law in os.listdir(REFERENCE_DIR):
		# 遍历所有章节
		for filename in os.listdir(os.path.join(REFERENCE_DIR, law)):
			if filename.endswith('.txt'):								
				# 存在一些无用的特殊文件, 如刑事诉讼法中有一个.swp文件, 只对后缀名为.txt的文件进行处理
				filepath = os.path.join(REFERENCE_DIR, law, filename)	
				with open(filepath, 'r', encoding='utf8') as f:
					text = f.read()				
				results = compiler.findall(text)
				results = list(map(lambda x: x[1: -1], results))	# 去除两边的书名号, 一般分词字典中不会出现书名号
				userdict.extend(list(filter(lambda result: len(result) < max_length, results)))
	
	# 2022/03/10 15:35:49 第二部分: 匹配章节名称
	reference_dataframe = pandas.read_csv(REFERENCE_PATH, sep='\t', header=0, dtype=str)	# 预处理后的参数书目文档
	reference_dataframe = reference_dataframe.fillna('')									# 参考书目的section字段存在缺失, 可使用空字符串填充
	
	userdict.extend(reference_dataframe['chapter_name'].tolist())
	
	# 导出用户字典
	counter = Counter(userdict)
	with open(export_path, 'w', encoding='utf8') as f:
		for token, frequency in counter.items():
			# f.write(f'{token}\t{frequency}\n')
			f.write(f'{token}\n')

	return userdict


