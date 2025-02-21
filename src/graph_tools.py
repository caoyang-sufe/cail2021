# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# 图模型相关工具函数

if __name__ == '__main__':
	import sys
	sys.path.append('../')

import os
import re
import dgl
import time
import json
import nltk
import warnings
import jieba.posseg as peg

from pprint import pprint
from networkx import DiGraph, draw
from matplotlib import pyplot as plt
from nltk.parse.stanford import StanfordParser, StanfordDependencyParser

from setting import *

from src.graph_tools import *

from src.data_tools import load_stopwords, filter_stopwords
from src.utils import load_args, timer

# 2022/01/02 20:14:05 绘制不同法律门类的词云, 去除停用词, 默认保存到TEMP_DIR下
@timer
def plot_reference_wordcloud(export_root=TEMP_DIR):
	reference_dataframe = pandas.read_csv(REFERENCE_PATH, sep='\t', header=0)
	stopwords = load_stopwords(stopword_names=None)
	for law, group_dataframe in reference_dataframe.groupby(['law']):
		group_dataframe = group_dataframe.reset_index(drop=True)
		contents = []
		for i in range(group_dataframe.shape[0]):
			content = ' '.join(filter_stopwords(tokens=eval(group_dataframe.loc[i, 'content']), stopwords=stopwords))
			contents.append(content)
		text = '\n'.join(contents)
		wordcloud = WordCloud().generate(text=text)
		wordcloud.to_file(os.path.join(TEMP_DIR, f'{law}.png'))

# 2022/03/01 13:56:32 生成词性标注, 目前使用jieba, 另外nltk.stanford句法解析包里也有词性标注的文件, 不过目前以jieba为准
@timer
def generate_pos_tags(sentence):
	pos_tags = peg.lcut(sentence, use_paddle=True)
	return pos_tags

# 2022/03/30 12:04:02 从句法树中调取词性标注, 经验证这个分词序列与原序列是完全相同的
# 2022/03/30 19:29:04 但是存在若干替换对应, 详见setting.py中的STANFORD_SPECIAL_SYMBOL_MAPPING字典中所示
def generate_pos_tags_from_parse_tree(parse_tree, regex_compiler = re.compile('\([^\(\)]+\)', re.I)):
	if not isinstance(parse_tree, str):
		# 2022/03/09 19:48:50 只对字符串形式的树进行处理, 原数据结构不便于处理
		parse_tree = str(parse_tree)

	results = regex_compiler.findall(parse_tree)
	pos_tags = list(map(lambda result: tuple(result[1: -1].split(' ', 1)), results))
	return pos_tags

# 2022/03/07 16:05:41 生成句法解析树, 使用的是stanford-parser-full-2020-11-17
@timer
def generate_parse_tree(tokens, language='chinese'):
	parser = StanfordParser(STANFORD_PARSER_MODEL_SUMMARY['jar'],
							STANFORD_PARSER_MODEL_SUMMARY['models'],
							model_path=STANFORD_PARSER_MODEL_SUMMARY['pcfg'][language])
	parse_tree = list(parser.parse(tokens))
	return parse_tree

# 2022/06/12 10:46:20 生成依存关系, 使用的是stanford-parser-full-2020-11-17
@timer
def generate_dependency(tokens, language='chinese'):
	parser = StanfordDependencyParser(STANFORD_PARSER_MODEL_SUMMARY['jar'],
									  STANFORD_PARSER_MODEL_SUMMARY['models'],
									  model_path=STANFORD_PARSER_MODEL_SUMMARY['pcfg'][language])
	dependency = list(parser.parse(tokens))
	return dependency

# 2022/04/28 23:30:39 改进后的parse_tree_to_graph函数: 使用编号命名节点, 将树节点的内容保存在节点数据中, 参数display表示是否可视化句法树
# 2022/05/20 09:35:22 添加新参数return_type, 用于区分返回networkx图与dgl图
# 2022/05/20 12:43:23 添加新参数ignore_text, 用于区分是否保留叶子节点上的文本内容
def parse_tree_to_graph(parse_tree, display=False, return_type='networkx', ignore_text=False):
	assert return_type in ['networkx', 'dgl'], f'Unknown param `return_type`: {return_type}'
	if not isinstance(parse_tree, str):
		# 2022/03/09 19:48:50 只对字符串形式的树进行处理, 原数据结构不便于处理
		warnings.warn(f'Expect `parse_tree` to be string, rather than {type(parse_tree)}')
		parse_tree = str(parse_tree)

	graph = DiGraph()		# 2022/03/09 20:48:52 绘制句法树的有向图
	current_index = 0		# 2022/03/09 20:48:52 记录已经解析到句法树字符串的位置
	stack = []				# 2022/03/09 20:48:52 用于存储句法树节点的栈, 栈顶元素记录当前所在分支的根节点
	node_id = -1			# 2022/04/23 16:09:19 全局记录节点的id, 考虑将节点信息存储在node.data中
	

	# 2022/04/28 23:29:29 合并后的添加节点函数
	def _add_node(_node_id, _text, _tag, _is_pos, _is_text):
		_node_data = {
			'node_id'	: _node_id,										# 2022/05/20 12:48:02 节点编号: 0, 1, 2, ...
			'text'		: _text,										# 2022/05/20 12:48:02 文本分词内容, 若_is_text为False, 则为None
			'tag_id'	: STANFORD_SYNTACTIC_TAG_INDEX.get(_tag, -1),	# 2022/05/20 12:48:02 句法树标签内容, 若_is_text为True, 则为None, 否则必然属于STANFORD_SYNTACTIC_TAG集合
			'is_pos'	: _is_pos,										# 2022/05/20 12:48:02 记录是否是句法树叶子节点上的词性标注
			'is_text'	: _is_text,										# 2022/05/20 12:48:02 记录是否是句法树叶子节点上的文本分词
		}
		graph.add_node(node_for_adding=_node_id, **_node_data)					# 添加新节点
		if stack:																# 若栈不为空
			_stack_top_node_id = stack[-1]['node_id']							# 取栈顶节点的编号
			graph.add_edge(u_of_edge=_stack_top_node_id, v_of_edge=_node_id)	# 则栈顶节点(即当前分支节点)指向新节点
		elif _is_text:
			raise Exception('Leaf node cannot find its parent !')
		if not _is_text:
			stack.append(_node_data)											# 最后将新节点(非叶/非文本)添加到栈中
	
	parse_tree_length = len(parse_tree)
	while current_index < parse_tree_length:
		# 左括号意味着新分支的开始
		if parse_tree[current_index] == '(':
			next_left_parenthese_index = parse_tree.find('(', current_index + 1)	# 寻找下一个左括号的位置
			next_right_parenthese_index = parse_tree.find(')', current_index + 1)	# 寻找下一个右括号的位置

			if next_left_parenthese_index == -1 and next_right_parenthese_index == -1:
				# 左括号后面一定还有括号
				raise Exception('There must be `)` or `(` after a `(` !')

			if next_left_parenthese_index < next_right_parenthese_index and next_left_parenthese_index >= 0:
				# 向右检索最先遇到左括号: 新节点出现
				new_node = parse_tree[current_index + 1: next_left_parenthese_index].replace(' ', '')	# 向右搜索先遇到左括号: 发现新节点

				# 2022/05/20 13:00:30 新增断言: 检索得到的新节点必然不是词性标注(叶子节点)
				assert new_node in STANFORD_SYNTACTIC_TAG and new_node not in STANFORD_POS_TAG, f'Unknown syntactic tags: {new_node}'

				node_id += 1																			# 更新节点编号
				_add_node(_node_id=node_id, _text=None, _tag=new_node, _is_pos=False, _is_text=False)	# 添加新节点
				current_index = next_left_parenthese_index												# 将current_index刷新到新的左括号处
			else:
				# 向右检索最先遇到右括号: 此时到达叶子节点
				leaf_node = parse_tree[current_index + 1: next_right_parenthese_index]					# 向右搜索先遇到右括号: 此时意味着已经到达叶子节点
				new_node, text = leaf_node.split(' ', 1)												# 叶子节点由词性标注与对应的文本内容两部分构成

				# 2022/05/20 13:00:30 新增断言: 检索得到的新节点必然是词性标注(叶子节点)
				assert new_node in STANFORD_POS_TAG, f'Unknown pos tags: {new_node}'

				node_id += 1																			# 更新节点编号
				_add_node(_node_id=node_id, _text=None, _tag=new_node, _is_pos=True, _is_text=False)	# 添加叶子节点
				if not ignore_text:
					node_id += 1																		# 更新节点编号
					_add_node(_node_id=node_id, _text=text, _tag=None, _is_pos=False, _is_text=True)	# 添加叶子节点上的文本内容
				current_index = next_right_parenthese_index + 1											# 将current_index刷新到右括号的下一个位置
				stack.pop(-1)																			# 弹出栈顶节点, 即叶子节点
		elif parse_tree[current_index] == ')':							# 右括号表示分支结束, 弹出栈顶节点(即当前分支的根节点)
			current_index += 1
			stack.pop(-1)
		elif parse_tree[current_index] == ' ':							# 空格则跳过
			current_index += 1
		else:															# 理论上不会出现其他情况, 除非字符串根本就不是一棵合法的句法树
			raise Exception(f'Illegal character: {parse_tree[current_index]}')

	if display:
		draw(graph, with_labels=True)
		plt.show()
	
	if return_type == 'networkx':
		return graph

	elif return_type == 'dgl':
		return dgl.from_networkx(graph, node_attrs=['node_id', 'tag_id',  'is_pos', 'is_text'])


# 2022/09/06 23:47:05 自底向上遍历句法树所有节点的函数
# 2022/09/11 15:24:01 应该需要记录每一个节点所在的层, 以及它在这一层的顺序编号
def traverse_parse_tree(parse_tree):
	"""2022/09/11 15:24:01 举个简单的输入输出示例:
	:param parse_tree		: (ROOT (IP (NP (CP (IP (NP (DNP (ADJP (ADVP (AD 将)) (ADJP (JJ 传统))) (DEG 的)) (NP (NN 篇目))) (VP (ADVP (AD 首次)) (VP (VV 改为) (NP (NP (NN 名例) (PU 、) (NN 吏) (PU 、) (NN 户) (PU 、) (NN 礼) (PU 、) (NN 兵)) (PU 、) (NP (NR 刑)) (PU 、) (NP (NN 工各律)))))) (DEC 的))) (VP (VC 是))))
	:return traverse_result: [
		[[ROOT]],
		[[IP]],
		[[PP, PU, NP, VP, PU]],
		[[P, NP], [，], [QP, NP], [ADVP, VP], [。]],
		[[根据], [NN, NN], [], [CD], [NN], [AD], [VV], []],
		[[], [新闻], [报导], [大部分], [中学生], [都], [近视]]
	]"""
	if not isinstance(parse_tree, str):
		# 2022/03/09 19:48:50 只对字符串形式的树进行处理, 原数据结构不便于处理
		warnings.warn(f'Expect `parse_tree` to be string, rather than {type(parse_tree)}')
		parse_tree = str(parse_tree)
	
	current_index = 0		# 2022/03/09 20:48:52 记录已经解析到句法树字符串的位置
	current_depth = -1		# 2022/09/11 15:28:38 记录当前树的深度, 每次遇到'('时会加1, 句法树的第一个字符必须是'('
	current_location = 0	# 2022/09/11 17:55:24 记录当前节点所在层的位置编号
	stack = []				# 2022/03/09 20:48:52 用于存储句法树节点属性, 栈顶元素记录的是当前所在分支的根节点
	node_id = -1			# 2022/04/23 16:09:19 全局记录节点的id
	node_id2data = {}		# 2022/09/11 15:28:38 记录节点属性, 比如node_id为0的节点属性为{'name': 'ROOT', 'depth': 0, 'location': 0, 'is_text': False}

	traverse_result = []	# 2022/09/11 15:32:03 算法返回结果
	
	def _padding_traverse_result(_current_depth, _current_location):
		# traverse_result第一重节点层数量检查
		if len(traverse_result) == _current_depth + 1:
			# 此时需要在traverse_result中新建下一个节点层
			traverse_result.append([])
		elif len(traverse_result) < _current_depth + 2:
			# 若`len(traverse_result)`不小于`current_depth + 2`, 则无需新建下一个节点层
			# 若`len(traverse_result)`小于`current_depth + 1`, 则之前漏建节点层, 说明算法逻辑有误
			raise Exception('Current depth should not be larger than `len(traverse_result) - 1`')	
		
		# tranvese_result第二重节点组数量检查
		if len(traverse_result[_current_depth + 1]) < _current_location + 2:
			# 此时需要在traverse_result的下一层中新建节点组
			# 注意与第一重节点层数量检查不同之处在于, 这里current_location可以远远超过`len(traverse_result[current_depth + 1])`, 不过需要新建空的节点组来补充缺少的部分
			for _ in range(_current_location - len(traverse_result[_current_depth + 1]) + 1):
				traverse_result[_current_depth + 1].append([])
				
	def _refresh_current_depth_and_current_location(_current_depth, _current_location):
		_new_current_depth = _current_depth + 1	
		_new_current_location = -1
		for _i in range(_current_location):
			_new_current_location += len(traverse_result[_new_current_depth][_i])
		_new_current_location += len(traverse_result[_new_current_depth][_current_location])
		return _new_current_depth, _new_current_location
	
	parse_tree_length = len(parse_tree)
	while current_index < parse_tree_length:
		# 左括号意味着新分支的开始
		if parse_tree[current_index] == '(':
			next_left_parenthese_index = parse_tree.find('(', current_index + 1)	# 寻找下一个左括号的位置
			next_right_parenthese_index = parse_tree.find(')', current_index + 1)	# 寻找下一个右括号的位置
			
			if next_left_parenthese_index == -1 and next_right_parenthese_index == -1:
				# 左括号后面一定还有括号
				raise Exception('There must be `)` or `(` after a `(` !')

			if next_left_parenthese_index < next_right_parenthese_index and next_left_parenthese_index >= 0:
				# 向右检索最先遇到左括号: 新节点出现
				node_id += 1																				# 更新节点编号	
				new_node = parse_tree[current_index + 1: next_left_parenthese_index].replace(' ', '')		# 向右搜索先遇到左括号: 发现新节点

				# 2022/05/20 13:00:30 新增断言: 检索得到的新节点必然不是词性标注(叶子节点)
				assert new_node in STANFORD_SYNTACTIC_TAG and new_node not in STANFORD_POS_TAG, f'Unknown syntactic tags: {new_node}'
				
				_padding_traverse_result(_current_depth=current_depth, _current_location=current_location)	# 填充traverse_result
				
				traverse_result[current_depth + 1][current_location].append(node_id)						# 将新节点插入到traverse_result中的正确位置上
				current_depth, current_location = _refresh_current_depth_and_current_location(_current_depth=current_depth, _current_location=current_location)
				
				node_data = {
					'name'		: new_node,
					'depth'		: current_depth,
					'location'	: current_location,
					'is_text'	: False,
				}
				stack.append(node_id)						# 更新栈顶元素
				node_id2data[node_id] = node_data			# 更新节点属性
				current_index = next_left_parenthese_index	# 将current_index刷新到新的左括号处
			else:
				# 向右检索最先遇到右括号: 此时到达叶子节点
				leaf_node = parse_tree[current_index + 1: next_right_parenthese_index]					# 向右搜索先遇到右括号: 此时意味着已经到达叶子节点
				new_node, text = leaf_node.split(' ', 1)												# 叶子节点由词性标注与对应的文本内容两部分构成

				# 2022/05/20 13:00:30 新增断言: 检索得到的新节点必然是词性标注(叶子节点)
				assert new_node in STANFORD_POS_TAG, f'Unknown pos tags: {new_node}'
				
				# 1. 处理词性标注
				node_id += 1																				# 更新节点编号
				_padding_traverse_result(_current_depth=current_depth, _current_location=current_location)	# 填充traverse_result
				
				traverse_result[current_depth + 1][current_location].append(node_id)						# 将新节点插入到traverse_result中的正确位置上
				current_depth, current_location = _refresh_current_depth_and_current_location(_current_depth=current_depth, _current_location=current_location)
				
				node_data = {
					'name'		: new_node,
					'depth'		: current_depth,
					'location'	: current_location,
					'is_text'	: False,
				}
				node_id2data[node_id] = node_data
				
				# 2. 处理文本内容
				node_id += 1	
				_padding_traverse_result(_current_depth=current_depth, _current_location=current_location)	# 填充traverse_result
				
				traverse_result[current_depth + 1][current_location].append(node_id)						# 将新节点插入到traverse_result中的正确位置上
				current_depth, current_location = _refresh_current_depth_and_current_location(_current_depth=current_depth, _current_location=current_location)
				
				node_data = {
					'name'		: text,
					'depth'		: current_depth,
					'location'	: current_location,
					'is_text'	: True,
				}
				node_id2data[node_id] = node_data
				current_index = next_right_parenthese_index + 1			# 将current_index刷新到右括号的下一个位置
				if stack:
					current_depth = node_id2data[stack[-1]]['depth']		# 更新current_depth为当前栈顶元素的树深
					current_location = node_id2data[stack[-1]]['location']	# 更新current_location为当前栈顶元素的位置编号
				else:
					assert current_index == parse_tree_length - 1, 'If stack is empty, end is reached .'

		elif parse_tree[current_index] == ')':							# 右括号表示分支结束, 弹出栈顶节点(即当前分支的根节点)
			current_index += 1
			stack.pop(-1)
			if stack:
				current_depth = node_id2data[stack[-1]]['depth']		# 更新current_depth为当前栈顶元素的树深
				current_location = node_id2data[stack[-1]]['location']	# 更新current_location为当前栈顶元素的位置编号
			else:
				assert current_index == parse_tree_length, f'If stack is empty, end is reached, but got `current_index={current_index}, parse_tree_length={parse_tree_length}`'
		elif parse_tree[current_index] == ' ':							# 空格则跳过
			current_index += 1
		else:															# 理论上不会出现其他情况, 除非字符串根本就不是一棵合法的句法树
			raise Exception(f'Illegal character: {parse_tree[current_index]}')
	
	return traverse_result, node_id2data
