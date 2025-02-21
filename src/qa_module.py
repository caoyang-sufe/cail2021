# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
# 问答模型的模块

if __name__ == '__main__':
	import sys
	sys.path.append('../')

import os
import dgl
import torch
import pandas

from copy import deepcopy

from dgl.nn import SAGEConv, HeteroGraphConv, GraphConv
from torch.nn import Module, Embedding, Linear, Dropout, RNN, LSTM, GRU, RNNCell, LSTMCell, GRUCell, BatchNorm1d, Sigmoid, CrossEntropyLoss, functional as F
from setting import *

class BaseLSTMEncoder(Module):
	"""基准LSTM编码器"""
	def __init__(self, d_hidden=128):
		super(BaseLSTMEncoder, self).__init__()
		self.d_hidden = d_hidden
		self.d_output = self.d_hidden
		self.is_bidirectional = True
		self.n_layers = 2
		if self.is_bidirectional:
			self.d_output = self.d_output // 2
		self.lstm = LSTM(input_size=self.d_hidden,
						 hidden_size=self.d_output,
						 num_layers=self.n_layers,
						 batch_first=True,
						 bidirectional=self.is_bidirectional)

	def forward(self, x):
		# 2022/02/21 12:52:11 模块输入x应当的dimension数为3, 第一个维度是batchsize, 第二个维度是序列长度seqlength, 第三个维度是序列中每个元素的编码embsize
		# 2022/02/28 15:58:10 输出结果hidden_output的形状与输入的x完全相同, max_hidden_output的dimension数为2, 相比x少了中间一唯, 即为(batchsize, embsize)
		batch_size = x.size()[0]
		initial_states = (torch.autograd.Variable(torch.zeros(self.n_layers + int(self.is_bidirectional) * self.n_layers, batch_size, self.d_output)).to(DEVICE),
						  torch.autograd.Variable(torch.zeros(self.n_layers + int(self.is_bidirectional) * self.n_layers, batch_size, self.d_output)).to(DEVICE))	# 这两个东西就是初始的h_0和c_0, LSTM中有
		hidden_output, final_states = self.lstm(x, initial_states)
		max_hidden_output = torch.max(hidden_output, dim=1)[0]
		return max_hidden_output, hidden_output


class BaseAttention(Module):
	"""基准注意力机制"""
	def __init__(self, d_hidden=128):
		super(BaseAttention, self).__init__()
		self.d_hidden = d_hidden
		self.linear = Linear(self.d_hidden, self.d_hidden)

	def forward(self, x, y):
		_x = self.linear(x)
		_y = torch.transpose(y, 1, 2)
		attention = torch.bmm(_x, _y)
		x_attention = torch.softmax(attention, dim=2)
		x_attention = torch.bmm(x_attention, y)
		y_attention = torch.softmax(attention, dim=1)
		y_attention = torch.bmm(torch.transpose(y_attention, 2, 1), x)
		return x_attention, y_attention, attention

class BaseAttentionPOS(Module):
	"""基准注意力机制: 带词性标注的版本"""
	def __init__(self, d_hidden=128):
		super(BaseAttentionPOS, self).__init__()
		self.d_hidden = d_hidden
		self.linear = Linear(self.d_hidden * 2, self.d_hidden)	# 改为*2是因为词性标注与分词进行拼接

	def forward(self, x, y):
		_x = self.linear(x)
		_y = torch.transpose(y, 1, 2)
		attention = torch.bmm(_x, _y)
		x_attention = torch.softmax(attention, dim=2)
		x_attention = torch.bmm(x_attention, y)
		y_attention = torch.softmax(attention, dim=1)
		y_attention = torch.bmm(torch.transpose(y_attention, 2, 1), x)
		return x_attention, y_attention, attention

class Doc2VecAttention(Module):
	"""用于Doc2Vec的注意力机制: 适用于将一句话表示为一个嵌入向量的情况"""
	def __init__(self, d_hidden=512):
		super(Doc2VecAttention, self).__init__()
		self.d_hidden = d_hidden
		self.linear1 = Linear(self.d_hidden, self.d_hidden)

	def forward(self, x, y):
		_x = self.linear(x)
		_y = torch.transpose(y, 1, 2)
		attention = torch.bmm(_x, _y)
		x_attention = torch.softmax(attention, dim=2)
		x_attention = torch.bmm(x_attention, y)
		y_attention = torch.softmax(attention, dim=1)
		y_attention = torch.bmm(torch.transpose(y_attention, 2, 1), x)
		return x_attention, y_attention, attention

class TreeRNNEncoder(Module):
	"""2022/08/26 11:50:47
	为每个句法树节点建立独立的RNN网络, 输入为句法树节点对应下面的若干分词或从句的表示, 维数为(n_words, input_size), 输出为(output_size)
	:param args: QAModelConfig类型的配置
		   args.tree_rnn_encoder_tag_name			: 节点名称, setting.py中STANFORD_SYNTACTIC_TAG中定义的名称
													  这里区分是否为STANFORD_POS_TAG(叶子节点)
													  叶子节点必然汇入非叶节点, 非叶节点也汇入非叶节点
													  因此叶子节点的输入输出维数可以不同, 非叶节点的输入输出维数必须相同
		   args.tree_rnn_encoder_rnn_type			: 使用的RNN编码器, 可选值为RNN, LSTM, GRU
		   args.tree_rnn_encoder_num_layers			: RNN编码器堆叠的层数
		   args.tree_rnn_encoder_bidirectional		: RNN编码器双向标志
		   args.tree_rnn_encoder_squeeze_strategy	: 压缩RNN输出的strategy, 可选值有mean(取均值), final(只取最后一层输出), fixed_weight(固定权重加权平均), variable_weight(可变权重加权平均, 即作为参数训练)"""
	def __init__(self, args, tag_name):	
		super(TreeRNNEncoder, self).__init__()
		self.args = deepcopy(args)
		self.tag_name = tag_name
		self.embedding_layer = None										# 当且仅当使用顺序编码值且tag_name为词性标注时, 可以存在嵌入曾
		
		# 句法树根节点
		if self.tag_name == 'ROOT':						
			# tree_rnn_encoder_node_hidden_size -> tree_rnn_encoder_root_output_size
			self.input_size = self.args.tree_rnn_encoder_node_hidden_size
			self.output_size = self.args.tree_rnn_encoder_root_output_size
			self.sequence_encoder = eval(self.args.tree_rnn_encoder_rnn_type)(input_size	= self.input_size, 
																			  hidden_size	= int(self.output_size / (1 + self.args.tree_rnn_encoder_bidirectional)),	# 双向RNN的输出维数是hidden_size的2倍
																			  num_layers	= self.args.tree_rnn_encoder_num_layers,
																			  batch_first	= True, 
																			  bidirectional	= self.args.tree_rnn_encoder_bidirectional)
			
		# 词性标注: 即叶子节点, 使用DNN编码器
		elif self.tag_name in STANFORD_POS_TAG:		
			# embedding_size -> tree_rnn_encoder_node_hidden_size				
			if self.args.document_embedding is not None:
				# 文档嵌入: 禁用, 句法树模型只能使用词嵌入
				raise Exception('Please use word embedding rather than document model !')
			elif self.args.word_embedding is None:
				# 顺序编码值嵌入: 需要建立嵌入层
				self.input_size = self.args.default_embedding_size
				self.embedding_layer = Embedding(num_embeddings=pandas.read_csv(REFERENCE_TOKEN2ID_PATH, sep='\t', header=0).shape[0], 
												 embedding_dim=self.input_size)
			elif self.args.word_embedding == 'word2vec':
				# word2vec词嵌入
				self.input_size = self.args.size_word2vec			
			elif self.args.word_embedding == 'fasttext':
				# fasttext词嵌入
				self.input_size = self.args.size_fasttext
			else:
				raise Exception(f'Unknown word embedding: {self.args.word_embedding}')
			self.output_size = self.args.tree_rnn_encoder_node_hidden_size	
			self.sequence_encoder = Linear(in_features=self.input_size, 
										   out_features=self.output_size, 
										   bias=True)

		# 句法结构标注: 即非叶节点, 使用RNN编码器
		elif self.tag_name in STANFORD_SYNTACTIC_TAG:		
			# tree_rnn_encoder_node_hidden_size -> tree_rnn_encoder_node_hidden_size
			self.input_size = self.args.tree_rnn_encoder_node_hidden_size	
			self.output_size = self.args.tree_rnn_encoder_node_hidden_size	
			self.sequence_encoder = eval(self.args.tree_rnn_encoder_rnn_type)(input_size	= self.input_size, 
																			  hidden_size	= int(self.output_size / (1 + self.args.tree_rnn_encoder_bidirectional)),	# 双向RNN的输出维数是hidden_size的2倍
																			  num_layers	= self.args.tree_rnn_encoder_num_layers,
																			  batch_first	= True, 
																			  bidirectional	= self.args.tree_rnn_encoder_bidirectional)
		else: 
			raise Exception(f'Unknown syntactic tag: {tag_name}')
								
		# 2022/09/05 22:12:01 这里我想的是可以考虑在variable_weight的情况下引入一个Parameter类型的一维权重张量self.squeeze_weight
		# 2022/09/05 22:12:01 但是需要确保该张量的元素和为1, 这比较麻烦, 或许可以在损失函数中引入self.squeeze_weight的正则项, 但是这太麻烦了
		# 2022/09/05 22:12:01 于是我决定暂时不实现这种情况了
		if self.args.tree_rnn_encoder_squeeze_strategy in ['mean', 'final']:
			self.args.squeeze_weight = None
		elif self.args.tree_rnn_encoder_squeeze_strategy == 'fixed_weight':
			# 2022/09/06 14:21:45 默认的固定权重定义为等比数列1/2, 1/4, 1/8, ...
			self.squeeze_weight = torch.Parameter(torch.FloatTensor([2 ** (-i) for i in range(self.args.default_max_child)]), requires_grad=False)
		elif self.args.tree_rnn_encoder_squeeze_strategy == 'variable_weight':
			# 2022/09/06 14:21:45 即将上面默认的固定权重改为可修改(requires_grad=True)
			# self.squeeze_weight = torch.Parameter(torch.FloatTensor([2 ** (-i) for i in range(1, self.args.default_max_child)]), requires_grad=True)
			raise NotImplementedError
		else:
			raise Exception(f'Unknown squeeze_strategy: {self.squeeze_strategy}')
			
		# 2022/09/20 14:11:43 对输入的x进行标准化
		self.normalization_layer = BatchNorm1d(num_features=self.input_size)
		
	def forward(self, x):
		"""2022/09/03 15:58:30
		前馈逻辑: 
		:param x: (batch_size, seq_len, input_size), 目前感觉batchsize或许只能取1"""
		
		assert x.shape[0] == 1, 'Currently we only consider batch 1 case'
		

		
		# 需要进行嵌入的情况: (batch_size, seq_len) -> (batch_size, seq_len, input_size)
		if self.embedding_layer is not None:			
			x = self.embedding_layer(x)

		elif x.shape[1] > 1:
			# 如果序列长度超过1, 考虑对输入张量进行标准化
			# 使用嵌入层则不做标准化
			x = self.normalization_layer(x.squeeze(0)).unsqueeze(0)

		# (batch_size, seq_len, input_size) -> (batch_size, seq_len, (1 + bidirectional) * hidden_size)
		if self.tag_name in STANFORD_POS_TAG:		
			y1 = self.sequence_encoder(x)		
		elif self.tag_name in STANFORD_SYNTACTIC_TAG:
			y1, _ = self.sequence_encoder(x)	
		else:
			raise Exception(f'Unknown word embedding: {self.args.word_embedding}')
		
		# (batch_size, seq_len, (1 + bidirectional) * hidden_size) -> (batch_size, (1 + bidirectional) * hidden_size)
		if self.args.tree_rnn_encoder_squeeze_strategy == 'mean':
			y2 = torch.mean(y1, axis=1)			
		elif self.args.tree_rnn_encoder_squeeze_strategy == 'final':
			y2 = y1[:, -1, :]
		elif self.args.tree_rnn_encoder_squeeze_strategy == 'fixed_weight':
			sequence_length = x.shape[1]
			y2 = y1[:, 0, :] * self.squeeze_weight[sequence_length - 2]	# 初始位置的权重跟下一个位置的权重是相同的, 即1/2, 1/4, 1/8, 1/8
			for i in range(1, sequence_length):
				y2 += y1[: -i, :] * self.squeeze_weight[i - 1]
		elif self.args.tree_rnn_encoder_squeeze_strategy == 'variable_weight':
			# 2022/09/06 22:41:04 实现逻辑与fixed_weight的情况是一样的
			raise NotImplementedError
		else:
			raise Exception(f'Unknown squeeze_strategy: {self.squeeze_strategy}')

		return y2

	
	

