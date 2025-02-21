# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
# 用于解题的问答模型

if __name__ == '__main__':
	import sys
	sys.path.append('../')

import torch
import pandas

from copy import deepcopy
from torch.nn import Module, Embedding, Linear, LSTM, RNN, GRU, Dropout, Sigmoid, CrossEntropyLoss, functional as F

from setting import *

from src.data_tools import encode_answer, decode_answer
from src.qa_module import BaseLSTMEncoder, BaseAttention, Doc2VecAttention, BaseAttentionPOS
from src.utils import load_args, timer


class BaseChoiceModel(Module):
	"""选择题Baseline模型: 不使用参考文献"""
	def __init__(self, args):
		super(BaseChoiceModel, self).__init__()
		self.d_hidden = args.default_embedding_size
		self.n_tokens = pandas.read_csv(REFERENCE_TOKEN2ID_PATH, sep='\t', header=0).shape[0]
		self.options_embedding = Embedding(self.n_tokens, self.d_hidden)
		self.question_embedding = Embedding(self.n_tokens, self.d_hidden)
		self.options_encoder = BaseLSTMEncoder(d_hidden=self.d_hidden)
		self.question_encoder = BaseLSTMEncoder(d_hidden=self.d_hidden)
		self.attention = BaseAttention(d_hidden=self.d_hidden)
		self.rank_module = Linear(self.d_hidden * 2, 1)
		self.multi_module = Linear(4, 16)

	def forward(self, data):
		options = data['options']
		question = data['question']
		batch_size = question.size()[0]
		n_options = TOTAL_OPTIONS
		embedded_options = self.options_embedding(options.view(batch_size * n_options, -1))
		embedded_question = self.question_embedding(torch.cat([question.view(batch_size, -1) for _ in range(n_options)]))	# 扩展问题的维度与选项相同
		_, encoded_options = self.options_encoder(embedded_options)
		_, encoded_question = self.question_encoder(embedded_question)
		options_attention, question_attention, attention = self.attention(encoded_options, encoded_question)
		y = torch.cat([torch.max(options_attention, dim=1)[0], torch.max(question_attention, dim=1)[0]], dim=1)
		y = self.rank_module(y.view(batch_size * n_options, -1))
		output = self.multi_module(y.view(batch_size, n_options))
		return output


class BaseChoiceModelPOS(Module):
	"""选择题Baseline模型(使用词性标注): 不使用参考文献"""
	def __init__(self, args):
		super(BaseChoiceModelPOS, self).__init__()
		self.d_hidden = args.default_embedding_size
		self.n_tokens = pandas.read_csv(REFERENCE_TOKEN2ID_PATH, sep='\t', header=0).shape[0]

		self.options_embedding = Embedding(self.n_tokens, self.d_hidden)
		self.questions_embedding = Embedding(self.n_tokens, self.d_hidden)
		self.pos_tags_embedding = Embedding(len(STANFORD_POS_TAG) + 1, self.d_hidden)	# +1是因为有padding

		self.options_encoder = BaseLSTMEncoder(d_hidden=self.d_hidden)
		self.question_encoder = BaseLSTMEncoder(d_hidden=self.d_hidden)
		self.pos_tags_encoder = BaseLSTMEncoder(d_hidden=self.d_hidden)
		self.attention = BaseAttentionPOS(d_hidden=self.d_hidden)
		self.rank_module = Linear(self.d_hidden * 3, 1)
		self.multi_module = Linear(4, 16)

	def forward(self, data):
		options = data['options']	# [32, 4, 128]
		question = data['question']	# [32, 256]
		option_a_pos_tags = data['option_a_pos_tags']	# [32, 128]
		option_b_pos_tags = data['option_b_pos_tags']	# [32, 128]
		option_c_pos_tags = data['option_c_pos_tags']	# [32, 128]
		option_d_pos_tags = data['option_d_pos_tags']	# [32, 128]

		pos_tags = torch.stack([option_a_pos_tags, option_b_pos_tags, option_c_pos_tags, option_d_pos_tags])	# [4, 32, 128]
		pos_tags = pos_tags.permute(1, 0, 2)	# [32, 4, 128]

		batch_size = question.size()[0]
		n_options = TOTAL_OPTIONS
		
		embedded_options = self.options_embedding(options.view(batch_size * n_options, -1))
		embedded_question = self.questions_embedding(torch.cat([question.view(batch_size, -1) for _ in range(n_options)]))	# 扩展问题的维度与选项相同
		# embedded_pos_tags = self.pos_tags_embedding(pos_tags.view(batch_size * n_options, -1))	# 无法完成, view只能用于连续存储的情况, 此处只能使用reshape
		embedded_pos_tags = self.pos_tags_embedding(pos_tags.reshape(batch_size * n_options, -1))

		_, encoded_options = self.options_encoder(embedded_options)		# [128, 128, 128]
		_, encoded_question = self.question_encoder(embedded_question)	# [128, 256, 128]
		_, encoded_pos_tags = self.pos_tags_encoder(embedded_pos_tags)	# [128, 128, 128]

		encoded_options_and_pos_tags = torch.cat([encoded_options, encoded_pos_tags], axis=-1)	# [128, 128, 256]

		options_and_pos_tags_attention, question_attention, attention = self.attention(encoded_options_and_pos_tags, encoded_question)	# [128, 128, 128], [128, 256, 256]
		y = torch.cat([torch.max(options_and_pos_tags_attention, dim=1)[0], torch.max(question_attention, dim=1)[0]], dim=1)	# [128, 384] (这个在正常情况下是[128, 256])

		y = self.rank_module(y.view(batch_size * n_options, -1))
		output = self.multi_module(y.view(batch_size, n_options))
		return output


class BaseJudgmentModel(Module):
	"""判断题Baseline模型: 不使用参考文献"""
	def __init__(self, args):
		super(BaseJudgmentModel, self).__init__()
		self.d_hidden = args.default_embedding_size
		self.n_tokens = pandas.read_csv(REFERENCE_TOKEN2ID_PATH, sep='\t', header=0).shape[0]
		self.embedding = Embedding(self.n_tokens, self.d_hidden)
		self.options_encoder = BaseLSTMEncoder(d_hidden=self.d_hidden)
		self.question_encoder = BaseLSTMEncoder(d_hidden=self.d_hidden)
		self.attention = BaseAttention(d_hidden=self.d_hidden)
		self.rank_module = Linear(self.d_hidden * 2, 1)
		self.activation_function = Sigmoid()

	def forward(self, data):
		option = data['option']			# [32, 4, 128]
		question = data['question']		# [32, 256]
		batch_size = question.size()[0]	# 32
		embedded_option = self.embedding(option.view(batch_size, -1))
		embedded_question = self.embedding(question.view(batch_size, -1))
		_, encoded_option = self.options_encoder(embedded_option)
		_, encoded_question = self.question_encoder(embedded_question)
		option_attention, question_attention, attention = self.attention(encoded_option, encoded_question)
		y = torch.cat([torch.max(option_attention, dim=1)[0], torch.max(question_attention, dim=1)[0]], dim=1)
		y = self.rank_module(y.view(batch_size, -1))
		output = self.activation_function(y).squeeze(-1)				# 如果不squeeze输出结果形如[[.9], [.8], [.5]], 希望得到形如[.9, .8, .5]的输出结果
		return output


class ReferenceChoiceModel(Module):
	"""选择题Baseline模型: 使用参考文献"""
	def __init__(self, args):
		super(ReferenceChoiceModel, self).__init__()
		self.d_hidden = args.default_embedding_size
		self.n_tokens = pandas.read_csv(REFERENCE_TOKEN2ID_PATH, sep='\t', header=0).shape[0]
		self.embedding = Embedding(self.n_tokens, self.d_hidden)
		self.options_encoder = BaseLSTMEncoder(d_hidden=self.d_hidden)
		self.question_encoder = BaseLSTMEncoder(d_hidden=self.d_hidden)
		self.reference_encoder = BaseLSTMEncoder(d_hidden=self.d_hidden)
		self.attention = BaseAttention(d_hidden=self.d_hidden)
		self.rank_module = Linear(64, 1)
		self.multi_module = Linear(4, 16)

	def forward(self, data):
		options = data['options']
		question = data['question']
		reference = data['reference']
		batch_size = question.size()[0]
		n_options = TOTAL_OPTIONS


		# torch.Size([32, 4, 128])
		# torch.Size([32, 256])
		# torch.Size([32, 32, 512])
		# torch.Size([32, 512, 128])
		# torch.Size([32, 256, 128])
		# torch.Size([32, 16384, 128])

		# print('options', options.shape)
		# print('question', question.shape)
		# print('reference', reference.shape)

		embedded_options = self.embedding(options.view(batch_size, -1))
		embedded_question = self.embedding(question.view(batch_size, -1))
		embedded_reference = self.embedding(reference.view(batch_size, -1))

		# print('embedded_options', embedded_options.shape)
		# print('embedded_question', embedded_question.shape)
		# print('embedded_reference', embedded_reference.shape)

		embedded_options_and_question = torch.cat([embedded_options, embedded_question], axis=1)

		# print('embedded_options_and_question', embedded_options_and_question.shape)

		_, encoded_options_and_question = self.options_encoder(embedded_options_and_question)

		# print('encoded_options_and_question', encoded_options_and_question.shape)
		_, encoded_reference = self.reference_encoder(embedded_reference)

		# print('encoded_reference', encoded_reference.shape)

		options_and_question_attention, reference_attention, attention = self.attention(encoded_options_and_question, encoded_reference)

		# print('options_and_question_attention', options_and_question_attention.shape)
		# print('reference_attention', reference_attention.shape)
		# print('attention', attention.shape)

		y = torch.cat([torch.max(options_and_question_attention, dim=1)[0], torch.max(reference_attention, dim=1)[0]], dim=1)

		# print('y', y.shape)

		y = self.rank_module(y.view(batch_size * n_options, -1))

		# print('y', y.shape)

		# options torch.Size([2, 4, 128])
		# question torch.Size([2, 256])
		# reference torch.Size([2, 32, 512])
		# embedded_options torch.Size([2, 512, 128])
		# embedded_question torch.Size([2, 256, 128])
		# embedded_reference torch.Size([2, 16384, 128])
		# embedded_options_and_question torch.Size([2, 768, 128])
		# encoded_options_and_question torch.Size([2, 768, 128])
		# encoded_reference torch.Size([2, 16384, 128])
		# options_and_question_attention torch.Size([2, 768, 128])
		# reference_attention torch.Size([2, 16384, 128])
		# attention torch.Size([2, 768, 16384])
		# y torch.Size([2, 256])
		# y torch.Size([8, 1])

		output = self.multi_module(y.view(batch_size, n_options))
		return output


class ReferenceChoiceModelPOS(Module):
	"""选择题Baseline模型: 使用参考文献"""
	def __init__(self, args):
		super(ReferenceChoiceModelPOS, self).__init__()
		self.d_hidden = args.default_embedding_size
		self.n_tokens = pandas.read_csv(REFERENCE_TOKEN2ID_PATH, sep='\t', header=0).shape[0]
		self.embedding = Embedding(self.n_tokens, self.d_hidden)
		self.options_encoder = BaseLSTMEncoder(d_hidden=self.d_hidden)
		self.question_encoder = BaseLSTMEncoder(d_hidden=self.d_hidden)
		self.reference_encoder = BaseLSTMEncoder(d_hidden=self.d_hidden)
		self.attention = BaseAttention(d_hidden=self.d_hidden)
		self.rank_module = Linear(64, 1)
		self.multi_module = Linear(4, 16)

	def forward(self, data):
		options = data['options']		# [32, 4, 128]
		question = data['question']		# [32, 256]

		option_a_pos_tags = data['option_a_pos_tags']	# [32, 128]
		option_b_pos_tags = data['option_b_pos_tags']	# [32, 128]
		option_c_pos_tags = data['option_c_pos_tags']	# [32, 128]
		option_d_pos_tags = data['option_d_pos_tags']	# [32, 128]

		reference = data['reference']	# [32, 3*6, 512]
		batch_size = question.size()[0]	# 32
		n_options = TOTAL_OPTIONS		# 4


		pos_tags = torch.stack([option_a_pos_tags, option_b_pos_tags, option_c_pos_tags, option_d_pos_tags])	# [4, 32, 128]
		pos_tags = pos_tags.permute(1, 0, 2)																	# [32, 4, 128]


		embedded_options = self.embedding(options.view(batch_size, -1))
		embedded_question = self.embedding(question.view(batch_size, -1))
		embedded_pos_tags = self.embedding(pos_tags.reshape(batch_size, -1))
		embedded_reference = self.embedding(reference.view(batch_size, -1))


		embedded_options_and_question = torch.cat([embedded_options, embedded_pos_tags, embedded_question], axis=1)

		_, encoded_options_and_question = self.options_encoder(embedded_options_and_question)

		_, encoded_reference = self.reference_encoder(embedded_reference)

		options_and_question_attention, reference_attention, attention = self.attention(encoded_options_and_question, encoded_reference)

		y = torch.cat([torch.max(options_and_question_attention, dim=1)[0], torch.max(reference_attention, dim=1)[0]], dim=1)

		y = self.rank_module(y.view(batch_size * n_options, -1))

		# options torch.Size([2, 4, 128])
		# question torch.Size([2, 256])
		# reference torch.Size([2, 32, 512])
		# embedded_options torch.Size([2, 512, 128])
		# embedded_question torch.Size([2, 256, 128])
		# embedded_reference torch.Size([2, 16384, 128])
		# embedded_options_and_question torch.Size([2, 768, 128])
		# encoded_options_and_question torch.Size([2, 768, 128])
		# encoded_reference torch.Size([2, 16384, 128])
		# options_and_question_attention torch.Size([2, 768, 128])
		# reference_attention torch.Size([2, 16384, 128])
		# attention torch.Size([2, 768, 16384])
		# y torch.Size([2, 256])
		# y torch.Size([8, 1])

		output = self.multi_module(y.view(batch_size, n_options))
		return output


class ReferenceJudgmentModel(Module):
	"""选择题Baseline模型: 使用参考文献"""
	def __init__(self, args):
		super(ReferenceJudgmentModel, self).__init__()
		self.d_hidden = args.default_embedding_size
		self.n_tokens = pandas.read_csv(REFERENCE_TOKEN2ID_PATH, sep='\t', header=0).shape[0]
		self.embedding = Embedding(self.n_tokens, self.d_hidden)
		self.option_encoder = BaseLSTMEncoder(d_hidden=self.d_hidden)
		self.question_encoder = BaseLSTMEncoder(d_hidden=self.d_hidden)
		self.reference_encoder = BaseLSTMEncoder(d_hidden=self.d_hidden)
		self.attention = BaseAttention(d_hidden=self.d_hidden)
		self.rank_module = Linear(self.d_hidden * 2, 1)
		self.activation_function = Sigmoid()

	def forward(self, data):
		option = data['option']
		question = data['question']
		reference = data['reference']
		batch_size = question.size()[0]
		n_option = TOTAL_OPTIONS

		# print('option', option.shape)
		# print('question', question.shape)
		# print('reference', reference.shape)

		embedded_option = self.embedding(option.view(batch_size, -1))
		embedded_question = self.embedding(question.view(batch_size, -1))
		embedded_reference = self.embedding(reference.view(batch_size, -1))

		# print('embedded_option', embedded_option.shape)
		# print('embedded_question', embedded_question.shape)
		# print('embedded_reference', embedded_reference.shape)

		embedded_option_and_question = torch.cat([embedded_option, embedded_question], axis=1)

		# print('embedded_option_and_question', embedded_option_and_question.shape)

		_, encoded_option_and_question = self.option_encoder(embedded_option_and_question)

		# print('encoded_option_and_question', encoded_option_and_question.shape)
		_, encoded_reference = self.reference_encoder(embedded_reference)

		# print('encoded_reference', encoded_reference.shape)

		option_and_question_attention, reference_attention, attention = self.attention(encoded_option_and_question, encoded_reference)

		# print('option_and_question_attention', option_and_question_attention.shape)
		# print('reference_attention', reference_attention.shape)
		# print('attention', attention.shape)

		y = torch.cat([torch.max(option_and_question_attention, dim=1)[0], torch.max(reference_attention, dim=1)[0]], dim=1)

		# print('y', y.shape)

		y = self.rank_module(y.view(batch_size, -1))

		# print('y', y.shape)

		output = self.activation_function(y).squeeze(-1)
		return output


class Doc2VecChoiceModel(Module):
	"""选择题Baseline模型: 不使用参考文献, 且使用预训练的文档嵌入,
	   注意由于文档嵌入没有顺序, 因此不使用RNN"""
	def __init__(self, args):
		super(Doc2VecChoiceModel, self).__init__()
		self.d_hidden = args.size_doc2vec
		self.attention = Doc2VecAttention(d_hidden=self.d_hidden)
		self.rank_module = Linear(int(self.d_hidden / 2), 1)
		self.multi_module = Linear(4, 16)

	def forward(self, data):
		options = data['options']	# [32, 4, 512]
		question = data['question']	# [32, 512]

		batch_size = question.size()[0]
		n_options = TOTAL_OPTIONS

		expanded_question = torch.cat([question for _ in range(4)]).view(batch_size, n_options, -1)	# [32, 4, 512]
		
		options_attention, question_attention, attention = self.attention(options, expanded_question)

		# print(f'options_attention: {options_attention.shape}')
		# print(f'question_attention: {question_attention.shape}')
		# print(f'attention: {attention.shape}')

		y = torch.cat([torch.max(options_attention, dim=1)[0], torch.max(question_attention, dim=1)[0]], dim=1)

		# print(f'y_cat: {y.shape}')

		y = self.rank_module(y.view(batch_size * n_options, -1))

		# print(f'y_rank: {y.shape}')

		output = self.multi_module(y.view(batch_size, n_options))

		# options_attention: torch.Size([32, 4, 512])
		# question_attention: torch.Size([32, 4, 512])
		# attention: torch.Size([32, 4, 4])
		# y_cat: torch.Size([32, 1024])

		return output


class BertChoiceModel(Module):
	"""选择题Baseline模型: 不使用参考文献, 且使用预训练的BERT嵌入,
	   注意由于BERT仍属于文档嵌入, 且文档嵌入没有顺序, 因此不使用RNN"""
	def __init__(self, args):
		super(BertChoiceModel, self).__init__()
		self.d_hidden = args.bert_hidden_size
		self.attention = Doc2VecAttention(d_hidden=self.d_hidden)
		self.rank_module = Linear(int(self.d_hidden / 2), 1)
		self.multi_module = Linear(4, 16)

	def forward(self, data):
		options = data['options']	# [32, 4, 768]
		question = data['question']	# [32, 768]

		batch_size = question.size()[0]
		n_options = TOTAL_OPTIONS

		expanded_question = torch.cat([question for _ in range(4)]).view(batch_size, n_options, -1)	# [32, 4, 512]

		options_attention, question_attention, attention = self.attention(options, expanded_question)

		# print(f'options_attention: {options_attention.shape}')
		# print(f'question_attention: {question_attention.shape}')
		# print(f'attention: {attention.shape}')

		y = torch.cat([torch.max(options_attention, dim=1)[0], torch.max(question_attention, dim=1)[0]], dim=1)

		# print(f'y_cat: {y.shape}')

		y = self.rank_module(y.view(batch_size * n_options, -1))

		# print(f'y_rank: {y.shape}')

		output = self.multi_module(y.view(batch_size, n_options))

		# options_attention: torch.Size([32, 4, 768])
		# question_attention: torch.Size([32, 4, 768])
		# attention: torch.Size([32, 4, 4])
		# y_cat: torch.Size([32, 1536])

		return output


class BertChoiceModelA(Module):
	"""选择题Baseline模型: 不使用参考文献, 且使用预训练的BERT嵌入,
	改版A直接使用全连接层来搞, 调了一下dropout的比率
	"""
	def __init__(self, args):
		super(BertChoiceModelA, self).__init__()
		self.d_hidden = 512
		self.linear_a = Linear(args.bert_hidden_size * 2, self.d_hidden)
		self.linear_b = Linear(args.bert_hidden_size * 2, self.d_hidden)
		self.linear_c = Linear(args.bert_hidden_size * 2, self.d_hidden)
		self.linear_d = Linear(args.bert_hidden_size * 2, self.d_hidden)

		self.linear_a1 = Linear(self.d_hidden, self.d_hidden)
		self.linear_b1 = Linear(self.d_hidden, self.d_hidden)
		self.linear_c1 = Linear(self.d_hidden, self.d_hidden)
		self.linear_d1 = Linear(self.d_hidden, self.d_hidden)

		self.rank_module = Linear(self.d_hidden * 4, 4)
		self.multi_module = Linear(4, 16)

		self.dropout_a = Dropout(p=0.5)
		self.dropout_b = Dropout(p=0.5)
		self.dropout_c = Dropout(p=0.5)
		self.dropout_d = Dropout(p=0.5)

		self.dropout_a1 = Dropout(p=0.5)
		self.dropout_b1 = Dropout(p=0.5)
		self.dropout_c1 = Dropout(p=0.5)
		self.dropout_d1 = Dropout(p=0.5)

	def forward(self, data):
		options = data['options']	# [32, 4, 768]
		question = data['question']	# [32, 768]

		batch_size = options.size()[0]
		n_options = options.size()[1]
		assert n_options == TOTAL_OPTIONS

		option_a = options[:, 0, :].squeeze(1)	# [32, 768]
		option_b = options[:, 1, :].squeeze(1)	# [32, 768]
		option_c = options[:, 2, :].squeeze(1)	# [32, 768]
		option_d = options[:, 3, :].squeeze(1)	# [32, 768]

		hidden_a = self.linear_a(torch.hstack([option_a, question]))	# [32, 1536] -> [32, 512]
		hidden_a = self.dropout_a(hidden_a)
		hidden_a = self.linear_a1(hidden_a)								# [32, 512] -> [32, 512]
		hidden_a = self.dropout_a1(hidden_a)


		hidden_b = self.linear_b(torch.hstack([option_b, question]))	# [32, 1536] -> [32, 512]
		hidden_b = self.dropout_b(hidden_b)
		hidden_b = self.linear_b1(hidden_b)								# [32, 512] -> [32, 512]
		hidden_b = self.dropout_b1(hidden_b)

		hidden_c = self.linear_c(torch.hstack([option_c, question]))	# [32, 1536] -> [32, 512]
		hidden_c = self.dropout_c(hidden_c)
		hidden_c = self.linear_c1(hidden_c)								# [32, 512] -> [32, 512]
		hidden_c = self.dropout_c1(hidden_c)

		hidden_d = self.linear_d(torch.hstack([option_d, question]))	# [32, 1536] -> [32, 512]
		hidden_d = self.dropout_d(hidden_d)
		hidden_d = self.linear_d1(hidden_d)								# [32, 512] -> [32, 512]
		hidden_d = self.dropout_d1(hidden_d)

		hidden = torch.hstack([hidden_a, hidden_b, hidden_c, hidden_d])	# [32, 512 * 4]

		y = self.rank_module(hidden)									# [32, 512 * 4] -> [32, 4]
		y = self.multi_module(y)										# [32, 4] -> [32, 16]

		return F.softmax(y)


class Word2VecChoiceModel(Module):
	"""选择题Baseline模型: 不使用参考文献, 且使用预训练的文档嵌入"""
	def __init__(self, args):
		super(Word2VecChoiceModel, self).__init__()
		if args.word_embedding == 'word2vec':
			self.d_hidden = args.size_word2vec
		elif args.word_embedding == 'fasttext':
			self.d_hidden = args.size_fasttext
		else:
			raise NotImplementedError
		self.options_encoder = BaseLSTMEncoder(d_hidden=self.d_hidden)
		self.question_encoder = BaseLSTMEncoder(d_hidden=self.d_hidden)
		self.attention = BaseAttention(d_hidden=self.d_hidden)
		self.rank_module = Linear(int(self.d_hidden / 2), 1)
		self.multi_module = Linear(4, 16)

	def forward(self, data):
		options = data['options']	# [32, 4, 128, 256]
		question = data['question']	# [32, 256, 256]

		batch_size = question.size()[0]
		n_options = TOTAL_OPTIONS

		expanded_question = torch.cat([question for _ in range(4)]).view(batch_size, n_options, -1, self.d_hidden)	# [32, 4, 256, 256]

		_, encoded_options = self.options_encoder(options.view(batch_size, -1, self.d_hidden))
		_, encoded_question = self.question_encoder(expanded_question.view(batch_size, -1, self.d_hidden))
		options_attention, question_attention, attention = self.attention(encoded_options, encoded_question)
		y = torch.cat([torch.max(options_attention, dim=1)[0], torch.max(question_attention, dim=1)[0]], dim=1)

		# print(f'options_attention: {options_attention.shape}')
		# print(f'question_attention: {question_attention.shape}')
		# print(f'attention: {attention.shape}')
		# print(f'y_cat: {y.shape}')

		# options_attention: torch.Size([32, 512, 256])
		# question_attention: torch.Size([32, 1024, 256])
		# attention: torch.Size([32, 512, 1024])
		# y_cat: torch.Size([32, 512])

		y = self.rank_module(y.view(batch_size * n_options, -1))
		output = self.multi_module(y.view(batch_size, n_options))
		return output


class ReferenceDoc2VecChoiceModel(Module):
	"""选择题Baseline模型: 使用参考文献与文档编码嵌入"""
	def __init__(self, args):
		super(ReferenceDoc2VecChoiceModel, self).__init__()
		self.d_hidden = args.size_doc2vec
		self.attention = Doc2VecAttention(d_hidden=self.d_hidden)
		self.rank_module = Linear(int(self.d_hidden / 2), 1)
		self.multi_module = Linear(4, 16)

	def forward(self, data):
		options = data['options']		# [32, 4, 512]
		question = data['question']		# [32, 512]
		reference = data['reference']	# [32, 12, 512]

		batch_size = question.size()[0]
		n_options = TOTAL_OPTIONS

		expanded_question = torch.cat([question for _ in range(4)]).view(batch_size, n_options, -1)	# [32, 4, 512]

		options_attention, _, _ = self.attention(options, reference)
		question_attention, _, _ = self.attention(expanded_question, reference)

		# print(f'options_attention: {options_attention.shape}')
		# print(f'question_attention: {question_attention.shape}')

		y = torch.cat([torch.max(options_attention, dim=1)[0], torch.max(question_attention, dim=1)[0]], dim=1)

		# print(f'y_cat: {y.shape}')

		y = self.rank_module(y.view(batch_size * n_options, -1))

		# print(f'y_rank: {y.shape}')

		output = self.multi_module(y.view(batch_size, n_options))

		# options_attention: torch.Size([2, 4, 512])
		# question_attention: torch.Size([2, 4, 512])
		# y_cat: torch.Size([2, 1024])
		# y_rank: torch.Size([8, 1])

		return output


class ReferenceBertChoiceModel(Module):
	"""选择题Baseline模型: 使用参考文献与文档编码嵌入"""
	def __init__(self, args):
		super(ReferenceBertChoiceModel, self).__init__()
		self.d_hidden = args.bert_hidden_size
		self.attention = Doc2VecAttention(d_hidden=self.d_hidden)
		self.rank_module = Linear(int(self.d_hidden / 2), 1)
		self.multi_module = Linear(4, 16)

	def forward(self, data):
		options = data['options']		# [32, 4, 768]
		question = data['question']		# [32, 768]
		reference = data['reference']	# [32, 12, 768]

		batch_size = question.size()[0]
		n_options = TOTAL_OPTIONS

		expanded_question = torch.cat([question for _ in range(4)]).view(batch_size, n_options, -1)	# [32, 4, 512]

		options_attention, _, _ = self.attention(options, reference)
		question_attention, _, _ = self.attention(expanded_question, reference)

		# print(f'options_attention: {options_attention.shape}')
		# print(f'question_attention: {question_attention.shape}')

		y = torch.cat([torch.max(options_attention, dim=1)[0], torch.max(question_attention, dim=1)[0]], dim=1)

		# print(f'y_cat: {y.shape}')

		y = self.rank_module(y.view(batch_size * n_options, -1))

		# print(f'y_rank: {y.shape}')

		output = self.multi_module(y.view(batch_size, n_options))

		# options_attention: torch.Size([2, 4, 512])
		# question_attention: torch.Size([2, 4, 512])
		# y_cat: torch.Size([2, 1024])
		# y_rank: torch.Size([8, 1])

		return output


class ReferenceBertChoiceModelA(Module):
	"""选择题Baseline模型: 使用参考文献与文档编码嵌入
	全连接层测试
	"""
	def __init__(self, args):
		super(ReferenceBertChoiceModelA, self).__init__()
		self.d_hidden = 512
		self.linear_a = Linear(args.bert_hidden_size * (args.num_top_subject * args.num_best_per_subject + 2), self.d_hidden)
		self.linear_b = Linear(args.bert_hidden_size * (args.num_top_subject * args.num_best_per_subject + 2), self.d_hidden)
		self.linear_c = Linear(args.bert_hidden_size * (args.num_top_subject * args.num_best_per_subject + 2), self.d_hidden)
		self.linear_d = Linear(args.bert_hidden_size * (args.num_top_subject * args.num_best_per_subject + 2), self.d_hidden)
		self.rank_module = Linear(self.d_hidden * 4, 4)
		self.multi_module = Linear(4, 16)

		self.dropout_a = Dropout(p=0.1)
		self.dropout_b = Dropout(p=0.1)
		self.dropout_c = Dropout(p=0.1)
		self.dropout_d = Dropout(p=0.1)

	def forward(self, data):
		options = data['options']		# [32, 4, 768]
		question = data['question']		# [32, 768]
		reference = data['reference']	# [32, 12, 768]

		batch_size = options.size()[0]
		n_options = options.size()[1]
		assert n_options == TOTAL_OPTIONS


		option_a = options[:, 0, :].squeeze(1)	# [32, 768]
		option_b = options[:, 1, :].squeeze(1)	# [32, 768]
		option_c = options[:, 2, :].squeeze(1)	# [32, 768]
		option_d = options[:, 3, :].squeeze(1)	# [32, 768]

		hidden_a = self.linear_a(torch.hstack([option_a, question, reference.view(batch_size, -1)]))	# [32, 1536 + 12 * 768] -> [32, 512]
		hidden_a = self.dropout_a(hidden_a)
		hidden_b = self.linear_b(torch.hstack([option_b, question, reference.view(batch_size, -1)]))	# [32, 1536 + 12 * 768] -> [32, 512]
		hidden_b = self.dropout_b(hidden_b)
		hidden_c = self.linear_c(torch.hstack([option_c, question, reference.view(batch_size, -1)]))	# [32, 1536 + 12 * 768] -> [32, 512]
		hidden_c = self.dropout_c(hidden_c)
		hidden_d = self.linear_d(torch.hstack([option_d, question, reference.view(batch_size, -1)]))	# [32, 1536 + 12 * 768] -> [32, 512]
		hidden_d = self.dropout_d(hidden_d)

		hidden = torch.hstack([hidden_a, hidden_b, hidden_c, hidden_d])	# [32, 512 * 4]

		y = self.rank_module(hidden)									# [32, 512 * 4] -> [32, 4]
		y = self.multi_module(y)										# [32, 4] -> [32, 16]

		return y


class ReferenceWord2VecChoiceModel(Module):
	"""选择题Baseline模型: 不使用参考文献, 且使用预训练的文档嵌入"""
	def __init__(self, args):
		super(ReferenceWord2VecChoiceModel, self).__init__()
		if args.word_embedding == 'word2vec':
			self.d_hidden = args.size_word2vec
		elif args.word_embedding == 'fasttext':
			self.d_hidden = args.size_fasttext
		else:
			raise NotImplementedError
		self.options_encoder = BaseLSTMEncoder(d_hidden=self.d_hidden)
		self.question_encoder = BaseLSTMEncoder(d_hidden=self.d_hidden)
		self.attention = BaseAttention(d_hidden=self.d_hidden)
		self.rank_module = Linear(int(self.d_hidden / 2), 1)
		self.multi_module = Linear(4, 16)

	def forward(self, data):
		options = data['options']		# [32, 4, 128, 256]
		question = data['question']		# [32, 256, 256]
		reference = data['reference']	# [32, 12, 512, 256]

		batch_size = question.size()[0]
		n_options = TOTAL_OPTIONS

		expanded_question = torch.cat([question for _ in range(4)]).view(batch_size, n_options, -1, self.d_hidden)	# [32, 4, 256, 256]

		_, encoded_options = self.options_encoder(options.view(batch_size, -1, self.d_hidden))
		_, encoded_question = self.question_encoder(expanded_question.view(batch_size, -1, self.d_hidden))
		_, encoded_reference = self.question_encoder(reference.view(batch_size, -1, self.d_hidden))

		options_attention, _, _ = self.attention(encoded_options, encoded_reference)
		question_attention, _, _ = self.attention(encoded_question, encoded_reference)

		y = torch.cat([torch.max(options_attention, dim=1)[0], torch.max(question_attention, dim=1)[0]], dim=1)

		y = self.rank_module(y.view(batch_size * n_options, -1))
		output = self.multi_module(y.view(batch_size, n_options))
		
		return output


#######################################################################
# -*- -*- -*- -*- -*- -*- 华 丽 的 分 界 线 -*- -*- -*- -*- -*- -*- -*- #
#######################################################################

class TreeChoiceModel(Module):
	"""使用树结构的选择题模型: 不使用参考文献"""
	def __init__(self, args):
		super(TreeChoiceModel, self).__init__()
		self.args = deepcopy(args)
		
		# 2022/09/13 18:58:26 这里的RNN类型沿用TreeRNNEncoder中的类型, 感觉没有必要定义那么多的配置项
		# 2022/09/13 18:58:26 这里的输入维数即TreeRNNEncoder从ROOT节点的输出维数, 即args.tree_rnn_encoder_root_output_size
		# 2022/09/13 18:58:26 其他几个参数都是根据配置项决定的, question与option使用相同的参数可便于后序计算
		self.question_aggregation_module = eval(self.args.tree_rnn_encoder_rnn_type)(input_size		= self.args.tree_rnn_encoder_root_output_size, 
																					 hidden_size	= int(self.args.tree_model_aggregation_module_output_size / (1 + self.args.tree_model_aggregation_module_bidirectional)),	# 双向RNN的输出维数是hidden_size的2倍
																					 num_layers		= self.args.tree_rnn_encoder_num_layers,
																				   	 batch_first	= True, 
																					 bidirectional	= self.args.tree_model_aggregation_module_bidirectional)
		
		self.option_aggregation_module = eval(self.args.tree_rnn_encoder_rnn_type)(input_size		= self.args.tree_rnn_encoder_root_output_size, 
																				   hidden_size		= int(self.args.tree_model_aggregation_module_output_size / (1 + self.args.tree_model_aggregation_module_bidirectional)),	# 双向RNN的输出维数是hidden_size的2倍
																				   num_layers		= self.args.tree_model_aggregation_module_num_layers,
																				   batch_first		= True, 
																				   bidirectional	= self.args.tree_model_aggregation_module_bidirectional)
		# 2022/09/13 18:58:26 同上沿用TreeRNNEncoder中的思路
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

		# 2022/09/13 20:32:38 解题模块: 目前直接用一个全连接层输出到一个神经元上作为判断的依据, 待改进
		# 2022/09/13 20:32:38 输入为question_parse_tree_squeezed_output与option_x_parse_tree_squeezed_output的拼接, 即形状为(batch_size, 2 * tree_model_aggregation_module_output_size)
		self.solver_module = Linear(2 * self.args.tree_model_aggregation_module_output_size, 1)
		
		# 2022/09/13 20:32:38 最后用一个全连接层将4个选项的输出映射到16维的向量上
		self.multi_choice_module = Linear(TOTAL_OPTIONS, TOTAL_OPTIONS * TOTAL_OPTIONS)
		
	def forward(self, data):
		"""2022/09/11 15:03:26
		:param data: 包含的字段:
		- question_parse_tree_outputs	: 若干句法树从ROOT节点处最终输出的张量列表, 每个的形状为(batch_size, tree_rnn_encoder_root_output_size)
		- option_a_parse_tree_outputs	: 若干句法树从ROOT节点处最终输出的张量列表, 每个的形状为(batch_size, tree_rnn_encoder_root_output_size)
		- option_b_parse_tree_outputs	: 若干句法树从ROOT节点处最终输出的张量列表, 每个的形状为(batch_size, tree_rnn_encoder_root_output_size)
		- option_c_parse_tree_outputs	: 若干句法树从ROOT节点处最终输出的张量列表, 每个的形状为(batch_size, tree_rnn_encoder_root_output_size)
		- option_d_parse_tree_outputs	: 若干句法树从ROOT节点处最终输出的张量列表, 每个的形状为(batch_size, tree_rnn_encoder_root_output_size)
		"""
		# 合并TreeRNNEncoder输出: (num_trees, batch_size, tree_rnn_encoder_root_output_size) -> (batch_size, num_trees, tree_rnn_encoder_root_output_size)
		question_parse_tree_outputs = torch.stack(data['question_parse_tree_outputs']).permute((1, 0, 2))
		option_a_parse_tree_outputs = torch.stack(data['option_a_parse_tree_outputs']).permute((1, 0, 2))
		option_b_parse_tree_outputs = torch.stack(data['option_b_parse_tree_outputs']).permute((1, 0, 2))
		option_c_parse_tree_outputs = torch.stack(data['option_c_parse_tree_outputs']).permute((1, 0, 2))
		option_d_parse_tree_outputs = torch.stack(data['option_d_parse_tree_outputs']).permute((1, 0, 2))
		
		# RNN聚合编码: (batch_size, num_trees, tree_rnn_encoder_root_output_size) -> (batch_size, num_trees, tree_model_aggregation_module_output_size)
		question_parse_tree_final_output, _ = self.question_aggregation_module(question_parse_tree_outputs)
		option_a_parse_tree_final_output, _ = self.option_aggregation_module(option_a_parse_tree_outputs)
		option_b_parse_tree_final_output, _ = self.option_aggregation_module(option_b_parse_tree_outputs)
		option_c_parse_tree_final_output, _ = self.option_aggregation_module(option_c_parse_tree_outputs)
		option_d_parse_tree_final_output, _ = self.option_aggregation_module(option_d_parse_tree_outputs)

		# squeeze: (batch_size, num_trees, tree_model_aggregation_module_output_size) -> (batch_size, tree_model_aggregation_module_output_size)
		if self.args.tree_rnn_encoder_squeeze_strategy == 'mean':
			question_parse_tree_squeezed_output = torch.mean(question_parse_tree_final_output, axis=1)			
			option_a_parse_tree_squeezed_output = torch.mean(option_a_parse_tree_final_output, axis=1)			
			option_b_parse_tree_squeezed_output = torch.mean(option_b_parse_tree_final_output, axis=1)			
			option_c_parse_tree_squeezed_output = torch.mean(option_c_parse_tree_final_output, axis=1)			
			option_d_parse_tree_squeezed_output = torch.mean(option_d_parse_tree_final_output, axis=1)			
		elif self.args.tree_rnn_encoder_squeeze_strategy == 'final':
			question_parse_tree_squeezed_output = question_parse_tree_final_output[:, -1, :]
			option_a_parse_tree_squeezed_output = option_a_parse_tree_final_output[:, -1, :]
			option_b_parse_tree_squeezed_output = option_b_parse_tree_final_output[:, -1, :]
			option_c_parse_tree_squeezed_output = option_c_parse_tree_final_output[:, -1, :]
			option_d_parse_tree_squeezed_output = option_d_parse_tree_final_output[:, -1, :]
		elif self.args.tree_rnn_encoder_squeeze_strategy == 'fixed_weight':
			question_sequence_length = question_parse_tree_final_output.shape[1]
			question_parse_tree_squeezed_output = question_parse_tree_final_output[:, 0, :] * self.squeeze_weight[question_sequence_length - 2]	# 初始位置的权重跟下一个位置的权重是相同的, 即1/2, 1/4, 1/8, 1/8
			for i in range(1, question_sequence_length):
				question_parse_tree_squeezed_output += question_parse_tree_final_output[: -i, :] * self.squeeze_weight[i - 1]
				
			option_a_sequence_length = option_a_parse_tree_final_output.shape[1]
			option_a_parse_tree_squeezed_output = option_a_parse_tree_final_output[:, 0, :] * self.squeeze_weight[option_a_sequence_length - 2]	# 初始位置的权重跟下一个位置的权重是相同的, 即1/2, 1/4, 1/8, 1/8
			for i in range(1, option_a_sequence_length):
				option_a_parse_tree_squeezed_output += option_a_parse_tree_final_output[: -i, :] * self.squeeze_weight[i - 1]			
				
			option_b_sequence_length = option_b_parse_tree_final_output.shape[1]
			option_b_parse_tree_squeezed_output = option_b_parse_tree_final_output[:, 0, :] * self.squeeze_weight[option_b_sequence_length - 2]	# 初始位置的权重跟下一个位置的权重是相同的, 即1/2, 1/4, 1/8, 1/8
			for i in range(1, option_b_sequence_length):
				option_b_parse_tree_squeezed_output += option_b_parse_tree_final_output[: -i, :] * self.squeeze_weight[i - 1]			
			
			option_c_sequence_length = option_c_parse_tree_final_output.shape[1]
			option_c_parse_tree_squeezed_output = option_c_parse_tree_final_output[:, 0, :] * self.squeeze_weight[option_c_sequence_length - 2]	# 初始位置的权重跟下一个位置的权重是相同的, 即1/2, 1/4, 1/8, 1/8
			for i in range(1, option_c_sequence_length):
				option_c_parse_tree_squeezed_output += option_c_parse_tree_final_output[: -i, :] * self.squeeze_weight[i - 1]			
		
			option_d_sequence_length = option_d_parse_tree_final_output.shape[1]
			option_d_parse_tree_squeezed_output = option_d_parse_tree_final_output[:, 0, :] * self.squeeze_weight[option_d_sequence_length - 2]	# 初始位置的权重跟下一个位置的权重是相同的, 即1/2, 1/4, 1/8, 1/8
			for i in range(1, option_d_sequence_length):
				option_d_parse_tree_squeezed_output += option_d_parse_tree_final_output[: -i, :] * self.squeeze_weight[i - 1]
		elif self.args.tree_rnn_encoder_squeeze_strategy == 'variable_weight':
			# 2022/09/06 22:41:04 实现逻辑与fixed_weight的情况是一样的
			raise NotImplementedError
		else:
			raise Exception(f'Unknown squeeze_strategy: {self.squeeze_strategy}')
		
		# question_parse_tree_squeezed_output: (batch_size, tree_model_aggregation_module_output_size)
		# option_a_parse_tree_squeezed_output: (batch_size, tree_model_aggregation_module_output_size)
		# option_b_parse_tree_squeezed_output: (batch_size, tree_model_aggregation_module_output_size)
		# option_c_parse_tree_squeezed_output: (batch_size, tree_model_aggregation_module_output_size)
		# option_d_parse_tree_squeezed_output: (batch_size, tree_model_aggregation_module_output_size)
		
		# 解题: 	(batch_size, 2 * tree_model_aggregation_module_output_size) -> (batch_size, 1)
		option_a_flag = self.solver_module(torch.hstack([question_parse_tree_squeezed_output, option_a_parse_tree_squeezed_output]))
		option_b_flag = self.solver_module(torch.hstack([question_parse_tree_squeezed_output, option_b_parse_tree_squeezed_output]))
		option_c_flag = self.solver_module(torch.hstack([question_parse_tree_squeezed_output, option_c_parse_tree_squeezed_output]))
		option_d_flag = self.solver_module(torch.hstack([question_parse_tree_squeezed_output, option_d_parse_tree_squeezed_output]))
		
		# 转为多项选择结果: (batch_size, TOTAL_OPTIONS) -> (batch_size, TOTAL_OPTIONS * TOTAL_OPTIONS)
		output = self.multi_choice_module(torch.hstack([option_a_flag, option_b_flag, option_c_flag, option_d_flag]))
		return output	
		

class TreeReferenceChoiceModel(Module):
	"""使用树结构的选择题模型: 不使用参考文献"""
	def __init__(self, args):
		super(TreeReferenceChoiceModel, self).__init__()
		
		
	def forward(self, data):
		pass


if __name__ == '__main__':

	pass
