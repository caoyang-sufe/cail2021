# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
# 默认参数配置: 尽量不要出现同名参数

import argparse

from copy import deepcopy

class BaseConfig:
	"""基础的全局配置"""
	parser = argparse.ArgumentParser('--')
	parser.add_argument('--filter_stopword', default=True, type=bool, help='是否过滤停用词')
	parser.add_argument('--retrieval_model_name', default='tfidf', type=str, help='最终使用的文档检索模型')
	parser.add_argument('--num_top_subject', default=3, type=int, help='每道题目对应的候选法律门类数量')
	parser.add_argument('--num_best_per_subject', default=6, type=int, help='每个法律门类下选取的参考段落数量, 对应num_top_subject配置参数')	# 2022/01/14 23:10:57 新增
	parser.add_argument('--use_reference', default=True, type=bool, help='是否使用参考书目文档')
	parser.add_argument('--use_userdict', default=True, type=bool, help='是否使用自定义的分词字典, 用于jieba分词')
	
	# 2022/03/30 19:54:33 新增更多的数据字段
	parser.add_argument('--use_pos_tags', default=False, type=bool, help='是否使用词性标注')
	parser.add_argument('--use_parse_tree', default=False, type=bool, help='是否使用句法解析树, 注意如果使用了句法解析树, 则必然使用词性标注')
	parser.add_argument('--generate_pos_tags_by', default='nltk', type=str, help='目前仅实现nltk, 也可以用jieba, 但是jieba的问题在于他必须用原句, 而不能用分词序列, 而原句输入后得到的词性分词结果与直接分词会有区别')
	
	parser.add_argument('--train_batch_size', default=32, type=int, help='训练集批训练的批大小')
	parser.add_argument('--valid_batch_size', default=32, type=int, help='验证集批训练的批大小')
	parser.add_argument('--test_batch_size', default=32, type=int, help='测试集批训练的批大小')
	
	parser.add_argument('--max_statement_length', default=256, type=int, help='题目题干分词序列最大长度')
	parser.add_argument('--max_option_length', default=128, type=int, help='题目选项分词序列最大长度')
	parser.add_argument('--max_reference_length', default=512, type=int, help='参考书目段落分词序列最大长度, 512超过参考书目文档段落分词长度的0.998分位数')
	
	parser.add_argument('--word_embedding', default=None, type=str, help='使用的词嵌入, 默认值None表示只用token2id的顺序编码值, 目前可选参数为word2vec, fasttext')
	parser.add_argument('--document_embedding', default=None, type=str, help='使用的文档嵌入, 默认值None表示不使用文档嵌入, 此时word_embedding参数必须非None, 目前可选参数为doc2vec, bert-base-chinese')
	
	# 模型训练的配置
	parser.add_argument('--num_epoch', default=32, type=int, help='训练轮数')
	parser.add_argument('--lr_multiplier', default=.95, type=float, help='lr_scheduler的gamma参数值, 即学习率的衰减')
	parser.add_argument('--learning_rate', default=.01, type=float, help='学习率或步长')
	parser.add_argument('--weight_decay', default=.0, type=float, help='权重衰减')
	
	parser.add_argument('--do_valid', default=True, type=bool, help='是否在训练过程中进行模型验证')
	parser.add_argument('--do_valid_plot', default=True, type=bool, help='是否在验证过程中绘制图像, 目前指对于判断题模型进行ROC曲线和PR曲线的绘制')

	parser.add_argument('--num_best', default=32, type=int, help='Similarity的num_best参数值, 是目前引用的参考文献数目')
	
	parser.add_argument('--test_thresholds', default=[.4, .5, .6], type=list, help='判断题测试的阈值')
	
	# 2022/02/21 12:46:00 这三个参数也提到全局部分来, 供qa_model调用
	parser.add_argument('--size_word2vec', default=256, type=int, help='gensim嵌入模型Word2Vec的嵌入维数, 即Word2Vec模型的size参数')
	parser.add_argument('--size_fasttext', default=256, type=int, help='gensim嵌入模型FastText的嵌入维数, 即FastText模型的size参数')
	parser.add_argument('--size_doc2vec', default=512, type=int, help='gensim嵌入模型Doc2Vec的嵌入维数, 即FastText模型的vector_size参数(size参数即将被弃用)')

	# 2022/03/23 14:37:08 新增参数bert_hidden_size, 主要用于Dataset中生成空文档的零张量表示
	parser.add_argument('--bert_hidden_size', default=768, type=int, help='bert-base-chinese模型默认的config中hidden_size配置参数的取值')
	
class DatasetConfig:
	"""数据集相关配置"""
	parser = deepcopy(BaseConfig.parser)

	parser.add_argument('--num_workers', default=0, type=int, help='DataLoader的num_workers参数值, 这个数值最好填零, 不然容易出错')

class RetrievalModelConfig:
	"""文档检索模型相关配置"""
	parser = deepcopy(BaseConfig.parser)
	
	parser.add_argument('--smartirs_tfidf', default=None, type=int, help='从{btnaldL}{xnftp}{xncub}的组合中挑选, 默认值为nfc, 详见https://radimrehurek.com/gensim/models/tfidfmodel.html')
	parser.add_argument('--pivot_tfidf', default=None, type=float, help='针对长文档进行的枢轴修正参数, 见https://radimrehurek.com/gensim/models/tfidfmodel.html')
	parser.add_argument('--slope_tfidf', default=.25, type=float, help='针对长文档进行的枢轴修正参数, 见https://radimrehurek.com/gensim/models/tfidfmodel.html')
		
	parser.add_argument('--num_topics_lsi', default=200, type=int, help='LSI模型的num_topics参数值')
	parser.add_argument('--decay_lsi', default=1., type=float, help='LSI模型的decay参数值')
	parser.add_argument('--power_iters_lsi', default=2, type=int, help='LSI模型的power_iters参数值')
	parser.add_argument('--extra_samples_lsi', default=100, type=int, help='LSI模型的extra_samples参数值')
	
	parser.add_argument('--num_topics_lda', default=100, type=int, help='LDA模型的num_topics参数值')
	parser.add_argument('--decay_lda', default=.5, type=float, help='LDA模型的decay参数值')
	parser.add_argument('--iterations_lda', default=50, type=float, help='LDA模型的iterations参数值')
	parser.add_argument('--gamma_threshold_lda', default=.001, type=float, help='LDA模型的gamma_threshold参数值')
	parser.add_argument('--minimum_probability_lda', default=.01, type=float, help='LDA模型的minimum_probability参数值')
			
	parser.add_argument('--kappa_hdp', default=1., type=float, help='HDP模型的kappa参数值')
	parser.add_argument('--tau_hdp', default=64., type=float, help='HDP模型的tau参数值')
	parser.add_argument('--K_hdp', default=15, type=int, help='HDP模型的K参数值')
	parser.add_argument('--T_hdp', default=150, type=int, help='HDP模型的T参数值')

class EmbeddingModelConfig:
	"""词嵌入模型相关配置"""
	parser = deepcopy(BaseConfig.parser)
	
	# 三个size_xxx配置参数移动到BaseConfig范围内
	# parser.add_argument('--size_word2vec', default=256, type=int, help='gensim嵌入模型Word2Vec的嵌入维数, 即Word2Vec模型的size参数')
	parser.add_argument('--min_count_word2vec', default=5, type=int, help='Word2Vec模型的min_count参数')
	parser.add_argument('--window_word2vec', default=5, type=int, help='Word2Vec模型的window参数')
	parser.add_argument('--workers_word2vec', default=3, type=int, help='Word2Vec模型的workers参数')

	# parser.add_argument('--size_fasttext', default=256, type=int, help='gensim嵌入模型FastText的嵌入维数, 即FastText模型的size参数')
	parser.add_argument('--min_count_fasttext', default=5, type=int, help='FastText模型的min_count参数')
	parser.add_argument('--window_fasttext', default=5, type=int, help='FastText模型的window参数')
	parser.add_argument('--workers_fasttext', default=3, type=int, help='FastText模型的workers参数')
	
	# parser.add_argument('--size_doc2vec', default=512, type=int, help='gensim嵌入模型Doc2Vec的嵌入维数, 即FastText模型的vector_size参数(size参数即将被弃用)')
	parser.add_argument('--min_count_doc2vec', default=5, type=int, help='Doc2Vec模型的min_count参数')
	parser.add_argument('--window_doc2vec', default=5, type=int, help='Doc2Vec模型的window参数')
	parser.add_argument('--workers_doc2vec', default=3, type=int, help='Doc2Vec模型的workers参数')
	
	parser.add_argument('--bert_output', default='pooler_output', type=str, help='BERT模型使用的输出, 默认pooler_output即池化后的输出结果, 形状为(1, 768), 也可以使用last_hidden_state, 形状为(1, 42, 768)')

class QAModelConfig:
	"""问答模型相关配置"""
	parser = deepcopy(BaseConfig.parser)
	
	parser.add_argument('--tree_rnn_encoder_node_hidden_size', default=128, type=int, help='句法树节点网络的隐层维数')
	parser.add_argument('--tree_rnn_encoder_root_output_size', default=256, type=int, help='句法树从ROOT节点最终输出维数')
	parser.add_argument('--tree_rnn_encoder_rnn_type', default='GRU', type=str, help='TreeRNNEncoder的RNN类型')
	parser.add_argument('--tree_rnn_encoder_num_layers', default=2, type=int, help='TreeRNNEncoder的RNN层数')
	parser.add_argument('--tree_rnn_encoder_bidirectional', default=False, type=bool, help='TreeRNNEncoder是否使用双向RNN')
	parser.add_argument('--tree_rnn_encoder_squeeze_strategy', default='final', type=str, help='TreeRNNEncoder的使用的降维策略, 选值有mean(取均值), final(只取最后一层输出), fixed_weight(固定权重加权平均), variable_weight(可变权重加权平均, 即作为参数训练)')
		
		
	parser.add_argument('--tree_model_aggregation_module_output_size', default=256, type=int, help='qa_model中句法树模型类中的aggregation_module最终输出维数')
	parser.add_argument('--tree_model_aggregation_module_num_layers', default=2, type=int, help='qa_model中句法树模型类中的aggregation_module层数')
	parser.add_argument('--tree_model_aggregation_module_bidirectional', default=False, type=bool, help='qa_model中句法树模型类中的aggregation_module是否双向')
	
	parser.add_argument('--default_embedding_size', default=128, type=int, help='默认的嵌入维数, 用于顺序编码值的嵌入层')
	parser.add_argument('--default_max_child', default=256, type=int, help='句法树节点可能拥有的最大子女数默认值, 在README.ipynb中有脚本统计')

class GraphConfig:
	"""图模型相关配置"""
	parser = deepcopy(BaseConfig.parser)
	parser.add_argument('--split_symbols', default=['。', '，', '；', ',', '？'], type=list, help='用于分隔语句的字符')


if __name__ == '__main__':
	config = BaseConfig()
	parser = config.parser
	args = parser.parse_args()
	print('num_best' in args)
