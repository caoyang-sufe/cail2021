# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn

import re
import os

os.environ['FONT_PATH'] = r'C:\Windows\Fonts\simfang.ttf'
os.environ['JAVAHOME'] = r'usr.'

import json
import time
import dill
import torch
import pandas
import gensim

from pprint import pprint
from nltk.parse.stanford import StanfordParser

from config import DatasetConfig, RetrievalModelConfig, EmbeddingModelConfig, GraphConfig, QAModelConfig

from setting import *

from src.dataset import BasicDataset
from src.data_tools import load_userdict_jieba, load_stopwords
from src.retrieval_model import GensimRetrievalModel
from src.embedding_model import GensimEmbeddingModel, TransformersEmbeddingModel
from src.evaluation_tools import evaluate_gensim_model_in_filling_subject
from src.plot_tools import train_plot_choice, train_plot_judgment
from src.graph import Graph
from src.graph_tools import generate_pos_tags, generate_pos_tags_from_parse_tree, generate_parse_tree, parse_tree_to_graph, generate_dependency, tranverse_parse_tree
from src.utils import load_args, initialize_logger

from src.qa_module import TreeRNNEncoder

parse_tree = '(ROOT (IP (PP (P 根据) (NP (NN 新闻) (NN 报导))) (PU ，) (NP (QP (CD 大部分)) (NP (NN 中学生))) (VP (ADVP (AD 都)) (VP (VV 近视)))))'

tranverse_result, node_id2data = tranverse_parse_tree(parse_tree=parse_tree)

pprint(tranverse_result)
pprint(node_id2data)

