# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
# 模型训练: 使用sklearn

import os
import gc
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from sklearn import preprocessing

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import auc, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc, precision_recall_curve

from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier

from matplotlib import pyplot as plt

from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, BCELoss
from torch.optim import Adam, lr_scheduler

from setting import *
from config import QAModelConfig, DatasetConfig

from src.dataset import generate_basic_dataloader, BasicDataset
from src.evaluation_tools import evaluate_qa_model_choice, evaluate_qa_model_judgment, evaluate_classifier
from src.plot_tools import plot_roc_curve, plot_pr_curve
from src.qa_model import BaseChoiceModel, BaseJudgmentModel, ReferenceChoiceModel, ReferenceJudgmentModel
from src.torch_tools import save_checkpoint
from src.utils import initialize_logger, terminate_logger, load_args, save_args, timer
from src.easy_machine_learning import EasyClassifier

def generate_dataset_for_sklearn(args, mode):
	dataset_train = BasicDataset(args=args, mode=mode, do_export=False, pipeline='judgment', for_debug=True).data
	dataset_valid = BasicDataset(args=args, mode=mode.replace('train', 'valid'), do_export=False, pipeline='judgment', for_debug=True).data
	dataset_test = BasicDataset(args=args, mode=mode.replace('train', 'test'), do_export=False, pipeline='judgment').data
	
	dataset_train = pd.concat([dataset_train, dataset_valid])
	'''
	id				: 题目编号
	question		: 题目题干
	option			: 每个选项
	subject			: use_reference配置为True时生效, 包含num_top_subject个法律门类
	reference		: use_reference配置为True时生效, 包含相关的num_best个参考书目文档段落
	type			: 零一值表示概念题或情景题
	label_judgment	: train或valid模式时生效, 零一值表示判断题的答案
	option_id		: 20211216更新, 记录判断题对应的原选择题编号(ABCD)
	'''
	
	if args.use_reference:
		dataset_X = np.hstack([np.vstack(dataset_train['question'].map(np.array).values),
							   np.vstack(dataset_train['option'].map(np.array).values)])
		test_X = np.hstack([np.vstack(dataset_test['question'].map(np.array).values),
							np.vstack(dataset_test['option'].map(np.array).values)])
	else:
		dataset_X = np.hstack([np.vstack(dataset_train['question'].map(np.array).values),
							   np.vstack(dataset_train['option'].map(np.array).values),
							   np.vstack(dataset_train['reference']).map(lambda x: np.array(x).reshape((-1, )))])
		test_X = np.hstack([np.vstack(dataset_test['question'].map(np.array).values),
							np.vstack(dataset_test['option'].map(np.array).values),
							np.vstack(dataset_test['reference']).map(lambda x: np.array(x).reshape((-1, )))])
	
	dataset_y = dataset_train['label_judgment'].values	
	question_ids = dataset_test['id'].tolist()
	option_ids = dataset_test['option_id'].tolist()
	
	return dataset_X, dataset_y, (test_X, question_ids, option_ids)

@timer
def train_model(model_name, mode='train_kd', dataset_X=None, dataset_y=None, **kwargs):
	
	if dataset_X is None or dataset_y is None:
		
		args = load_args(DatasetConfig)
		for key, value in kwargs.items():
			args.__setattr__(key, value)	
		
		dataset_X, dataset_y, _ = generate_dataset_for_sklearn(args=args, mode=mode)
	
	mode_suffix = mode.split('_')[-1]
	
	
	if model_name == 'lgb':
		params = {
			'boosting_type': 'gbdt',
			'num_leaves': 256,
			'max_depth': 8,
			'learning_rate': .001,
			'n_estimators': 128,
			'subsample_for_bin': 200000,
			'objective': None,
			'class_weight': None,
			# 'min_split_gain': .0,
			# 'min_child_weight': 0,
			# 'min_child_samples': 100,
			# 'subsample': 1.,
			# 'subsample_freq': 0,
			# 'colsample_bytree': 1.,
			'reg_alpha': 0,
			'reg_lambda': 0,
			'random_state': None,
			'n_jobs': 1,
			'silent': True,
			'importance_type': 'split',
		}
		
		easyclf = EasyClassifier(dataset_X, dataset_y, LGBMClassifier, params)
		if mode_suffix == 'kd':
			easyclf.kfold_train(n_splits=5,
								shuffle=True,
								pr_title_formatter='PR Curve of Random Forest {}'.format,
								roc_title_formatter='ROC Curve of Random Forest {}'.format,
								model_export_path_formatter='temp/sklearn_test/lgb/kd/rf_fold_{}.m'.format,
								pr_export_path_formatter='temp/sklearn_test/lgb/kd/pr_curve_fold_{}.png'.format,
								roc_export_path_formatter='temp/sklearn_test/lgb/kd/roc_curve_fold_{}.png'.format,
								evaluation_export_path='temp/sklearn_test/lgb/kd/rf_eval.json')
		elif mode_suffix == 'ca':
			easyclf.kfold_train(n_splits=5,
								shuffle=True,
								pr_title_formatter='PR Curve of Random Forest {}'.format,
								roc_title_formatter='ROC Curve of Random Forest {}'.format,
								model_export_path_formatter='temp/sklearn_test/lgb/ca/rf_fold_{}.m'.format,
								pr_export_path_formatter='temp/sklearn_test/lgb/ca/pr_curve_fold_{}.png'.format,
								roc_export_path_formatter='temp/sklearn_test/lgb/ca/roc_curve_fold_{}.png'.format,
								evaluation_export_path='temp/sklearn_test/lgb/ca/rf_eval.json')	
							

	if model_name == 'xgb':
		# XGBoost

		params = {
			'n_estimators': 128,
			'use_label_encoder': False,
			'max_depth': 8,
			'num_leaves': 256,
			'learning_rate': .01,
			'verbosity': 0,
			# 'objective': None,
			'booster': 'dart',
			# 'tree_method': None,
			'n_jobs': 1,
			'gamma': .001,
			# 'min_child_weight': .01,
			# 'max_delta_step': .01,
			# 'subsample': 1.,
			# 'colsample_bytree': 1.,
			# 'colsample_bylevel': 1.,
			# 'colsample_bynode': 1.,
			'reg_alpha': .01,
			'reg_lambda': .01,
			# 'scale_pos_weight': .01,
			# 'base_score': 1,
			'random_state': None,
			# 'missing': np.nan,
			'num_parallel_tree': 10,
			# 'monotone_constraints': None,
			# 'interaction_constraints': None,
			# 'importance_type': 'gain',
			# 'gpu_id': None,
			# 'validate_parameters': None,
		}

		easyclf = EasyClassifier(dataset_X, dataset_y, XGBClassifier, params)
		if mode_suffix == 'kd':
			easyclf.kfold_train(n_splits=5,
								shuffle=True,
								pr_title_formatter='PR Curve of Random Forest {}'.format,
								roc_title_formatter='ROC Curve of Random Forest {}'.format,
								model_export_path_formatter='temp/sklearn_test/xgb/kd/rf_fold_{}.m'.format,
								pr_export_path_formatter='temp/sklearn_test/xgb/kd/pr_curve_fold_{}.png'.format,
								roc_export_path_formatter='temp/sklearn_test/xgb/kd/roc_curve_fold_{}.png'.format,
								evaluation_export_path='temp/sklearn_test/xgb/kd/rf_eval.json')
		elif mode_suffix == 'ca':
			easyclf.kfold_train(n_splits=5,
								shuffle=True,
								pr_title_formatter='PR Curve of Random Forest {}'.format,
								roc_title_formatter='ROC Curve of Random Forest {}'.format,
								model_export_path_formatter='temp/sklearn_test/xgb/ca/rf_fold_{}.m'.format,
								pr_export_path_formatter='temp/sklearn_test/xgb/ca/pr_curve_fold_{}.png'.format,
								roc_export_path_formatter='temp/sklearn_test/xgb/ca/roc_curve_fold_{}.png'.format,
								evaluation_export_path='temp/sklearn_test/xgb/ca/rf_eval.json')	
							
	
	if model_name == 'lr':
		# Logistic Regression
		params = {
			'penalty': 'l2',
			'dual': False,
			'tol': 1e-4,
			'C': 1.0,
			'fit_intercept': True,
			'intercept_scaling': 1.0,
			'class_weight': None,
			'random_state': None,
			'solver': 'liblinear',
			'max_iter': 100,
			'multi_class': 'ovr',
			'verbose': 0,
			'warm_start': False,
			'n_jobs': None,
		}
		easyclf = EasyClassifier(dataset_X, dataset_y, LogisticRegression, params)
		if mode_suffix == 'kd':
			easyclf.kfold_train(n_splits=5,
								shuffle=True,
								pr_title_formatter='PR Curve of Random Forest {}'.format,
								roc_title_formatter='ROC Curve of Random Forest {}'.format,
								model_export_path_formatter='temp/sklearn_test/lr/kd/rf_fold_{}.m'.format,
								pr_export_path_formatter='temp/sklearn_test/lr/kd/pr_curve_fold_{}.png'.format,
								roc_export_path_formatter='temp/sklearn_test/lr/kd/roc_curve_fold_{}.png'.format,
								evaluation_export_path='temp/sklearn_test/lr/kd/rf_eval.json')
		elif mode_suffix == 'ca':
			easyclf.kfold_train(n_splits=5,
								shuffle=True,
								pr_title_formatter='PR Curve of Random Forest {}'.format,
								roc_title_formatter='ROC Curve of Random Forest {}'.format,
								model_export_path_formatter='temp/sklearn_test/lr/ca/rf_fold_{}.m'.format,
								pr_export_path_formatter='temp/sklearn_test/lr/ca/pr_curve_fold_{}.png'.format,
								roc_export_path_formatter='temp/sklearn_test/lr/ca/roc_curve_fold_{}.png'.format,
								evaluation_export_path='temp/sklearn_test/lr/ca/rf_eval.json')	
		
	if model_name == 'dt':
		# Decision tree

		params = {
			'criterion': 'gini',
			'splitter': 'best',
			'max_depth': None,
			'min_samples_split': 2,
			'min_samples_leaf': 1,
			'min_weight_fraction_leaf': 0.0,
			'max_features': None,
			'random_state': None,
			'max_leaf_nodes': None,
			'min_impurity_decrease': 0.0,
			'class_weight': None,
		}

		easyclf = EasyClassifier(dataset_X, dataset_y, DecisionTreeClassifier, params)
		if mode_suffix == 'kd':
			easyclf.kfold_train(n_splits=5,
								shuffle=True,
								pr_title_formatter='PR Curve of Random Forest {}'.format,
								roc_title_formatter='ROC Curve of Random Forest {}'.format,
								model_export_path_formatter='temp/sklearn_test/dt/kd/rf_fold_{}.m'.format,
								pr_export_path_formatter='temp/sklearn_test/dt/kd/pr_curve_fold_{}.png'.format,
								roc_export_path_formatter='temp/sklearn_test/dt/kd/roc_curve_fold_{}.png'.format,
								evaluation_export_path='temp/sklearn_test/dt/kd/rf_eval.json')
		elif mode_suffix == 'ca':
			easyclf.kfold_train(n_splits=5,
								shuffle=True,
								pr_title_formatter='PR Curve of Random Forest {}'.format,
								roc_title_formatter='ROC Curve of Random Forest {}'.format,
								model_export_path_formatter='temp/sklearn_test/dt/ca/rf_fold_{}.m'.format,
								pr_export_path_formatter='temp/sklearn_test/dt/ca/pr_curve_fold_{}.png'.format,
								roc_export_path_formatter='temp/sklearn_test/dt/ca/roc_curve_fold_{}.png'.format,
								evaluation_export_path='temp/sklearn_test/dt/ca/rf_eval.json')	

	if model_name == 'rf':
		# Random Forest

		params = {
			'n_estimators': 100,
			'criterion': 'gini',
			'max_depth': None,
			'min_samples_split': 2,
			'min_samples_leaf': 1,
			'min_weight_fraction_leaf': 0.0,
			'max_features': 'auto',
			'max_leaf_nodes': None,
			'min_impurity_decrease': 0.0,
			'min_impurity_split': None,
			'bootstrap': True,
			'oob_score': False,
			'n_jobs': None,
			'random_state': None,
			'verbose': 0,
			'warm_start': False,
			'class_weight': None,
			'ccp_alpha': 0.0,
			'max_samples': None,
		}

		# easyclf = EasyClassifier(dataset_X_onehot, dataset_y, RandomForestClassifier, params)
		easyclf = EasyClassifier(dataset_X, dataset_y, RandomForestClassifier, params)
		
		if mode_suffix == 'kd':
			easyclf.kfold_train(n_splits=5,
								shuffle=True,
								pr_title_formatter='PR Curve of Random Forest {}'.format,
								roc_title_formatter='ROC Curve of Random Forest {}'.format,
								model_export_path_formatter='temp/sklearn_test/rf/kd/rf_fold_{}.m'.format,
								pr_export_path_formatter='temp/sklearn_test/rf/kd/pr_curve_fold_{}.png'.format,
								roc_export_path_formatter='temp/sklearn_test/rf/kd/roc_curve_fold_{}.png'.format,
								evaluation_export_path='temp/sklearn_test/rf/kd/rf_eval.json')

		elif mode_suffix == 'ca':
			easyclf.kfold_train(n_splits=5,
								shuffle=True,
								pr_title_formatter='PR Curve of Random Forest {}'.format,
								roc_title_formatter='ROC Curve of Random Forest {}'.format,
								model_export_path_formatter='temp/sklearn_test/rf/ca/rf_fold_{}.m'.format,
								pr_export_path_formatter='temp/sklearn_test/rf/ca/pr_curve_fold_{}.png'.format,
								roc_export_path_formatter='temp/sklearn_test/rf/ca/roc_curve_fold_{}.png'.format,
								evaluation_export_path='temp/sklearn_test/rf/ca/rf_eval.json')


def test_model(models, model_name, mode='train_kd', test_X=None, question_ids=None, option_ids=None, **kwargs):
	if test_X is None or question_ids is None or option_ids is None:
		args = load_args(DatasetConfig)
		for key, value in kwargs.items():
			args.__setattr__(key, value)	
		_, _, (test_X, question_ids, option_ids) = generate_dataset_for_sklearn(args=args, mode=mode)
	

	predicts = []
	for model in models:
		predict_y = model.predict(test_X)
		print(len(predict_y))
		predicts.append(predict_y)
		
	answer = {}	
	
	print(len(question_ids))
	print(len(option_ids))
	
	
	for i, (question_id, option_id) in enumerate(zip(question_ids, option_ids)):
		
		count_0 = 0
		count_1 = 0
		for predict in predicts:
			if predict[i] == 0:
				count_0 += 1
			elif predict[i] == 1:
				count_1 += 1
			else:
				assert False
		
		if count_1 > count_0:
			# 投票
			if question_id in answer:
				answer[question_id].append(option_id)
			else:
				answer[question_id] = [option_id]
		
		else:
			if question_id in answer:
				pass
			else:
				answer[question_id] = []			
	
	# 导出答案
	with open(os.path.join(TEMP_DIR, f'answer_{model_name}_{mode.split("_")[-1]}.json'), 'w', encoding='utf8') as f:
		json.dump(answer, f, indent=4)
		
	return answer

for mode in ['train_kd', 'train_ca']:
	kwargs = {
		'use_reference'	: True,
		'num_best': 32,
	}
	args = load_args(DatasetConfig)
	for key, value in kwargs.items():
		args.__setattr__(key, value)	
	dataset_X, dataset_y, (test_X, question_ids, option_ids) = generate_dataset_for_sklearn(args=args, mode=mode)
	# for model_name in ['lr', 'dt', 'rf']:
	for model_name in ['rf']:
		print('*' * 64)
		print(mode, model_name)
		print('*' * 64)
		
		# train_model(model_name=model_name, mode=mode, dataset_X=dataset_X, dataset_y=dataset_y, test_X=test_X, **kwargs)
		
		models = []
		for fold in range(1, 6):
			model = joblib.load(f'temp/sklearn_test/{model_name}/{mode.split("_")[-1]}/rf_fold_{str(fold).zfill(2)}.m')
			models.append(model)
			
		test_model(models=models, model_name=model_name, mode=mode, test_X=test_X, question_ids=question_ids, option_ids=option_ids, **kwargs)
			
		
