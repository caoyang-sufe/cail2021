# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# sklearn模型训练的简单脚本

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


class EasyClassifier(object):
	"""这是一个很实用的sklearn机器学习方法的封装类"""
	def __init__(self, input_data, target_data, model, params):
		assert input_data.shape[0] == target_data.shape[0], f'Inconsistent dimension of input data and target data: {input_data.shape[0]} and {target_data.shape[0]}'
		self.data_size = input_data.shape[0]
		self.input_data = input_data.copy()
		self.target_data = target_data.copy()
		self.model = model
		self.params = params.copy()

	def kfold_train(self, 
					n_splits=5, 
					shuffle=True, 
					pr_title_formatter=None, 
					roc_title_formatter=None, 
					model_export_path_formatter=None, 
					pr_export_path_formatter=None, 
					roc_export_path_formatter=None, 
					evaluation_export_path=None):
		kfold = KFold(n_splits=n_splits, shuffle=shuffle)
		evaluation_info = {f'Fold{str(_fold).zfill(2)}': {} for _fold in range(1, n_splits + 1)}
		fold = 0
		
		for train_index, test_index in kfold.split(range(self.data_size)):
			fold += 1
			print(f'Fold {fold}')
			fold_string = str(fold).zfill(2)

			# Model training
			classifier = self.model(**self.params)
			input_train_data = self.input_data[train_index]
			target_train_data = self.target_data[train_index]
			input_test_data = self.input_data[test_index]
			target_test_data = self.target_data[test_index]
			classifier.fit(input_train_data, target_train_data)

			# Model prediction
			predict_test_data = classifier.predict(input_test_data)
			predict_proba_test_data = classifier.predict_proba(input_test_data)[:, 1]

			# Evaluate model
			_confusion_matrix, _accuracy_score, _roc_auc_score = self.evaluate_classifier(classifier=classifier, 
																						  target=target_test_data, 
																						  predict=predict_test_data, 
																						  predict_proba=predict_proba_test_data)
			evaluation_info[f'Fold{fold_string}']['confusion matrix'] = _confusion_matrix.tolist()
			evaluation_info[f'Fold{fold_string}']['accuracy score'] = _accuracy_score
			evaluation_info[f'Fold{fold_string}']['auc score'] = _roc_auc_score

			# Dump model
			if model_export_path_formatter is not None:
				joblib.dump(classifier, model_export_path_formatter(fold_string))

			# Plot
			self.plot_pr_curve(target_test_data, 
							   predict_proba_test_data, 
							   title=None if pr_title_formatter is None else pr_title_formatter(fold_string), 
							   export_path=None if pr_export_path_formatter is None else pr_export_path_formatter(fold_string))
			self.plot_roc_curve(target_test_data, 
								predict_proba_test_data, 
								title=None if roc_title_formatter is None else roc_title_formatter(fold_string), 
								export_path=None if roc_export_path_formatter is None else roc_export_path_formatter(fold_string))
			
		# Logging
		if evaluation_export_path is not None:
			json.dump(obj=evaluation_info, 
					  fp=open(evaluation_export_path, 'w', encoding='utf8'), 
					  indent=2, 
					  ensure_ascii=False, 
					  sort_keys=True)
		self.generate_markdown_table(evaluation_info)

	def split_train(self, 
					test_size=.2, 
					train_size=.8, 
					shuffle=True, 
					pr_title=None,
					roc_title=None, 
					model_export_path=None, 
					pr_export_path=None, 
					roc_export_path=None, 
					evaluation_export_path=None):
		evaluation_info = {}
		
		# Model training
		classifier = self.model(**self.params)
		input_train_data, input_test_data, target_train_data, target_test_data = train_test_split(sampled_X, sampled_y, test_size=test_size, train_size=train_size, shuffle=shuffle)
		classifier.fit(input_train_data, target_train_data)

		# Model prediction
		predict_test_data = classifier.predict(input_test_data)
		predict_proba_test_data = classifier.predict_proba(input_test_data)

		# Evaluate model
		_confusion_matrix, _accuracy_score, _roc_auc_score = self.evaluate_classifier(classifier=classifier, 
																					  target=target_test_data, 
																					  predict=predict_test_data, 
																					  predict_proba=predict_proba_test_data)
		evaluation_info['confusion matrix'] = _confusion_matrix.tolist()
		evaluation_info['accuracy score'] = _accuracy_score
		evaluation_info['auc score'] = _roc_auc_score        

		# Dump model
		if model_export_path is not None:
			joblib.dump(classifier, model_export_path)

		# Plot
		self.plot_pr_curve(target=target_test_data, 
						   predict_proba=predict_proba_test_data, 
						   title=pr_title, 
						   export_path=pr_export_path)
		self.plot_roc_curve(target=target_test_data, 
							predict_proba=predict_proba_test_data, 
							title=roc_title, 
							export_path=roc_export_path)
		
		# Logging
		if evaluation_export_path is not None:
			json.dump(obj=evaluation_info, 
					  fp=open(evaluation_export_path, 'w', encoding='utf8'), 
					  indent=2, 
					  ensure_ascii=False, 
					  sort_keys=True)

	def evaluate_classifier(self, classifier, target, predict, predict_proba):
		_confusion_matrix = confusion_matrix(target, predict)
		_accuracy_score = accuracy_score(target, predict)
		_roc_auc_score = roc_auc_score(target, predict_proba)
		return _confusion_matrix, _accuracy_score, _roc_auc_score

	def plot_roc_curve(self, target, predict_proba, title=None, export_path=None):
		fpr, tpr, thresholds = roc_curve(target, predict_proba, pos_label=1)
		_auc = auc(fpr, tpr)
		plt.plot(fpr, tpr, 'g-', label=f'ROC(AUC={round(_auc, 3)})')
		plt.plot([0, 1], [0, 1], 'r--', label='Luck')
		plt.xlim([-0.05, 1.05])
		plt.ylim([-0.05, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('ROC Curve' if title is None else title)
		plt.legend()
		if export_path is None:
			plt.show()
		else:
			plt.savefig(export_path)
		plt.close()

	def plot_pr_curve(self, target, predict_proba, title=None, export_path=None):
		precision, recall, thresholds = precision_recall_curve(target, predict_proba)
		diff = float('inf')
		for _precision, _recall, threshold in zip(precision, recall, thresholds):
			if abs(_precision - _recall) < diff:
				balance_point = (_recall, _precision, threshold)
				diff = abs(_precision - _recall)
		plt.plot(recall, precision, 'g-', label=f'PR')
		plt.plot([0, 1], [0, 1], 'y--', label='Balance')
		plt.plot([balance_point[0]], [balance_point[1]], 'ro', label=f'Balance point: ({round(balance_point[0], 3)}, {round(balance_point[0], 3)})')
		plt.xlim([-0.05, 1.05])
		plt.ylim([-0.05, 1.05])
		plt.xlabel('Recall Rate')
		plt.ylabel('Precision Rate')
		plt.title('PR Curve' if title is None else title)
		plt.legend()
		if export_path is None:
			plt.show()
		else:
			plt.savefig(export_path)
		plt.close()

	def generate_markdown_table(self, evaluation_info, export_path='markdown.txt'):
		
		a = '{matrix}'
		markdown_formatter = fr'''|                 |  精确度  |                          混淆矩阵                           | $\rm AUC$ |
| :-------------: | :------: | :---------------------------------------------------------: | :----------: |
| 第$1$折验证结果 | ${round(evaluation_info['Fold01']['accuracy score'], 4)}$ | $\left[\begin{a}{evaluation_info['Fold01']['confusion matrix'][0][0]}&{evaluation_info['Fold01']['confusion matrix'][0][1]}\\{evaluation_info['Fold01']['confusion matrix'][1][0]}&{evaluation_info['Fold01']['confusion matrix'][1][1]}\end{a}\right]$ |   ${round(evaluation_info['Fold01']['auc score'], 4)}$   |
| 第$2$折验证结果 | ${round(evaluation_info['Fold02']['accuracy score'], 4)}$ | $\left[\begin{a}{evaluation_info['Fold02']['confusion matrix'][0][0]}&{evaluation_info['Fold02']['confusion matrix'][0][1]}\\{evaluation_info['Fold02']['confusion matrix'][1][0]}&{evaluation_info['Fold02']['confusion matrix'][1][1]}\end{a}\right]$ |   ${round(evaluation_info['Fold02']['auc score'], 4)}$   |
| 第$3$折验证结果 | ${round(evaluation_info['Fold03']['accuracy score'], 4)}$ | $\left[\begin{a}{evaluation_info['Fold03']['confusion matrix'][0][0]}&{evaluation_info['Fold03']['confusion matrix'][0][1]}\\{evaluation_info['Fold03']['confusion matrix'][1][0]}&{evaluation_info['Fold03']['confusion matrix'][1][1]}\end{a}\right]$ |   ${round(evaluation_info['Fold03']['auc score'], 4)}$   |
| 第$4$折验证结果 | ${round(evaluation_info['Fold04']['accuracy score'], 4)}$ | $\left[\begin{a}{evaluation_info['Fold04']['confusion matrix'][0][0]}&{evaluation_info['Fold04']['confusion matrix'][0][1]}\\{evaluation_info['Fold04']['confusion matrix'][1][0]}&{evaluation_info['Fold04']['confusion matrix'][1][1]}\end{a}\right]$ |   ${round(evaluation_info['Fold04']['auc score'], 4)}$   |
| 第$5$折验证结果 | ${round(evaluation_info['Fold05']['accuracy score'], 4)}$ | $\left[\begin{a}{evaluation_info['Fold05']['confusion matrix'][0][0]}&{evaluation_info['Fold05']['confusion matrix'][0][1]}\\{evaluation_info['Fold05']['confusion matrix'][1][0]}&{evaluation_info['Fold05']['confusion matrix'][1][1]}\end{a}\right]$ |   ${round(evaluation_info['Fold05']['auc score'], 4)}$   |'''
		
		with open(export_path, 'w', encoding='utf8') as f:
			f.write(markdown_formatter)


