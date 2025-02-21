# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# 错误分析

import os
import json
import pandas

from setting import *
from config import DatasetConfig

from src.data_tools import encode_answer, decode_answer
from src.dataset import BasicDataset
from src.utils import load_args

from src.evaluation_tools import calc_score_and_accuracy_for_choice_answer

# 展示题目及其答案
def display_answer(answer_path, dataset_path=None, mode='test', do_export=False, display_reference=True):
	
	with open(answer_path, 'r', encoding='utf8') as f:
		answer_dict = json.load(f)
		
	reference_dataframe = pandas.read_csv(REFERENCE_PATH, sep='\t', header=0)
	reference_dataframe['content'] = reference_dataframe['content'].map(lambda _content: ''.join(eval(_content)))

	if dataset_path is None:
		args = load_args(Config=DatasetConfig)																																
		dataset = BasicDataset(args=args, mode=mode, do_export=False, pipeline='display', for_debug=False)
		dataset_dataframe = dataset.data	
		dataset_dataframe.to_csv(f'{mode}_display.csv', header=True, index=False, sep='\t')
	
	else:
		dataset_dataframe = pandas.read_csv(f'{mode}_display.csv', header=0, sep='\t', dtype=str)
		dataset_dataframe['statement'] = dataset_dataframe['statement'].map(eval)
		dataset_dataframe['option_a'] = dataset_dataframe['option_a'].map(eval)
		dataset_dataframe['option_b'] = dataset_dataframe['option_b'].map(eval)
		dataset_dataframe['option_c'] = dataset_dataframe['option_c'].map(eval)
		dataset_dataframe['option_d'] = dataset_dataframe['option_d'].map(eval)
		dataset_dataframe['reference'] = dataset_dataframe['reference'].map(eval)
		dataset_dataframe['reference_index'] = dataset_dataframe['reference_index'].map(eval)
		dataset_dataframe['subject'] = dataset_dataframe['subject'].map(eval)
	
	for i in range(dataset_dataframe.shape[0]):
		_id = dataset_dataframe.loc[i, 'id']
		statement = ''.join(dataset_dataframe.loc[i, 'statement'])
		option_a = ''.join(dataset_dataframe.loc[i, 'option_a'])
		option_b = ''.join(dataset_dataframe.loc[i, 'option_b'])
		option_c = ''.join(dataset_dataframe.loc[i, 'option_c'])
		option_d = ''.join(dataset_dataframe.loc[i, 'option_d'])
		target_encoded_answer = int(dataset_dataframe.loc[i, 'label_choice'])
		target_answer = decode_answer(encoded_answer=target_encoded_answer, result_type=str)
		reference = dataset_dataframe.loc[i, 'reference']
		reference_index = dataset_dataframe.loc[i, 'reference_index']
		subject = dataset_dataframe.loc[i, 'subject']
		print(f'Predicted: {answer_dict[_id]}')
		print(f'Target: {target_answer}')
		print(f'Score: {calc_score_and_accuracy_for_choice_answer(target_encoded_answer=target_encoded_answer, predicted_encoded_answer=encode_answer(decoded_answer=answer_dict[_id]))}')
		print(statement)
		print(f'A: {option_a}')
		print(f'B: {option_b}')
		print(f'C: {option_c}')
		print(f'D: {option_d}')
		print(f'subject: {subject}')
		for _subject in subject:
			if _subject > 0: 
				print(f'  - {INDEX2SUBJECT[_subject - 1]}')
				
		if display_reference:
			print(f'reference subject: {len(reference)}')
			for _reference in reference:
				print(f'  - {"".join(_reference)}')
				print('*' * 64)
			print(f'reference total: {len(reference_index)}')
			for index in reference_index:
				print(f'  - {reference_dataframe.loc[index, "content"]}')
				print('*' * 64)
		print('#' * 64)
		input('pause ... ')

	
if __name__ == '__main__':
	mode = 'valid'
	answer_path = 'temp/answer_BaseChoiceModel.json'
	dataset_path = 'valid_display.csv'
	display_answer(answer_path=answer_path, dataset_path=dataset_path, mode='valid', do_export=False, display_reference=False)
