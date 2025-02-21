# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
# 其他工具函数

if __name__ == '__main__':
	import sys
	sys.path.append('../')

import os
import time
import json
import logging

from setting import *

from functools import wraps

# 程序计时的装饰器
def timer(function):
	@wraps(function)
	def wrapper(*args, **kwargs):
		start_time = time.time()
		function_return = function(*args, **kwargs)
		end_time = time.time()
		print('Function `{}` runtime is {} seconds.'.format(function.__name__, end_time - start_time))
		return function_return
	return wrapper

# 初始化日志配置
def initialize_logger(filename, filemode='w'):
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)

	formatter = logging.Formatter('%(asctime)s | %(filename)s | %(levelname)s | %(message)s')
	file_handler = logging.FileHandler(filename, mode='w', encoding='utf8')
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	console.setFormatter(formatter)
	logger.addHandler(console)
	
	return logger

# 终止日志句柄
def terminate_logger(logger):
	for handler in logger.handlers[:]:
		logger.removeHandler(handler)

# 加载配置参数
def load_args(Config):
	config = Config()
	parser = config.parser
	try:
		return parser.parse_args()			# 常规加载配置参数
	except:
		return parser.parse_known_args()[0]	# JupyterNotebook中加载配置参数

# 保存配置参数
def save_args(args, save_path=None):

	class _MyEncoder(json.JSONEncoder):
		"""自定义的特殊类型数据的序列化编码器"""
		def default(self, obj):
			if isinstance(obj, type) or isinstance(obj, types.FunctionType):
				return str(obj)
			return json.JSONEncoder.default(self, obj)
			
	if save_path is None:
		save_path = os.path.join(LOGGING_DIR, f'config_{time.strftime("%Y%m%d%H%M%S")}.json')
	with open(save_path, 'w') as f:
		f.write(json.dumps(vars(args), cls=_MyEncoder))

# 判断字符是否为汉字
def is_chinese(uchar):
	if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
		return True
	return False

# 判断字符是否为数字
def is_number(uchar):
	if uchar >= u'\u0030' and uchar <= u'\u0039':
		return True
	return False

# 判断字符是否为字母
def is_alphabet(uchar):
	if uchar >= u'\u0041' and uchar <= u'\u005a':
		# 
		return True
	if uchar >= u'\u0061' and uchar <= u'\u007a':
		return True
	return False

# 判断字符是否为特殊符号
def is_symbol(uchar):
	if is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar):
		return False
	return True


