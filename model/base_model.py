#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#

class BaseModel:
	def __init__(self):
		pass

	def run(self, fetches, X=None, y=None):
		raise NotImplementedError

	def dump(self, model_path):
		raise NotImplementedError