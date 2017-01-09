#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#
import multiprocessing
import pandas as pd
import numpy as np
import os
import sys
from util import log
def _apply_df(params):
	df = pd.DataFrame()

	df, func, axis, args, kwargs = params
	log.log("%s apply to data with shape %s"%(os.getpid(), df.shape))
	df = df.apply(func, axis, args=args, **kwargs)
	log.log("%s finish applying to data with shape %s"%(os.getpid(), df.shape))
	return df

def apply_by_multiprocessing(df, func, axis = 0, processes = 1, args=(), **kwargs):
	if processes > 1:
		pool = multiprocessing.Pool(processes=processes)
		manager = multiprocessing.Manager()
		ns = manager.Namespace()
		ns.params = [(df_part, func, axis, args, kwargs)
						   for df_part in np.array_split(df, processes, axis=1 if axis == 0 else 0)]
		result = pool.map(_apply_df,
						  ns.params)
		pool.close()
		return pd.concat(list(result), axis=1 if axis == 0 else 0)
	else:
		return df.apply(func,axis=axis,args=args,**kwargs)

cnt = {}
def test_func(x):
	global cnt
	if os.getpid() in cnt:
		cnt[os.getpid()] += 1
	else:
		cnt[os.getpid()] = 1

	result = (x*1000000+os.getpid()) * 100 + cnt[os.getpid()]
	sys.stdout.flush()
	return result

def test_apply_by_multiprocessing(df):
	print df
	result = apply_by_multiprocessing(df, test_func, axis=0, processes=4)
	print result

def test_apply_by_single_processing(df):
	print df
	result = df.apply(test_func, axis=1)
	print result

def join_files(filenames, output_path, sep = " "):
	try:
		fins = []
		for filename in filenames:
			f = open(filename)
			fins.append(f)
		fout = open(output_path, mode="w+")
		some_fin_ends = False
		while not some_fin_ends:
			newline = ""
			for fin in fins:
				cur_line = fin.readline()
				if len(cur_line) == 0:
					some_fin_ends = True
					break
				newline = newline + sep + cur_line[:-1]
			if not some_fin_ends:
				fout.write(newline + '\n')
		for fin in fins:
			fin.close()
		fout.close()
	except IOError,e:
		log.log("Fuck! IOError")

if __name__ == "__main__":
	df = pd.DataFrame([range(10,20,1), range(20,30,1)])
	test_apply_by_multiprocessing(df)

