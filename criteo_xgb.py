#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#
import glob
from sklearn.externals import joblib
from scipy import sparse
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from util.log import print_with_time

def combine_data():
	disc_paths = glob.glob("dataset/ctr/criteo/*20170111*discrete*one_hot_discrete*_in_*.pkl")
	contin_path = glob.glob("dataset/ctr/criteo/*20170111*contin_df.pkl")[0]

	contin_df = joblib.load(contin_path)
	[path for path in disc_paths]

default_xgb_params = {
    'max_depth': 6,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.8,
    'objective':'binary:logistic',
    'gamma': 0.1,
	'eval_metric':'auc'
}

def train_with_xgb():
	print_with_time("Load data")
	df = pd.read_csv("dataset/ctr/criteo/dac_sample.txt", sep="\t", header=None)

	print_with_time("split data")

	df.columns = ['label'] + ["I_"+str(i) for i in xrange(1,14)] + ["C_"+str(i) for i in xrange(14,40)]
	line_num = df.shape[0]*0.9
	train_labels = df.iloc[:line_num, 0]
	test_labels = df.iloc[line_num:, 0]
	train_data_df = df.iloc[:line_num,1:]
	test_data_df = df.iloc[line_num:,1:]

	print_with_time("Set DMatrix")

	n_round = 30
	train_data = xgb.DMatrix(train_data_df.values, train_labels.values)
	test_data = xgb.DMatrix(test_data_df.values, test_labels.values)

	print_with_time("train data")

	bst = xgb.train(default_xgb_params, train_data, n_round,
	                evals=[(train_data, 'train'), (test_data, 'test')], verbose_eval=5)
	print_with_time("predict data")
	y_pred = bst.predict(test_data)
	print_with_time("cal auc")
	auc = roc_auc_score(test_labels.values, y_pred)
	print auc
	print_with_time("Finish all")

if __name__ == "__main__":
	train_with_xgb()