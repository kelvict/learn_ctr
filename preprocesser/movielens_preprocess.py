#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#

import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.externals import joblib
import pandas.core.algorithms as algos
from util import preprocess as preprocess_util

default_1m_path = "dataset/recommend/ml-1m/"
default_1m_rating_path = default_1m_path+"ratings.dat"
default_1m_movies_path = default_1m_path+"movies.dat"
default_1m_users_path = default_1m_path+"users.dat"

default_1m_labels_path = default_1m_path+"labels.csv"
default_1m_timestamp_path = default_1m_path+"timestamp.csv"
default_1m_timeinfo_dummies_path = default_1m_path + "timeinfo_mats.pkl"
default_1m_user_item_field_sizes_path = default_1m_path+"%s_field_sizes.pkl"
default_1m_user_item_csr_mats_path = default_1m_path+"%s_csr_mats.pkl"

default_1m_user_item_train_data_path = default_1m_path+"%s_train_data.pkl"
default_1m_user_item_test_data_path = default_1m_path+"%s_test_data.pkl"

def preprocess_1m():
	ratings_df = pd.read_csv(default_1m_rating_path,sep="::", header=None,
	                      names=["user", "item", "rate", "timestamp"],
	                      engine="python")
	ratings_df = ratings_df.iloc[np.random.permutation(len(ratings_df))].reset_index(drop=True)
	print ratings_df.shape
	ratings = ratings_df['rate'].astype(np.float32)
	ratings.to_csv(default_1m_labels_path, header=None, index=None)
	timestamp = ratings_df['timestamp']

	time_info = timestamp.apply(preprocess_util.extract_datetime_info_from_time_stamp)
	n_day_interval = 100 #365
	n_days_delta_interval = 1000 #1038
	uniq_bins = np.unique(algos.quantile(time_info['day'], np.linspace(0,1,n_day_interval)))
	time_info['day'] = pd.tools.tile._bins_to_cuts(time_info['day'],uniq_bins,include_lowest=True)
	uniq_bins = np.unique(algos.quantile(time_info['days_delta'], np.linspace(0,1,n_days_delta_interval)))
	time_info['days_delta'] = pd.tools.tile._bins_to_cuts(time_info['days_delta'],uniq_bins,include_lowest=True)
	time_info_mats = []
	time_info_field_sizes = []
	for col_name in time_info.columns:
		lbl_enc = LabelEncoder()
		col = lbl_enc.fit_transform(time_info[col_name])
		onehot_enc = OneHotEncoder()
		mat = onehot_enc.fit_transform(np.atleast_2d(col).T)
		time_info_mats.append(mat)
		time_info_field_sizes.append(mat.shape[1])

	timestamp.to_csv(default_1m_timestamp_path, header=None, index=None)

	user_onehot_enc = OneHotEncoder()
	user_mat = user_onehot_enc.fit_transform(
		np.atleast_2d(ratings_df['user'].values).T).tocsr()
	item_onehot_enc = OneHotEncoder()
	item_mat = item_onehot_enc.fit_transform(
		np.atleast_2d(ratings_df['item'].values).T).tocsr()

	field_sizes = [user_mat.shape[1], item_mat.shape[1]]
	dataset = [user_mat, item_mat]
	joblib.dump(field_sizes, default_1m_user_item_field_sizes_path%"rate")
	joblib.dump(dataset, default_1m_user_item_csr_mats_path%"rate")
	preprocess_util.split_train_test_data(
		default_1m_user_item_csr_mats_path%"rate", default_1m_labels_path,
		0.9,
		default_1m_user_item_train_data_path%"rate", default_1m_user_item_test_data_path%"rate")

	field_sizes = field_sizes + time_info_field_sizes
	dataset = dataset + time_info_mats
	joblib.dump(field_sizes, default_1m_user_item_field_sizes_path%"rate_time")
	joblib.dump(dataset, default_1m_user_item_csr_mats_path%"rate_time")
	preprocess_util.split_train_test_data(
		default_1m_user_item_csr_mats_path%"rate_time", default_1m_labels_path,
		0.9,
		default_1m_user_item_train_data_path%"rate_time", default_1m_user_item_test_data_path%"rate_time")

if __name__ == "__main__":
	pass