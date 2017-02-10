#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib

from util import preprocess as preprocess_util

default_1m_path = "dataset/recommend/ml-1m/"
default_1m_rating_path = default_1m_path+"ratings.dat"
default_1m_movies_path = default_1m_path+"movies.dat"
default_1m_users_path = default_1m_path+"users.dat"

default_1m_labels_path = default_1m_path+"labels.csv"
default_1m_timestamp_path = default_1m_path+"timestamp.csv"
default_1m_user_item_field_sizes_path = default_1m_path+"user_item_field_sizes.pkl"
default_1m_user_item_csr_mats_path = default_1m_path+"user_item_csr_mats.pkl"

default_1m_user_item_train_data_path = default_1m_path+"user_item_train_data.pkl"
default_1m_user_item_test_data_path = default_1m_path+"user_item_test_data.pkl"
def preprocess_1m():
	ratings_df = pd.read_csv(default_1m_rating_path,sep="::", header=None,
	                      names=["user", "item", "rate", "timestamp"],
	                      engine="python")
	ratings_df = ratings_df.iloc[np.random.permutation(len(ratings_df))].reset_index(drop=True)

	ratings = ratings_df['rate'].astype(np.float32)
	ratings.to_csv(default_1m_labels_path, header=None, index=None)
	timestamp = ratings_df['timestamp']
	timestamp.to_csv(default_1m_timestamp_path, header=None, index=None)
	user_onehot_enc = OneHotEncoder()
	user_mat = user_onehot_enc.fit_transform(
		np.atleast_2d(ratings_df['user'].values).T).tocsr()
	item_onehot_enc = OneHotEncoder()
	item_mat = item_onehot_enc.fit_transform(
		np.atleast_2d(ratings_df['item'].values).T).tocsr()
	field_sizes = [user_mat.shape[1], item_mat.shape[1]]
	dataset = [user_mat, item_mat]
	joblib.dump(field_sizes, default_1m_user_item_field_sizes_path)
	joblib.dump(dataset, default_1m_user_item_csr_mats_path)
	preprocess_util.split_train_test_data(
		default_1m_user_item_csr_mats_path, default_1m_labels_path,
		0.9,
		default_1m_user_item_train_data_path, default_1m_user_item_test_data_path)

if __name__ == "__main__":
	pass