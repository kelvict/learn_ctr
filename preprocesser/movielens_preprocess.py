#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#

import pandas as pd
import numpy as np
import datetime
from scipy.sparse import vstack
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MultiLabelBinarizer
from sklearn.externals import joblib
import pandas.core.algorithms as algos
from util import preprocess as preprocess_util
from util.log import log, now_str

import gc
import copy

default_1m_path = "dataset/recommend/ml-1m/"
default_1m_rating_path = default_1m_path+"ratings.dat"
default_1m_movies_path = default_1m_path+"movies.dat"
default_1m_users_path = default_1m_path+"users.dat"

default_1m_labels_path = default_1m_path+"labels.csv"
default_1m_timestamp_path = default_1m_path+"timestamp.csv"
default_1m_pre_rates_path = default_1m_path+"pre_rates.pkl"
default_1m_timeinfo_dummies_path = default_1m_path + "timeinfo_mats.pkl"
default_1m_user_item_field_sizes_path = default_1m_path+"%s_field_sizes.pkl"
default_1m_user_item_csr_mats_path = default_1m_path+"%s_csr_mats.pkl"

default_1m_user_item_train_data_path = default_1m_path+"%s_train_data.pkl"
default_1m_user_item_test_data_path = default_1m_path+"%s_test_data.pkl"

def get_pre_rate(ratings_df):
	ratings_df.sort_values(['user','timestamp'], inplace=True)
	ratings_df.reset_index(inplace=True)
	pre_rate = []
	for i in xrange(5):
		pre_rate.append([[]])
	for i in xrange(1,ratings_df.shape[0]):
		if i%10000==1:
			print now_str(), " make rating list ", i
		if ratings_df['user'][i-1] != ratings_df['user'][i]:
			for j in xrange(len(pre_rate)):
				pre_rate[j].append([])
		else:
			for j in xrange(len(pre_rate)):
				pre_rate[j].append(copy.copy(pre_rate[j][i-1]))
				if ratings_df['rate'][i-1] == j+1:
					pre_rate[j][i].append(ratings_df['item'][i-1])
	return pre_rate

def make_dataframe_with_pre_rate(pre_rate,cnt_limit=20):
	for i in xrange(len(pre_rate)):
		for j in xrange(len(pre_rate[i])):
			pre_rate[i][j] = set(pre_rate[i][j][-cnt_limit:])
	return pd.DataFrame(pre_rate,index=["Rate_%d"%(i+1) for i in range(len(pre_rate))]).T

def preprocess_1m(random_seed=None):
	ratings_df = pd.read_csv(default_1m_rating_path,sep="::", header=None,
	                      names=["user", "item", "rate", "timestamp"],
	                      engine="python")
	#ratings_df = ratings_df.iloc[:3000,:]
	if random_seed is not None:
		np.random.seed(random_seed)
	#Handle Prev Rate
	pre_rate = get_pre_rate(ratings_df)
	joblib.dump(pre_rate, default_1m_pre_rates_path)
	pre_rate_df = make_dataframe_with_pre_rate(pre_rate,cnt_limit=20)
	del pre_rate
	gc.collect()

	ratings_df = pd.concat([ratings_df, pre_rate_df], axis=1)

	#Shuffle
	ratings_df = ratings_df.iloc[np.random.permutation(len(ratings_df))].reset_index(drop=True)

	print ratings_df.shape
	ratings = ratings_df['rate'].astype(np.float32)
	ratings.to_csv(default_1m_labels_path, header=None, index=None)
	timestamp = ratings_df['timestamp']


	#Handle time info
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

	#Handle User
	users_df = pd.read_csv(default_1m_users_path,sep="::", header=None,
	                       names=["uid", "gender", "age", "job", "zipcode"],
	                       engine="python")
	lbl_enc = LabelEncoder()
	onehot_enc = OneHotEncoder()
	gender_mat = onehot_enc.fit_transform(
		np.atleast_2d(lbl_enc.fit_transform(users_df['gender'])).T)
	age_mat = onehot_enc.fit_transform(
		np.atleast_2d(lbl_enc.fit_transform(users_df['age'])).T)
	job_mat = onehot_enc.fit_transform(
		np.atleast_2d(lbl_enc.fit_transform(users_df['job'])).T)
	zipcode_mat = onehot_enc.fit_transform(
		np.atleast_2d(lbl_enc.fit_transform(users_df['zipcode'])).T)
	uids = users_df['uid'].values.tolist()
	uids_to_idx = {}
	for i in xrange(len(uids)):
		uids_to_idx[uids[i]] = i
	uids = ratings_df['user'].values.tolist()
	gender_mat = vstack([gender_mat[uids_to_idx[uids[i]]] for i in xrange(len(uids))])
	age_mat = vstack([age_mat[uids_to_idx[uids[i]]] for i in xrange(len(uids))])
	job_mat = vstack([job_mat[uids_to_idx[uids[i]]] for i in xrange(len(uids))])
	zipcode_mat = vstack([zipcode_mat[uids_to_idx[uids[i]]] for i in xrange(len(uids))])
	user_mats = [gender_mat, age_mat, job_mat, zipcode_mat]
	user_field_sizes = [gender_mat.shape[1], age_mat.shape[1], job_mat.shape[1], zipcode_mat.shape[1]]
	for mat in user_mats:
		print mat.shape
		log(str(mat.shape))
	#Handle Movies
	movies_df = pd.read_csv(default_1m_movies_path,sep="::", header=None,
	                        names=["mid", "name&year", "types"],
	                        engine="python")
	movies_df["year"] = [string[-5:-1] for string in movies_df['name&year'].values]
	movies_df["name"] = [string[:-7] for string in movies_df['name&year'].values]
	types = [set(string.split('|')) for string in movies_df['types']]
	mlb = MultiLabelBinarizer(sparse_output=True)
	movie_type_mat = mlb.fit_transform(types)
	lbl_enc = LabelEncoder()
	onehot_enc = OneHotEncoder()
	year_col = lbl_enc.fit_transform(movies_df['year'])
	movie_year_mat = onehot_enc.fit_transform(np.atleast_2d(year_col).T)

	mids = movies_df['mid'].values.tolist()
	mids_to_idx = {}
	for i in xrange(len(mids)):
		mids_to_idx[mids[i]] = i
	mids = ratings_df['item'].values.tolist()
	movie_type_mat = vstack([movie_type_mat[mids_to_idx[mids[i]]] for i in xrange(len(mids))])
	movie_year_mat = vstack([movie_year_mat[mids_to_idx[mids[i]]] for i in xrange(len(mids))])
	movie_mats = [movie_type_mat, movie_year_mat]
	movie_field_sizes = [movie_type_mat.shape[1], movie_year_mat.shape[1]]
	for mat in movie_mats:
		print mat.shape
		log(str(mat.shape))

	#Encode Prev Rates
	pre_rate_mats = []
	for i in xrange(5):
		mlb = MultiLabelBinarizer(sparse_output=True)
		pre_rate_mats.append(mlb.fit_transform(ratings_df["Rate_%d"%(i+1)].values))
	pre_rate_field_sizes = [mat.shape[1] for mat in pre_rate_mats]

	#Encode User and item
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

	field_sizes = field_sizes + user_field_sizes + movie_field_sizes
	dataset = dataset + user_mats + movie_mats
	joblib.dump(field_sizes, default_1m_user_item_field_sizes_path%"rate_time_user_movie")
	joblib.dump(dataset, default_1m_user_item_csr_mats_path%"rate_time_user_movie")
	preprocess_util.split_train_test_data(
		default_1m_user_item_csr_mats_path%"rate_time_user_movie", default_1m_labels_path,
		0.9,
		default_1m_user_item_train_data_path%"rate_time_user_movie", default_1m_user_item_test_data_path%"rate_time_user_movie")

	field_sizes = field_sizes + pre_rate_field_sizes
	dataset = dataset + pre_rate_mats
	joblib.dump(field_sizes, default_1m_user_item_field_sizes_path%"rate_time_user_movie_pre_rate")
	joblib.dump(dataset, default_1m_user_item_csr_mats_path%"rate_time_user_movie_pre_rate")
	preprocess_util.split_train_test_data(
		default_1m_user_item_csr_mats_path%"rate_time_user_movie_pre_rate", default_1m_labels_path,
		0.9,
		default_1m_user_item_train_data_path%"rate_time_user_movie_pre_rate", default_1m_user_item_test_data_path%"rate_time_user_movie_pre_rate")

if __name__ == "__main__":
	preprocess_1m()