#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#

import pandas as pd
import numpy as np
import json
import copy
from util import preprocess as preproc
from sklearn.externals import joblib
prefix = "dataset/recommend/yelp/"
academic_dataset_json_prefix = "yelp_academic_dataset_%s.json"

from util.log import log_and_print
import random
import gc

def build_df(objs, keys):
	df_dict = {}
	for key in keys:
		df_dict[key] = [obj[key] for obj in objs]
	df = pd.DataFrame(df_dict)
	return df

def get_reviews_users_businesses(is_test=True):
	reviews = []
	businesses = []
	users = []

	for line in open(prefix+academic_dataset_json_prefix%("review" if not is_test else "review_10k")):
		reviews.append(json.loads(line))
	for line in open(prefix+academic_dataset_json_prefix%("user")):
		users.append(json.loads(line))
	for line in open(prefix+academic_dataset_json_prefix%("business")):
		businesses.append(json.loads(line))
	return reviews, users, businesses

def get_reviews_users_businesses_photos(is_test=True):
	reviews, users, businesses= get_reviews_users_businesses(is_test)
	f = open(prefix+"photo_id_to_business_id.json")
	photos = json.load(f)
	return reviews, users, businesses, photos

def get_uid_to_bids_map(reviews):
	uid_to_bids = {}
	for review in reviews:
		if review['user_id'] not in uid_to_bids:
			uid_to_bids[review['user_id']] = [review['business_id']]
		else:
			uid_to_bids[review['user_id']].append(review['business_id'])
	return uid_to_bids

def get_uid_to_reviews(reviews):
	uid_to_reviews = {}
	for review in reviews:
		if review['user_id'] not in uid_to_reviews:
			uid_to_reviews[review['user_id']] = [review]
		else:
			uid_to_reviews[review['user_id']].append(review)
	return uid_to_reviews

def get_bid_to_business_map(businesses):
	bid_to_business_map = {}
	for business in businesses:
		bid_to_business_map[business['business_id']] = business
	return bid_to_business_map

def get_uid_to_user_map(users):
	uid_to_user_map = {}
	for user in users:
		uid_to_user_map[user['user_id']] = user
	return uid_to_user_map

def filter_with_set(objs, key, field_set):
	return [obj for obj in objs if obj[key] in field_set]

def get_server_data(is_test=False, min_reviews_cnt=20):
	reviews, users, businesses, photos = get_reviews_users_businesses_photos(is_test)
	print "Data loaded"
	reviews_5p = [review for review in reviews if review['stars'] == 5]
	#inside_photos = [photo for photo in photos if photo['label']=="outside"]
	#inside_photos_bids = set([photo['business_id'] for photo in inside_photos])
	#reviews_5p_in_bids = [review for review in reviews_5p if review['business_id'] in inside_photos_bids]
	reviews_5p_in_bids = reviews_5p
	print "get_uid_to_reviews"
	#uid_to_reviews_map = get_uid_to_reviews(reviews_5p_in_bids)
	uid_to_reviews_map = get_uid_to_reviews(reviews_5p_in_bids)
	limit_uid_to_reviews_map = {}
	if min_reviews_cnt > 0:
		for key in uid_to_reviews_map:
			if len(uid_to_reviews_map[key]) > min_reviews_cnt:
				limit_uid_to_reviews_map[key] =  uid_to_reviews_map[key]
	uid_to_user_map = get_uid_to_user_map(users)
	bid_to_business_map = get_bid_to_business_map(businesses)
	return limit_uid_to_reviews_map, uid_to_user_map, bid_to_business_map

def get_user_page_data(i, limit_uid_to_reviews_map, uid_to_user_map, bid_to_business_map):
	reviews = limit_uid_to_reviews_map[limit_uid_to_reviews_map.keys()[i%len(limit_uid_to_reviews_map.keys())]]
	reviews = sorted(reviews, key=lambda d:d['date'])
	user = uid_to_user_map[reviews[0]['user_id']]
	review_records = []

	for review in reviews:
		review['business'] = bid_to_business_map[review['business_id']]
		review_records.append(review)

	visit_records = []
	rec_records = []
	records_size = len(review_records)
	if records_size < 10:
		rec_records = review_records
	else:
		rec_records = review_records[-10:]
		visit_records = visit_records[-20:-10]


	return {
		'user':user,
		"rec_records":rec_records,
		"visit_records":visit_records
	}


def get_business_attr_key_values(businesses):
	key_to_option = {}
	key_to_type = {}
	for b in businesses:
		if "attributes" in b and b['attributes'] is not None:
			for attr in b['attributes']:
				name = attr[:attr.find(": ")]
				value = attr[attr.find(": ")+2:]
				if "{" not in value:
					if name not in key_to_option:
						key_to_option[name] = set()
						key_to_type[name] = "exclusive"
					key_to_option[name].add(value)
				else:
					values = value[value.find("{")+1:value.find("}")].split(", ")
					for value in values:
						if name not in key_to_option:
							key_to_option[name] = set()
							key_to_type[name] = "multiple"
						key_to_option[name].add(value)
	return key_to_option, key_to_type

def get_business_attr_value_df(businesses):
	key_to_option, key_to_type = get_business_attr_key_values(businesses)
	business_attr_dict = {}
	for key in key_to_option.keys():
		business_attr_dict[key] = []
	for b in businesses:
		unhandled_keys = key_to_option.keys()
		if "attributes" in b and b['attributes'] is not None:
			for attr in b['attributes']:
				name = attr[:attr.find(": ")]
				value = attr[attr.find(": ")+2:]

				unhandled_keys.remove(name)
				type = key_to_type[name]
				if type == "exclusive":
					business_attr_dict[name].append(value)
				elif type == "multiple":
					values = value[value.find("{")+1:value.find("}")].split(", ")
					values_set = set()
					for value in values:
						values_set.add(value)
					business_attr_dict[name].append(values_set)
		for key in unhandled_keys:
			type = key_to_type[key]
			if type == "exclusive":
				business_attr_dict[key].append(None)
			elif type == "multiple":
				business_attr_dict[key].append(set())
	return pd.DataFrame(business_attr_dict)

def preprocess(random_seed=0, trainset_rate=0.9, is_test=False, n_friend_sample=10, suffix=""):
	attr_str = str(n_friend_sample)+"_"+str(random_seed)+"_"+str(trainset_rate).replace(".","p")\
	           +("_test" if is_test else "") + (suffix if len(suffix)==0 else "_"+suffix)
	output_pattern = prefix+"%s_"+attr_str+".json"
	field_sizes_output_pattern = prefix+"%s_field_sizes_"+attr_str+".pkl"
	csr_mats_output_pattern = prefix+"%s_csr_mats_"+attr_str+".pkl"
	train_data_output_pattern = prefix+"%s_train_data_"+attr_str+".pkl"
	test_data_output_pattern = prefix+"%s_test_data_"+attr_str+".pkl"
	label_output_path = prefix+"labels_"+attr_str+".csv"
	reviews = []
	businesses = []
	users = []
	log_and_print("Loading Reviews")
	for line in open(prefix+academic_dataset_json_prefix%("review" if not is_test else "review_10k")):
		reviews.append(json.loads(line))
	log_and_print("Loading Users")
	for line in open(prefix+academic_dataset_json_prefix%("user")):
		users.append(json.loads(line))
	log_and_print("Loading Business")
	for line in open(prefix+academic_dataset_json_prefix%("business")):
		businesses.append(json.loads(line))

	log_and_print("Build Review DF")
	#Handle reviews
	review_keys = [u'funny', u'user_id', u'review_id', u'text',
	               u'business_id', u'stars', u'date', u'useful',
	               u'type', u'cool']
	review_keys.remove("type")
	review_keys.remove("text")
	review_keys.remove("review_id")
	reviews_df = build_df(reviews, keys=review_keys)
	rbids = reviews_df['business_id'].values.tolist()
	ruids = reviews_df['user_id'].values.tolist()
	del reviews
	gc.collect()
	log_and_print("Shuffle Reviews")
	#Shuffle
	if random_seed is not None:
		np.random.seed(random_seed)
	reviews_df = reviews_df.iloc[np.random.permutation(len(reviews_df))].reset_index(drop=True)

	log_and_print("Dump Label")
	labels = reviews_df['stars'].astype(np.float32)
	labels.to_csv(label_output_path, header=None, index=None)

	rate_mats = []
	exclusive_discrete_keys = [u'user_id', u'business_id']
	for key in exclusive_discrete_keys:
		log_and_print("Handle %s in Rate Mats"%key)
		rate_mats.append(preproc.get_csr_mat_from_exclusive_field(reviews_df[key]))
	#["funny","useful","cool","day","days_delta","year","month","weekend"]
	review_mats = []
	log_and_print("Handle Contin Feat in Review Mats")
	review_mats.append(preproc.get_csr_mat_from_contin_field(reviews_df['funny'],n_interval=30))
	review_mats.append(preproc.get_csr_mat_from_contin_field(reviews_df['useful'],n_interval=30))
	review_mats.append(preproc.get_csr_mat_from_contin_field(reviews_df['cool'],n_interval=30))

	review_time_info_df = reviews_df[u'date'].apply(preproc.extract_date_info_from_str)
	log_and_print("Handle Date in Review Mats")
	review_mats.append(preproc.get_csr_mat_from_contin_field(review_time_info_df['day'], n_interval=100, estimate_interval_size=10))
	review_mats.append(preproc.get_csr_mat_from_contin_field(review_time_info_df['days_delta'], n_interval=100, estimate_interval_size=15))
	review_time_info_keys = review_time_info_df.keys().tolist()
	review_time_info_keys.remove('day')
	review_time_info_keys.remove('days_delta')
	for key in review_time_info_keys:
		log_and_print("Handle %s in Review Mats"%key)
		review_mats.append(
			preproc.get_csr_mat_from_exclusive_field(review_time_info_df[key])
		)
	del reviews_df
	del review_time_info_df
	gc.collect()
	#Handle
	user_keys = [u'yelping_since', u'useful', u'compliment_photos',
	             u'compliment_list', u'compliment_funny', u'funny',
	             u'review_count', u'friends', u'fans', u'type',
	             u'compliment_note', u'compliment_plain', u'compliment_writer',
	             u'compliment_cute', u'average_stars', u'user_id',
	             u'compliment_more', u'elite', u'compliment_hot',
	             u'cool', u'name', u'compliment_profile', u'compliment_cool']
	user_keys.remove("name")
	user_keys.remove("type")
	if n_friend_sample <= 0:
		user_keys.remove("friends")
	log_and_print("Build User df")
	users_df = build_df(users, keys= user_keys)
	del users
	users_df.set_index(users_df['user_id'])
	user_mats = []

	#Sample Friends
	if n_friend_sample > 0 and n_friend_sample < 15000:
		log_and_print("Sample Friends in User df")
		users_df['friends'] = users_df['friends'].apply(
			lambda x:random.sample(x, n_friend_sample)
			if len(x) > n_friend_sample else x)
	if n_friend_sample > 0:
		log_and_print("To CSR Mat of Friends in User df")
		user_mats.append(preproc.get_csr_mat_from_multiple_field(users_df['friends']))
	contin_field_names_with_30_intervals = [u'useful', u'compliment_photos',
	             u'compliment_list', u'compliment_funny', u'funny',
	             u'review_count', u'fans',
	             u'compliment_note', u'compliment_plain', u'compliment_writer',
	             u'compliment_cute',
	             u'compliment_more',  u'compliment_hot',
	             u'cool', u'compliment_profile', u'compliment_cool']

	for key in contin_field_names_with_30_intervals:
		log_and_print("Handle %s in User Mats"%key)
		user_mats.append(preproc.get_csr_mat_from_contin_field(users_df[key], n_interval=30))

	log_and_print("Handle %s in Review Mats"%"average_stars")
	user_mats.append(preproc.get_csr_mat_from_contin_field(
		users_df["average_stars"], uniq_bins=[i/2.0 for i in range(0, 12)]))
	log_and_print("Handle %s in Review Mats"%"Yelping Since")
	yelp_time_info_df = users_df['yelping_since'].apply(preproc.extract_date_info_from_str)
	user_mats.append(preproc.get_csr_mat_from_contin_field(yelp_time_info_df['day'], n_interval=20,estimate_interval_size=15))
	user_mats.append(preproc.get_csr_mat_from_contin_field(yelp_time_info_df['days_delta'], n_interval=30, estimate_interval_size=5))

	yelp_time_info_keys = yelp_time_info_df.keys().tolist()
	yelp_time_info_keys.remove('day')
	yelp_time_info_keys.remove('days_delta')

	for key in yelp_time_info_keys:
		log_and_print("Handle %s in Review Mats"%key)
		user_mats.append(preproc.get_csr_mat_from_exclusive_field(yelp_time_info_df[key]))

	uids = users_df['user_id'].values.tolist()

	del users_df
	del yelp_time_info_df
	gc.collect()

	log_and_print("Join Expand User Mats")
	uids_to_idx = {}
	for i in xrange(len(uids)):
		uids_to_idx[uids[i]] = i
	for i in xrange(len(user_mats)):
		user_mats[i] = preproc.join_expand(user_mats[i], ruids, uids_to_idx)

	#Handle business
	business_keys = [u'city', u'neighborhood', u'name', u'business_id', u'longitude',
	                 u'hours', u'state', u'postal_code', u'categories', u'stars',
	                 u'address', u'latitude', u'review_count', u'attributes',
	                 u'type', u'is_open']
	business_keys.remove("name")
	business_keys.remove("type")
	business_keys.remove("address")
	business_keys.remove("hours")
	business_keys.remove("attributes")
	log_and_print("Build businesses_df")
	businesses_df = build_df(businesses, keys=business_keys)
	businesses_df.set_index(businesses_df['business_id'])

	log_and_print("get_business_attr_value_df")
	business_attr_df = get_business_attr_value_df(businesses)
	del businesses
	gc.collect()
	business_attr_df.set_index(businesses_df['business_id'])

	business_mats = []

	exclusive_discrete_keys = [u'city', u'neighborhood', u'state', u'postal_code', u'stars', u'is_open']
	for key in exclusive_discrete_keys:
		log_and_print("Handle %s in business df"%key)
		business_mats.append(preproc.get_csr_mat_from_exclusive_field(businesses_df[key]))
	business_mats.append(preproc.get_csr_mat_from_multiple_field(businesses_df['categories']))
	for key in business_attr_df.keys():
		log_and_print("Handle %s in business attr df"%key)
		if business_attr_df[key][0] is set:
			business_mats.append(preproc.get_csr_mat_from_multiple_field(business_attr_df[key]))
		else:
			business_mats.append(preproc.get_csr_mat_from_exclusive_field(business_attr_df[key]))
		log_and_print("Handle %s in business df"%"contine field")
	business_mats.append(preproc.get_csr_mat_from_contin_field(businesses_df['longitude'], n_interval=100))
	business_mats.append(preproc.get_csr_mat_from_contin_field(businesses_df['latitude'], n_interval=100))
	business_mats.append(preproc.get_csr_mat_from_contin_field(businesses_df['review_count'], n_interval=30))
	bids = businesses_df['business_id'].values.tolist()

	del businesses_df
	del business_attr_df
	gc.collect()

	log_and_print("Join Expand Business")
	bids_to_idx = {}
	for i in xrange(len(bids)):
		bids_to_idx[bids[i]] = i
	for i in xrange(len(business_mats)):
		business_mats[i] = preproc.join_expand(business_mats[i], rbids, bids_to_idx)


	name = "rate"
	log_and_print("Dumping %s"%name)
	dataset = rate_mats
	field_sizes = preproc.get_field_sizes(dataset)
	joblib.dump(field_sizes, field_sizes_output_pattern%name)
	joblib.dump(dataset, csr_mats_output_pattern%name)
	preproc.split_train_test_data(
		csr_mats_output_pattern%name, label_output_path,
		trainset_rate,
		train_data_output_pattern%name, test_data_output_pattern%name
	)

	name = "rate_review_user_business"
	log_and_print("Dumping %s"%name)
	dataset = rate_mats + review_mats + user_mats + business_mats
	field_sizes = preproc.get_field_sizes(dataset)
	joblib.dump(field_sizes, field_sizes_output_pattern%name)
	joblib.dump(dataset, csr_mats_output_pattern%name)
	preproc.split_train_test_data(
		csr_mats_output_pattern%name, label_output_path,
		trainset_rate,
		train_data_output_pattern%name, test_data_output_pattern%name
	)









