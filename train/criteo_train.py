#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score

import numpy as np
import json
import util
import time
import copy

from model.product_nets import LR, FM, FNN, CCPM, PNN1, PNN2, RecIPNN, biasedMF
from scipy.sparse import vstack
from util import preprocess as preproc
CRITEO_MODELS = [LR, FM, FNN, CCPM, PNN1, PNN2, RecIPNN, biasedMF]
SPLIT_BY_FIELD_MODELS = [FNN, CCPM, PNN1, PNN2, RecIPNN, biasedMF]
CTR_MODELS = [LR, FM, FNN, CCPM, PNN1, PNN2]
REC_MODELS = [RecIPNN, biasedMF]
def create_default_conf(dump_path, model_name=LR.__name__):
	for Model in CRITEO_MODELS:
		if model_name == Model.__name__:
			return util.train.create_conf(
				dump_path, model_name, model_params=Model.default_params,
				trainset_csr_pkl_path="dataset/ctr/criteo/trainset_csr.pkl",
				testset_csr_pkl_path="dataset/ctr/criteo/testset_csr.pkl",
				labels_pkl_path="dataset/ctr/criteo/labels.pkl",
				field_sizes_pkl_path="dataset/ctr/criteo/field_sizes.pkl"
			)
	return None

def test_dfs_make_confs():
	conf = {
		"model_params":{'a':1, 'b':1, 'c':1, 'd':1},
		"other_params":[1,2,3,4]
	        }
	grid_param_conf = {
		'a':[1,2,3],
		'b':[[1],[2],[3]],
		'd':['a','j']
	}
	confs = []
	dfs_make_confs(conf, 0, grid_param_conf, confs)
	print confs

def dfs_make_confs(conf, depth, grid_param_conf, confs):
	if depth >= len(grid_param_conf.keys()):
		confs.append(copy.deepcopy(conf))
		print "Add conf"
		print conf
		print confs
		return
	cur_key = grid_param_conf.keys()[depth]
	for param in grid_param_conf[cur_key]:
		print param
		conf['model_params'][cur_key] = param
		dfs_make_confs(conf, depth+1,grid_param_conf, confs)

def modify_conf_with_params(conf, params):
	if 'lr' in params:
		conf['model_params']['learning_rate'] = params['lr']
	if 'n_embd' in params:
		if conf['model_name'] == 'biasedMF':
			conf['model_params']['embd_size'] = params['n_embd']
		else:
			conf['model_params']['layer_sizes'][1] = params['n_embd']
	if 'width' in params and conf['model_name'] != 'biasedMF':
		for i in range(2, len(conf['model_params']['layer_sizes'])-1):
			conf['model_params']['layer_sizes'][i] = params['width']
	if 'act' in params and conf['model_name'] != 'biasedMF':
		for i in range(2, len(conf['model_params']['layer_sizes'])-1):
			conf['model_params']['layer_acts'][i] = params['act'] if params['act'] != "null" else None
	if 'reg' in params:
		if conf['model_name'] != 'biasedMF':
			for i in xrange(len(conf['model_params']['layer_l2'])):
				conf['model_params']['layer_l2'][i] = params['reg']
			conf['model_params']['kernel_l2'] = params['reg']
		else:
			conf['model_params']['reg_rate'] = params['reg']
	if 'dropout' in params and conf['model_name'] != 'biasedMF':
		for i in range(1,len(conf['model_params']['layer_keeps'])):
			conf['model_params']['layer_keeps'][i] = params['dropout']
	if 'data_suffix' in params:
		path_names = ['trainset_csr_pkl_path', 'testset_csr_pkl_path', 'field_sizes_pkl_path']
		for name in path_names:
			conf[name] += params['data_suffix']

def train_model_with_conf(conf_path, grid_param_conf_path=None, ctr_or_recommend=True, predict_batch_size=10000, params=None):
	fi = open(conf_path)
	conf = json.load(fi)
	fi.close()
	confs = []
	if grid_param_conf_path is not None:
		fi = open(grid_param_conf_path)
		grid_param_conf = json.load(fi)
		fi.close()
		dfs_make_confs(conf, 0, grid_param_conf, confs)
	else:
		confs = [conf]
	for conf in confs:
		if params is not None:
			modify_conf_with_params(conf, params)
		time_str = util.log.config_log("./log/criteo_%s_"%(conf['model_name']))
		util.log.log("Train with conf: %s"%conf_path)
		for Model in CRITEO_MODELS:
			if conf['model_name'] == Model.__name__:
				field_sizes = joblib.load(conf['field_sizes_pkl_path'])
				if Model in SPLIT_BY_FIELD_MODELS:
					conf['model_params']['layer_sizes'][0] = field_sizes
				else:
					conf['model_params']['input_dim'] = sum(field_sizes)
				util.log.log("[Config]")
				util.log.pretty_print_json_obj(conf)
				model = Model(**conf['model_params'])

				util.train.train(model,
								 should_split_by_field=Model in SPLIT_BY_FIELD_MODELS,
								 train_log_path="./log/criteo_%s_.%s.log.json"%(conf['model_name'], time_str),
								 ctr_or_recommend=ctr_or_recommend,
								 **conf)
				if conf['should_dump_model']:
					model.dump(conf["model_dump_path"])
				break

def predict_candidate(model, n_user_limit, n_candidate, batch_size=100000):
	rating_mats = None
	raw_user_mats = None
	raw_business_mats = None

	uid_2_bid_set = None

	raw_uids = None
	raw_bids = None

	limit_uids = None
	uid_2_raw_idx = None
	business_num = raw_business_mats[0].shape[0]
	uid_to_candidates = {}
	for uid in (limit_uids):
		user_rows = [mat[uid_2_raw_idx[uid]] for mat in raw_user_mats]
		cur_uids = [uid] * business_num
		cur_user_mats = [vstack([row] * business_num) for row in user_rows]
		cur_rating_mats = []
		for ids in [cur_uids, raw_bids]:
			cur_rating_mats.append(preproc.get_csr_mat_from_exclusive_field(ids))
		dataset = [cur_rating_mats, cur_user_mats, raw_business_mats]
		preds = util.train.predict(model, dataset, batch_size)
		pred_pairs = [(raw_bids[i], preds[i]) for i in xrange(len(preds))]
		sorted(pred_pairs,key=lambda x: x[2], reverse=True)
		exist_bid_set = uid_2_bid_set[uid]
		candidates = []
		for pair in pred_pairs:
			if len(candidates) >= n_candidate:
				break
			if pair not in exist_bid_set:
				candidates.append(pair)
		uid_to_candidates[uid] = candidates
	return uid_to_candidates


if __name__ == "__main__":
	test_dfs_make_confs()