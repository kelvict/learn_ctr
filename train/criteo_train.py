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

from model.product_nets import LR, FM, FNN, CCPM, PNN1, PNN2, RecIPNN


CRITEO_MODELS = [LR, FM, FNN, CCPM, PNN1, PNN2, RecIPNN]
SPLIT_BY_FIELD_MODELS = [FNN, CCPM, PNN1, PNN2, RecIPNN]
CTR_MODELS = [LR, FM, FNN, CCPM, PNN1, PNN2]
REC_MODELS = [RecIPNN]
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

def train_model_with_conf(conf_path, grid_param_conf_path=None, ctr_or_recommend=True, predict_batch_size=10000):
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
		util.log.config_log("./log/criteo_%s_"%(conf['model_name']))
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
								 train_log_path="./log/criteo_%s_.%s.log.json"%(conf['model_name'], time.strftime("%Y%m%d_%H%M%S")),
								 ctr_or_recommend=ctr_or_recommend,
								 **conf)
				if conf['should_dump_model']:
					model.dump(conf["model_dump_path"])
				break

if __name__ == "__main__":
	test_dfs_make_confs()