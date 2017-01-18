#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score

import numpy as np
import json
import util

from model.product_nets import LR, FM, FNN, CCPM, PNN1, PNN2


CRITEO_MODELS = [LR, FM, FNN, CCPM, PNN1, PNN2]
SPLIT_BY_FIELD_MODELS = [FNN, CCPM, PNN1, PNN2]
def create_default_conf(dump_path, model_name=LR.__name__):
	for Model in CRITEO_MODELS:
		if model_name == Model.__name__:
			return util.train.create_conf(
				dump_path, model_name, model_params=Model.default_params,
				trainset_csr_pkl_path="dataset/ctr/criteo/trainset_csr.pkl",
				labels_pkl_path="dataset/ctr/criteo/labels.pkl",
				field_sizes_pkl_path="dataset/ctr/criteo/field_sizes.pkl"
			)
	return None

def train_model_with_conf(conf_path):
	fi = open(conf_path)
	conf = json.load(fi)
	fi.close()
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
							 **conf)
			if conf['should_dump_model']:
				model.dump(conf["model_dump_path"])

			break


if __name__ == "__main__":
	pass