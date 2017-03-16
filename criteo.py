#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	#Handle Different Dataset
	#MovieLens
	parser.add_argument("--ml", action="store_true", help="do movie_lens action")
	parser.add_argument("--make_multi_dataset", action="store_true", help="make multi dataset")
	#Yelp
	parser.add_argument("--yelp", action="store_true")
	parser.add_argument("--n_friend", type=int,default=100)
	#param
	parser.add_argument("--params", action="store_true", help="params input")
	parser.add_argument("--n_embd", type=int, help="embedding size")
	parser.add_argument("--width", type=int)
	parser.add_argument("--act", type=str)
	parser.add_argument("--reg", type=float)
	parser.add_argument("--lr", type=float)
	parser.add_argument("--dropout", type=float)
	parser.add_argument("--data_suffix", type=str)
	#Preprocess Argument
	parser.add_argument("-p", "--preprocess", action="store_true", help="should preprocess data")
	parser.add_argument("-i", "--input", type=str, help="set input data path")
	parser.add_argument("-o", "--output", type=str, help="set output data path")
	parser.add_argument("-t", "--test", action="store_true", help="should run in test mode")
	parser.add_argument("-s", "--split_num", type=int, default=2, help="split n part to handle")
	parser.add_argument("--discrete_min_freq", type=int,default=1000)
	parser.add_argument("--continue_min_freq", type=int,default=20)
	parser.add_argument("--continue_n_interval", type=int,default=1000)
	parser.add_argument("--split_by_field", action="store_true", help="if split by field")
	parser.add_argument("--add_xgb_feat", action="store_true", help="add xgboost feature")
	parser.add_argument("--drop_contin_feat", action="store_true", help="drop contine feature")
	#Split trainset and testset
	parser.add_argument("--split_train_test", action="store_true")
	parser.add_argument("--dataset_path", type=str,)
	parser.add_argument("--labels_path", type=str, help="set labels path")
	parser.add_argument("--trainset_rate", type=float)
	parser.add_argument("--traindata_dump_path", type=str)
	parser.add_argument("--testdata_dump_path", type=str)
	#Split By Col
	parser.add_argument("--split_field", action="store_true", help="should split by col")
	parser.add_argument("--field_sizes_path", type=str, help="set field sizes path")

	#Train Argument
	parser.add_argument("--train", action="store_true", help="should train data with model")
	parser.add_argument("--create_conf", type=str, help="create default config", default="")
	parser.add_argument("--grid_param_conf", type=str, default="")
	parser.add_argument("--conf_path", type=str, help="config path", default="")
	parser.add_argument("--gpu", type=str, help="Set CUDA_VISIBLE_DEVICES", default="")
	args = parser.parse_args()

	params = None
	if args.params:
		params = {}
		if args.n_embd:
			params["n_embd"] = args.n_embd
		if args.width:
			params["width"] = args.width
		if args.act:
			params["act"] = args.act
		if args.lr:
			params["lr"] = args.lr
		if args.reg:
			params["reg"] = args.reg
		if args.dropout:
			params["dropout"] = args.dropout
		if args.data_suffix:
			params["data_suffix"] = args.data_suffix
	if args.yelp:
		if args.preprocess:
			from util import log
			log.config_log("./log/yelp_preprocess_log.log")
			from preprocesser import yelp_preprocess
			if args.make_multi_dataset:
				random_seeds = [0, 1, 2, 3, 4]
				trainset_rates = [0.1 * i for i in range(1,10)]
				for random_seed in random_seeds:
					for trainset_rate in trainset_rates:
						yelp_preprocess.preprocess(random_seed, trainset_rate, args.test, args.n_friend)
			else:
				random_seed = 0
				trainset_rate = 0.9
				yelp_preprocess.preprocess(random_seed, trainset_rate, args.test, args.n_friend)
		elif args.train:
			import os
			if len(args.gpu) != 0:
				os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

			from train import criteo_train
			if len(args.create_conf) != 0:
				criteo_train.create_default_conf(args.conf_path, args.create_conf)
			else:
				conf_paths = args.conf_path.split(";")
				grid_param_conf_path = None
				if len(args.grid_param_conf) != 0:
					grid_param_conf_path = args.grid_param_conf
				for conf_path in conf_paths:
					print "Train with Conf path %s"%conf_path
					criteo_train.train_model_with_conf(
						conf_path, grid_param_conf_path=grid_param_conf_path,
						ctr_or_recommend=False, predict_batch_size=10000, params=params)
	elif args.ml:
		if args.preprocess:
			from preprocesser import movielens_preprocess
			if args.make_multi_dataset:
				random_seeds = [4]
				trainset_rates = [0.1 * i for i in range(3,10)]
				for random_seed in random_seeds:
					for trainset_rate in trainset_rates:
						movielens_preprocess.preprocess_1m(
							random_seed=random_seed, trainset_rate=trainset_rate)
			else:
				movielens_preprocess.preprocess_1m(output_suffix=None)
		elif args.train:
			import os
			if len(args.gpu) != 0:
				os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

			from train import criteo_train
			if len(args.create_conf) != 0:
				criteo_train.create_default_conf(args.conf_path, args.create_conf)
			else:
				conf_paths = args.conf_path.split(";")
				grid_param_conf_path = None
				if len(args.grid_param_conf) != 0:
					grid_param_conf_path = args.grid_param_conf
				for conf_path in conf_paths:
					print "Train with Conf path %s"%conf_path
					criteo_train.train_model_with_conf(
						conf_path, grid_param_conf_path=grid_param_conf_path,
						ctr_or_recommend=False, predict_batch_size=100000, params=params)
	elif args.preprocess:
		print args.input
		from preprocesser import criteo_preprocesser
		input_path = args.input
		criteo_preprocesser.preprocess(
			input_path, args.test, args.split_num, args.discrete_min_freq, args.continue_n_interval, args.continue_min_freq,
			split_by_field=args.split_by_field, add_xgb_feat=args.add_xgb_feat, drop_contin_feat=args.drop_contin_feat)
		print "Finish Preprocessing"
	elif args.train:
		import os
		if len(args.gpu) != 0:
			os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

		from train import criteo_train
		if len(args.create_conf) != 0:
			criteo_train.create_default_conf(args.conf_path, args.create_conf)
		else:
			conf_paths = args.conf_path.split(";")
			for conf_path in conf_paths:
				print "Train with Conf path %s"%conf_path
				criteo_train.train_model_with_conf(conf_path, params=params)

	elif args.split_field:
		from util import preprocess, log
		log.config_log("./log/split_%s_"%(args.input.replace('/', '_')))
		preprocess.split_sparse_data_by_field(
			args.input, args.field_sizes_path, args.output)
	elif args.split_train_test:
		from util import preprocess, log
		log.config_log("./log/split_trainset_testset_%s.log"%(args.dataset_path.replace('/', '_')))
		preprocess.split_train_test_data(
			args.dataset_path, args.labels_path,
			args.trainset_rate, args.traindata_dump_path,
			args.testdata_dump_path)
	else:
		mode = 1
		if mode == 0:
			print "Start testing"
			from preprocesser import criteo_preprocesser
			input_path = "dataset/ctr/criteo/dac_sample/dac_sample.csv"
			criteo_preprocesser.preprocess(
				input_path, False, 2, 1000, 1000, 1000,split_by_field=True, add_xgb_feat=False, drop_contin_feat=False)
			print "Finish Preprocessing"
		elif mode == 1:
			from train import criteo_train
			str = "./conf/test_LR_no_contin.conf"
			conf_paths = str.split(";")
			for conf_path in conf_paths:
				criteo_train.train_model_with_conf(conf_path)
		elif mode == 2:
			from train import criteo_train
			model_names = ["FM"]
			for name in model_names:
				criteo_train.create_default_conf("conf/%s.conf"%(name), name)
		elif mode == 3:
			from util import preprocess, log
			dataset_path = "dataset/ctr/criteo/dac_sample/dac_sample.csv.20170122_155305.discrete_1000_contin_1000_1000.all_split_by_col_csr_mats.pkl"
			disc_contin_path = "dataset/ctr/criteo/dac_sample/dac_sample.csv.20170209_163314.discrete_1000_26_contin_1000_1000_13_xgb_0.all_split_by_field_csr_mats.pkl"
			disc_xgb_path = "dataset/ctr/criteo/dac_sample/dac_sample.csv.20170209_162932.discrete_1000_26_contin_1000_1000_0_xgb_15.all_split_by_field_csr_mats.pkl"
			disc_contin_xgb_path = "dataset/ctr/criteo/dac_sample/dac_sample.csv.20170209_162704.discrete_1000_26_contin_1000_1000_13_xgb_15.all_split_by_field_csr_mats.pkl"
			labels_path = "dataset/ctr/criteo/dac_sample/dac_sample.csv.20170209_161536.labels.txt"
			log.config_log("./log/split_trainset_testset_%s.log"%(dataset_path.replace('/', '_')))
			for path in [disc_contin_path, disc_xgb_path, disc_contin_xgb_path]:
				print "Split path %s"%path
				preprocess.split_train_test_data(
					path,
					labels_path,
					0.75,
					path+".train.pkl",
					path+".test.pkl")

