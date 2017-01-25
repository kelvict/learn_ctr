#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

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
	parser.add_argument("--conf_path", type=str, help="config path", default="")
	parser.add_argument("--gpu", type=str, help="Set CUDA_VISIBLE_DEVICES", default="")
	args = parser.parse_args()

	if args.preprocess:
		print args.input
		from preprocesser import criteo_preprocesser
		input_path = args.input
		criteo_preprocesser.preprocess(
			input_path, args.test, args.split_num, args.discrete_min_freq, args.continue_n_interval, args.continue_min_freq,
			split_by_field=args.split_by_field)
		print "Finish Preprocessing"
	elif args.train:
		import os
		if len(args.gpu) != 0:
			os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

		from train import criteo_train
		if len(args.create_conf) != 0:
			criteo_train.create_default_conf(args.conf_path, args.create_conf)
		else:
			criteo_train.train_model_with_conf(args.conf_path)
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
		mode = 3
		if mode == 0:
			print "Start testing"
			from preprocesser import criteo_preprocesser
			input_path = "dataset/ctr/criteo/dac_sample/dac_sample.csv"
			criteo_preprocesser.preprocess(
				input_path, False, 2, 1000, 1000, 1000,split_by_col=True)
			print "Finish Preprocessing"
		elif mode == 1:
			from train import criteo_train
			criteo_train.train_model_with_conf("conf/test_FNN.conf")
		elif mode == 2:
			from train import criteo_train
			model_names = ["FM"]
			for name in model_names:
				criteo_train.create_default_conf("conf/%s.conf"%(name), name)
		elif mode == 3:
			from util import preprocess, log
			dataset_path = "dataset/ctr/criteo/dac_sample/dac_sample.csv.20170122_155305.discrete_1000_contin_1000_1000.all_split_by_col_csr_mats.pkl"
			log.config_log("./log/split_trainset_testset_%s.log"%(dataset_path.replace('/', '_')))
			preprocess.split_train_test_data(
				"dataset/ctr/criteo/dac_sample/dac_sample.csv.20170122_155305.discrete_1000_contin_1000_1000.all_split_by_col_csr_mats.pkl",
				"dataset/ctr/criteo/dac_sample/dac_sample.csv.20170122_155305.labels.txt",
				0.75,
				"dataset/ctr/criteo/dac_sample/dac_sample.csv.20170122_155305.discrete_1000_contin_1000_1000.all_split_by_col_csr_mats_train.pkl",
				"dataset/ctr/criteo/dac_sample/dac_sample.csv.20170122_155305.discrete_1000_contin_1000_1000.all_split_by_col_csr_mats_test.pkl")

