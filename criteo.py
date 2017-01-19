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

	#Train Argument
	parser.add_argument("--train", action="store_true", help="should train data with model")
	parser.add_argument("--create_conf", type=str, help="create default config", default="")
	parser.add_argument("--conf_path", type=str, help="config path", default="")
	parser.add_argument("--gpu", type=str, help="Set CUDA_VISIBLE_DEVICES")
	args = parser.parse_args()

	if args.preprocess:
		print args.input
		from preprocesser import criteo_preprocesser
		input_path = args.input
		criteo_preprocesser.preprocess(
			input_path, args.test, args.split_num, args.discrete_min_freq, args.continue_n_interval, args.continue_min_freq)
		print "Finish Preprocessing"
	elif args.train:
		import os
		os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

		from train import criteo_train
		if len(args.create_conf) != 0:
			criteo_train.create_default_conf(args.conf_path, args.create_conf)
		else:
			criteo_train.train_model_with_conf(args.conf_path)
	else:
		mode = 1
		if mode == 0:
			print "Start testing"
			from preprocesser import criteo_preprocesser
			input_path = "dataset/ctr/criteo/dac_sample/dac_sample.csv"
			criteo_preprocesser.preprocess(
				input_path, False, 2, 1000, 1000, 1000)
			print "Finish Preprocessing"
		elif mode == 1:
			from train import criteo_train
			criteo_train.train_model_with_conf("conf/test_LR.conf")
