#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#
import argparse
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-p", "--preprocess", action="store_true", help="should preprocess data")
	parser.add_argument("-i", "--input", type=str, help="set input data path")
	parser.add_argument("-o", "--output", type=str, help="set output data path")
	parser.add_argument("-t", "--test", action="store_true", help="should run in test mode")
	parser.add_argument("-s", "--split_num", type=int, default=2, help="split n part to handle")
	parser.add_argument("--discrete_min_freq", type=int,default=1000)
	parser.add_argument("--continue_min_freq", type=int,default=20)
	parser.add_argument("--continue_n_interval", type=int,default=1000)
	args = parser.parse_args()

	if args.preprocess is True:
		print args.input
		from preprocesser import criteo_preprocesser
		input_path = args.input
		criteo_preprocesser.preprocess(
			input_path, args.test, args.split_num, args.discrete_min_freq, args.continue_n_interval, args.continue_min_freq)
		print "Finish Preprocessing"
	else:
		print "Start testing"
		from preprocesser import criteo_preprocesser
		input_path = "dataset/ctr/criteo/dac_sample/dac_sample.csv"
		criteo_preprocesser.preprocess(
			input_path, False, 2, 1000, 1000, 1000)
		print "Finish Preprocessing"