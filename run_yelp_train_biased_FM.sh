#!/usr/bin/env bash
for conf_id in 1; do
	for lr in 0.005; do
		for random_seed in `seq 0 0`; do
			for trainset_rate in 0.9; do
				for n_embd in 75 100 50; do #100 is best now
					for reg in 0.001; do #0.001 is best now
						python criteo.py --ml --train  --gpu 0 \
						--conf_path ./conf/yelp_biased_MF_rate_${conf_id}.conf \
						--params --lr ${lr} --n_embd ${n_embd} --reg ${reg}\
						1>log/yelp_train_rate_real_biased_MF_${conf_id}.log 2>&1
						sleep 2s
					done
				done
			done
		done
	done
done
