#!/usr/bin/env bash
for conf_id in 1; do
	for lr in 0.0005; do
		for random_seed in `seq 0 4`; do
			for trainset_rate in 0.9; do
				for n_embd in 50; do
					for reg in 0.001; do
						python criteo.py --ml --train  --gpu 1 \
						--conf_path ./conf/biased_MF_rate_${conf_id}.conf \
						--params --width ${width} --lr ${lr} --n_embd ${n_embd} --reg ${reg}\
						--data_suffix ${random_seed}_${trainset_rate} \
						1>log/ml1m_train_rate_real_biased_MF_${conf_id}.log 2>&1&
						sleep 2s
					done
				done
			done
		done
	done
done
