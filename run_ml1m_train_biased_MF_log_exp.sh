#!/usr/bin/env bash
for conf_id in 1; do
	for lr in 0.0005 0.0001; do
		for random_seed in 0; do
			for trainset_rate in 0.9; do
				for n_embd in 100 50; do #100 is best now
					for reg in 0.0001 0.00001; do #0.001 is best now
						python criteo.py --ml --train  --gpu 1 \
						--conf_path ./conf/biased_MF_log_exp_rate_${conf_id}.conf \
						--params --lr ${lr} --n_embd ${n_embd} --reg ${reg}\
						--data_suffix .${random_seed}_${trainset_rate} \
						1>log/ml1m_train_rate_real_biased_MF_log_exp_${conf_id}.log 2>&1&
						sleep 2s
					done
				done
			done
		done
	done
done
