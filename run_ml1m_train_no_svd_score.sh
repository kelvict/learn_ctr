#!/usr/bin/env bash
for conf_id in 50; do
	for lr in 0.0001; do
		for random_seed in `seq 0 4`; do
			for trainset_rate in 0.9; do
				for width in 5 8 10; do
					for n_embd in 100; do
						for reg in 0.001; do
							python criteo.py --ml --train  --gpu 1 \
							--conf_path ./conf/RecIPNN_rate_time_user_movie_no_svd_score_${conf_id}.conf \
							--params --width ${width} --lr ${lr} --n_embd ${n_embd} --reg ${reg}\
							--data_suffix ${random_seed}_${trainset_rate} \
							1>log/ml1m_train_rate_time_user_movie_no_svd_score_${conf_id}.log 2>&1&
							sleep 2s
						done
					done
				done
			done
		done
	done
done
