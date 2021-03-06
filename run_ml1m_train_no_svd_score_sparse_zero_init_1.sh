#!/usr/bin/env bash
for conf_id in 52; do
	for lr in 0.0002; do
		for random_seed in `seq 0 4`; do
			for trainset_rate in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1; do
				for width in 10; do
					for n_embd in 100; do
						for reg in 0.001; do
							python criteo.py --ml --train  --gpu 1 \
							--conf_path ./conf/RecIPNN_rate_time_user_movie_no_svd_score_zero_init_${conf_id}.conf \
							--params --width ${width} --lr ${lr} --n_embd ${n_embd} --reg ${reg}\
							--data_suffix .${random_seed}_${trainset_rate} \
							1>log/ml1m_train_rate_time_user_movie_no_svd_score_zero_init_${conf_id}.log 2>&1
							sleep 2s
						done
					done
				done
			done
		done
	done
done
