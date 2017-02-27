#!/usr/bin/env bash
for i in `seq 21 21`; do
	python criteo.py --ml --train  --gpu 1 --conf_path ./conf/RecIPNN_rate_time_user_movie_no_svd_score_${i}_sparse.conf 1>log/ml1m_train_rate_time_user_movie_no_svd_score_${i}_sparse.log 2>&1&
	sleep 2s
done