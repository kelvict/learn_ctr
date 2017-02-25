#!/usr/bin/env bash
for i in `seq 1 6`; do
	python criteo.py --ml --train  --gpu 1 --conf_path ./conf/RecIPNN_rate_time_user_movie_${i}.conf 1>log/ml1m_train_rate_time_user_movie_${i}.log 2>&1&
	sleep 2s
done