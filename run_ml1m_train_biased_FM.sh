#!/usr/bin/env bash
for i in `seq 1 5`; do
	python criteo.py --ml --train  --gpu 0 --conf_path ./conf/RecIPNN_rate_biased_MF_${i}.conf 1>log/ml1m_train_rate_biased_MF_${i}.log 2>&1&
	sleep 2s
done