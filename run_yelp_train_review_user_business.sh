#!/usr/bin/env bash
for conf_id in 1; do
	for lr in 0.001; do
		for width in 10 5; do
			for n_embd in 100 150 50; do
				for reg in 0.001 0.01; do
					python criteo.py --ml --train  --gpu 0 \
					--conf_path ./conf/yelp_RecIPNN_rate_review_user_business_${conf_id}.conf \
					--params --width ${width} --lr ${lr} --n_embd ${n_embd} --reg ${reg}\
					1>./log/yelp_RecIPNN_rate_review_user_business_${conf_id}.log 2>&1
					sleep 2s
				done
			done
		done
	done
done
