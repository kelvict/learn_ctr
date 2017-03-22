#!/usr/bin/env bash
for conf_id in 1 13 23; do
	for lr in 0.0001; do
		for reg in 0.001; do
			for n_embd in 50; do
				for width in 15 20; do
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
