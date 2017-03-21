#!/usr/bin/env bash
for conf_id in 11; do
	for lr in 0.0002; do
		for reg in 0.001 0.01; do
			for n_embd in 50; do
				for width in 5 10; do
					python criteo.py --ml --train  --gpu 1 \
					--conf_path ./conf/yelp_RecIPNN_rate_review_user_business_${conf_id}.conf \
					--params --width ${width} --lr ${lr} --n_embd ${n_embd} --reg ${reg}\
					1>./log/yelp_RecIPNN_rate_review_user_business_${conf_id}.log 2>&1
					sleep 2s
				done
			done
		done
	done
done
