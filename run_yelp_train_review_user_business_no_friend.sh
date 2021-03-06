#!/usr/bin/env bash
for conf_id in 12; do
	for lr in 0.00005; do
		for width in 5 10; do
			for n_embd in 50 75; do
				for reg in 0.005; do
					python criteo.py --ml --train  --gpu 1 \
					--conf_path ./conf/yelp_RecIPNN_rate_review_user_business_no_friend_${conf_id}.conf \
					--params --width ${width} --lr ${lr} --n_embd ${n_embd} --reg ${reg}\
					1>./log/yelp_RecIPNN_rate_review_user_business_no_friend_${conf_id}.log 2>&1
					sleep 2s
				done
			done
		done
	done
done
