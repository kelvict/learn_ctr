#!/usr/bin/env bash
for conf_id in 1; do
	for lr in 0.0002; do
		for width in 20; do
			for n_embd in 75 50 100; do
				for reg in 0.005 0.0005; do
					python criteo.py --ml --train  --gpu 1 \
					--conf_path ./conf/yelp_RecIPNN_rate_${conf_id}.conf \
					--params --width ${width} --lr ${lr} --n_embd ${n_embd} --reg ${reg}\
					1>./log/yelp_RecIPNN_rate_${conf_id}.log 2>&1
					sleep 2s
				done
			done
		done
	done
done
