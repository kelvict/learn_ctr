#!/usr/bin/env bash
for i in `seq 1 1`; do
	echo ${i}
	python criteo.py --train --gpu 0 --conf_path ./conf/CCPM_${i}.conf 1>criteo_train_CCPM_${i}.log 2>&1
done