#!/usr/bin/env bash
for i in `seq 1 3`; do
	echo ${i}
	python criteo.py --train --gpu 0 --conf_path ./conf/FNN_${i}.conf 1>criteo_train_FNN_${i}.log 2>&1
done