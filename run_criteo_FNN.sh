#!/usr/bin/env bash
python criteo.py --train --gpu 0 --conf_path ./conf/FNN.conf 1>criteo_train_FNN.log 2>&1