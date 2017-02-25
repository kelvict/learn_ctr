#!/usr/bin/env bash
python criteo.py --ml --train  --gpu 1 --conf_path ./conf/RecIPNN_rate_biased_MF.conf 1>log/ml1m_train_rate_biased_MF_sparse.log 2>&1&
sleep 2s
python criteo.py --ml --train  --gpu 1 --conf_path ./conf/RecIPNN_rate_biased_MF_sparse.conf 1>log/ml1m_train_rate_biased_MF_sparse.log 2>&1&