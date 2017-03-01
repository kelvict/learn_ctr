#!/usr/bin/env bash
#python criteo.py --ml --preprocess 1>log/ml1m_preprocess.log 2>&1
python criteo.py --ml --preprocess --make_multi_dataset 1>log/ml1m_preprocess.log 2>&1