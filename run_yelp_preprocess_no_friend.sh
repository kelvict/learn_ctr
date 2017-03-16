#!/usr/bin/env bash
python criteo.py --yelp --preprocess --n_friend 0 --output_suffix no_friend 1>log/yelp_preprocess.log 2>&1