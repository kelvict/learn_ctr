echo "Preprocess 1"
python criteo.py -p -i dataset/ctr/criteo/train.txt -s 4 --discrete_min_freq=50 --continue_min_freq=2 --continue_n_interval=100000 --split_by_field --add_xgb_feat --drop_contin_feat 1>criteo_preprocess1.log 2>&1
echo "Preprocess 2"
python criteo.py -p -i dataset/ctr/criteo/train.txt -s 4 --discrete_min_freq=50 --continue_min_freq=2 --continue_n_interval=100000 --split_by_field --add_xgb_feat 1>criteo_preprocess2.log 2>&1