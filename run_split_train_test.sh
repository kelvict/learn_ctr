python criteo.py --dataset_path dataset/ctr/criteo/train.txt.20170209_182138.discrete_50_26_contin_100000_2_0_xgb_15.all_split_by_field_csr_mats.pkl \
--labels_path dataset/ctr/criteo/train.txt.20170210_011753.labels.txt \
--trainset_rate 0.9 \
--traindata_dump_path dataset/ctr/criteo/train.txt.20170209_182138.discrete_50_26_contin_100000_2_0_xgb_15.all_split_by_field_csr_mats.pkl.train.pkl \
--testdata_dump_path dataset/ctr/criteo/train.txt.20170209_182138.discrete_50_26_contin_100000_2_0_xgb_15.all_split_by_field_csr_mats.pkl.test.pkl

python criteo.py --dataset_path dataset/ctr/criteo/train.txt.20170210_011753.discrete_50_26_contin_100000_2_13_xgb_15.all_split_by_field_csr_mats.pkl \
--labels_path dataset/ctr/criteo/train.txt.20170210_011753.labels.txt \
--trainset_rate 0.9 \
--traindata_dump_path dataset/ctr/criteo/train.txt.20170210_011753.discrete_50_26_contin_100000_2_13_xgb_15.all_split_by_field_csr_mats.pkl.train.pkl \
--testdata_dump_path dataset/ctr/criteo/train.txt.20170210_011753.discrete_50_26_contin_100000_2_13_xgb_15.all_split_by_field_csr_mats.pkl.test.pkl
