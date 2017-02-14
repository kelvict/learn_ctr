python criteo.py --split_train_test --dataset_path dataset/ctr/criteo/train.txt.20170211_011802.discrete_50_26_contin_100000_2_0_xgb_40.all_split_by_field_csr_mats.pkl \
--labels_path dataset/ctr/criteo/train.txt.20170211_011802.labels.txt \
--trainset_rate 0.9 \
--traindata_dump_path dataset/ctr/criteo/train.txt.20170211_011802.discrete_50_26_contin_100000_2_0_xgb_40.all_split_by_field_csr_mats.pkl.train.pkl \
--testdata_dump_path dataset/ctr/criteo/train.txt.20170211_011802.discrete_50_26_contin_100000_2_0_xgb_40.all_split_by_field_csr_mats.pkl.test.pkl \
> criteo_split_train_test1.log

python criteo.py --split_train_test --dataset_path dataset/ctr/criteo/train.txt.20170211_105206.discrete_50_26_contin_100000_2_13_xgb_40.all_split_by_field_csr_mats.pkl \
--labels_path dataset/ctr/criteo/train.txt.20170211_105206.labels.txt \
--trainset_rate 0.9 \
--traindata_dump_path dataset/ctr/criteo/train.txt.20170211_105206.discrete_50_26_contin_100000_2_13_xgb_40.all_split_by_field_csr_mats.pkl.train.pkl \
--testdata_dump_path dataset/ctr/criteo/train.txt.20170211_105206.discrete_50_26_contin_100000_2_13_xgb_40.all_split_by_field_csr_mats.pkl.test.pkl \
> criteo_split_train_test2.log