python criteo.py --ml --train  --gpu 1 --conf_path ./conf/RecIPNN_rate.conf 1>log/ml1m_train_rate.log 2>&1 &
sleep 2s
python criteo.py --ml --train  --gpu 1 --conf_path ./conf/RecIPNN_rate_time.conf 1>log/ml1m_train_rate_time.log 2>&1 &
sleep 2s
python criteo.py --ml --train  --gpu 1 --conf_path ./conf/RecIPNN_rate_no_svd_score.conf 1>log/ml1m_train_rate_no_svd_score.log 2>&1 &
sleep 2s
python criteo.py --ml --train  --gpu 1 --conf_path ./conf/RecIPNN_rate_time_no_svd_score.conf 1>log/ml1m_train_rate_time_no_svd_score.log 2>&1 &