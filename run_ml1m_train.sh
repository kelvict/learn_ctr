python criteo.py --ml --train  --gpu 1 --conf_path ./conf/RecIPNN_rate_time_user_movie_no_svd_score.conf 1>log/ml1m_train_rate_time_user_movie_no_svd_score.log 2>&1 &
sleep 2s
python criteo.py --ml --train  --gpu 1 --conf_path ./conf/RecIPNN_rate_time_user_movie.conf 1>log/ml1m_train_rate_time_user_movie.log 2>&1 &