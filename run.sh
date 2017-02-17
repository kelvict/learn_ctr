for i in `seq 1 2`; do
	echo "Train ./conf/PNN1_"${i}"_40xgb_no_contin.conf"
	python criteo.py --train --gpu 0 --conf_path ./conf/PNN1_${i}_40xgb_no_contin.conf 1>criteo_train_${i}_xgb_no_contin.log 2>&1
	echo "Train ./conf/PNN1_"${i}"_40xgb_no_contin.conf Finish"
done
for i in `seq 1 2`; do
	echo "Train ./conf/PNN1_"${i}"_40xgb_contin.conf"
	python criteo.py --train --gpu 0 --conf_path ./conf/PNN1_${i}_40xgb_contin.conf 1>criteo_train_${i}_xgb.log 2>&1
	echo "Train ./conf/PNN1_"${i}"_40xgb_contin.conf Finish"
done
