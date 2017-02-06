for i in `seq 1 10`; do
	echo "Train ./conf/PNN1_"${i}".conf"
	python criteo.py --train --gpu 0 --conf_path ./conf/PNN1_${i}.conf 1>criteo_train_${i}.log 2>&1
	echo "Train ./conf/PNN1_"${i}".conf Finish"
done