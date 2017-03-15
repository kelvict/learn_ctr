#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#
import multiprocessing
import pandas as pd
import numpy as np
import os
import sys
from util import log, train as train_util
import util
from sklearn.externals import joblib
import gc
import datetime
import time
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MultiLabelBinarizer
from scipy.sparse import vstack
import pandas.core.algorithms as algos

def get_csr_mat_from_exclusive_field(col):
    lbl_enc = LabelEncoder()
    onehot_enc = OneHotEncoder()
    return onehot_enc.fit_transform(np.atleast_2d(lbl_enc.fit_transform(col)).T)

def get_csr_mat_from_multiple_field(col):
    col.apply(lambda x:set() if x is None else x)
    mlb = MultiLabelBinarizer(sparse_output=True)
    return mlb.fit_transform(col)

def get_csr_mat_from_contin_field(col, n_interval=10, uniq_bins=None,estimate_interval_size=None):
    if estimate_interval_size is not None:
        n_interval = (col.max() - col.min()) / float(estimate_interval_size)
    if uniq_bins is None:
        uniq_bins = np.unique(algos.quantile(col, np.linspace(0, 1, n_interval)))
	col = pd.tools.tile._bins_to_cuts(col, uniq_bins, include_lowest=True)
    return get_csr_mat_from_exclusive_field(col)

def join_expand(csr_mat, ids, ids_to_idx):
    return vstack([csr_mat[ids_to_idx[ids[i]]] for i in xrange(len(ids))])

def _apply_df(params):
    df = pd.DataFrame()

    df, func, axis, args, kwargs = params
    log.log("%s apply to data with shape %s"%(os.getpid(), df.shape))
    df = df.apply(func, axis, args=args, **kwargs)
    log.log("%s finish applying to data with shape %s"%(os.getpid(), df.shape))
    return df

def apply_by_multiprocessing(df, func, axis = 0, processes = 1, args=(), **kwargs):
    if processes > 1:
        pool = multiprocessing.Pool(processes=processes)
        manager = multiprocessing.Manager()
        ns = manager.Namespace()
        ns.params = [(df_part, func, axis, args, kwargs)
                           for df_part in np.array_split(df, processes, axis=1 if axis == 0 else 0)]
        result = pool.map(_apply_df,
                          ns.params)
        pool.close()
        return pd.concat(list(result), axis=1 if axis == 0 else 0)
    else:
        return df.apply(func,axis=axis,args=args,**kwargs)

cnt = {}
def test_func(x):
    global cnt
    if os.getpid() in cnt:
        cnt[os.getpid()] += 1
    else:
        cnt[os.getpid()] = 1

    result = (x*1000000+os.getpid()) * 100 + cnt[os.getpid()]
    sys.stdout.flush()
    return result

def test_apply_by_multiprocessing(df):
    print df
    result = apply_by_multiprocessing(df, test_func, axis=0, processes=4)
    print result

def test_apply_by_single_processing(df):
    print df
    result = df.apply(test_func, axis=1)
    print result

def join_files(filenames, output_path, sep = " "):
    try:
        fins = []
        for filename in filenames:
            f = open(filename)
            fins.append(f)
        fout = open(output_path, mode="w+")
        some_fin_ends = False
        while not some_fin_ends:
            newline = ""
            for fin in fins:
                cur_line = fin.readline()
                if len(cur_line) == 0:
                    some_fin_ends = True
                    break
                newline = newline + sep + cur_line[:-1]
            if not some_fin_ends:
                fout.write(newline + '\n')
        for fin in fins:
            fin.close()
        fout.close()
    except IOError,e:
        log.log("IOError")

def get_field_sizes(csr_mats):
    return [mat.shape[1] for mat in csr_mats]

def get_field_sizes_from_one_hot_encoders(one_hot_encoders):
    field_sizes = []
    for i in xrange(len(one_hot_encoders)):
        field_idxs = one_hot_encoders[i].feature_indices_
        for j in xrange(len(field_idxs)-1):
            field_sizes.append(field_idxs[j+1]-field_idxs[j])
    return field_sizes

def get_field_idxs_from_field_size(field_sizes):
    total_size = 0
    field_idxs = []
    for i in xrange(len(field_sizes)):
        field_idxs.append(total_size)
        total_size = total_size + field_sizes[i]
    return field_idxs

def split_sparse_data_by_field(data_pkl_path, field_sizes_path, dump_path):
    import gc
    log.log("Loading data and field sizes")
    data = joblib.load(data_pkl_path)
    field_sizes = joblib.load(field_sizes_path)
    field_offsets = get_field_idxs_from_field_size(field_sizes)
    log.log("Start to csc")
    data = data.tocsc()
    gc.collect()
    log.log("Start to split")
    fields = []
    for i in range(len(field_offsets) - 1):
        log.log("Split col %d"%i)
        start_ind = field_offsets[i]
        end_ind = field_offsets[i + 1]
        field_i = data[0][:, start_ind:end_ind]
        fields.append(field_i)
    fields.append(data[0][:, field_offsets[-1]:])
    log.log("Clear old data")
    del data

    gc.collect()
    log.log("To CSR")
    for i in xrange(len(fields)):
        log.log("Col %d To CSR"%i)
        fields[i] = fields[i].tocsr()
    log.log("Start to dump fields")
    joblib.dump(fields, dump_path)
    log.log("Finish whole split")
    return fields

def extract_datetime_info_from_time_stamp(timestamp):
    dt = datetime.datetime.fromtimestamp(timestamp)
    timedelta = datetime.datetime.now()-dt
    days_delta = timedelta.days
    return pd.Series([dt.year, dt.month, dt.day, dt.hour, dt.weekday(), dt.minute, days_delta],
                     index=["year", "month", "day", "hour", "weekday", "minute", "days_delta"])

def extract_date_info_from_str(string):
    date_format = "%Y-%m-%d"
    dt = datetime.datetime.fromtimestamp(time.mktime(time.strptime(string, date_format)))
    timedelta = datetime.datetime.now()-dt
    days_delta = timedelta.days
    return pd.Series([dt.year, dt.month, dt.day, dt.weekday(), days_delta],
                     index=["year", "month", "day", "weekday", "days_delta"])

def split_train_test_data(dataset_path, labels_path, trainset_rate, train_data_dump_path, test_data_dump_path):
    util.log.log("Loading trainset and labels")
    dataset = joblib.load(dataset_path)
    labels = pd.read_csv(labels_path, header=None)
    train_set_size = int(trainset_rate * labels.shape[0])
    util.log.log("Start to split trainset and testset")

    train_labels = labels[:train_set_size]
    test_labels = labels[train_set_size:]
    if not isinstance(dataset, list):
        util.log.log("Start to handling testset")
        test_set = dataset[train_set_size:]
        util.log.log("Get Testset")
        test_data = (test_set, test_labels)
        joblib.dump(test_data,test_data_dump_path)
        del test_data
        del test_set
        gc.collect()
        util.log.log("Start to handling trainset")
        dataset = dataset[:train_set_size]
        util.log.log("Get Trainset")
        train_data = (dataset, train_labels)
        util.log.log("Start to dump")
        joblib.dump(train_data,train_data_dump_path)
        del train_data
        del dataset
        gc.collect()
    else:
        util.log.log("Start to handling testset")
        test_set = [field[train_set_size:] for field in dataset]
        util.log.log("Get Testset")
        test_data = (test_set, test_labels)
        util.log.log("Start to dump")
        joblib.dump(test_data, test_data_dump_path)
        del test_data
        del test_set
        gc.collect()
        util.log.log("Start to handling trainset")
        for i in xrange(len(dataset)):
            util.log.log("handling %d"%i)
            dataset[i] = dataset[i][:train_set_size]
        util.log.log("Get Trainset")
        train_data = (dataset, train_labels)
        util.log.log("Start to dump")
        joblib.dump(train_data,train_data_dump_path)
        util.log.log("dumped")

    util.log.log("End split trainset and testset labels")

if __name__ == "__main__":
    df = pd.DataFrame([range(10,20,1), range(20,30,1)])
    test_apply_by_multiprocessing(df)

