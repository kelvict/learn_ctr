#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#
import sys

import pandas as pd
import numpy as np
import pandas.core.algorithms as algos
import time
import os
from util import preprocess
import gc
from scipy import sparse
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


default_whole_data_path = 'dac_sample/dac_sample.csv'

def get_labels_contin_discrete_feats(filepath=default_whole_data_path):
    # Load Data and Separate into labels discrete contin
    df = pd.read_csv(filepath, sep='\t', header=None)
    df.columns = ['label'] + ["I_"+str(i) for i in xrange(1,14)] + ["C_"+str(i) for i in xrange(14,40)]
    discrete_df = df.iloc[:,14:]
    contin_df = df.iloc[:,1:14]
    labels = df.iloc[:,0]
    return labels, contin_df, discrete_df

too_few_key_word = "TOO_FEW"

def get_discrete_col_replacing_too_few_feats(discrete_col, value_cnts=None, min_value_cnt=2):
    #将频数太小的category设为too_few
    if value_cnts is None:
        value_cnts = discrete_col.value_counts()
    vc_lt = value_cnts[value_cnts < min_value_cnt]
    return discrete_col.map(lambda x: x if x not in vc_lt else too_few_key_word)

max_val = 9999999
min_val = -9999999

def get_discrete_col_from_contin_col(contin_col):
    #连续值离散
    #前几个uniq_value作为边界
    #然后从大于最后一个已有边界的分位数
    uniq_bins = algos.quantile(contin_col, np.linspace(0,1,1000))
    non_nan_unique_items = np.array([i for i in np.unique(contin_col) if not np.isnan(i)])
    normal_bins = non_nan_unique_items[:min(non_nan_unique_items.shape[0], 30)]
    bins = np.unique(np.sort(np.hstack([[min_val],uniq_bins, normal_bins,[max_val]])))
    return pd.tools.tile._bins_to_cuts(contin_col,bins,include_lowest=True)

contin_feat_means_path = "contin_feat_means.pkl"

def dump_contin_feat_means(contin_feats, path=contin_feat_means_path):
    contin_feats.mean().to_pickle(path)

def load_contin_feat_means(path=contin_feat_means_path):
    return pd.read_pickle(path)

def get_contin_value_with_feat_name_per_row(row, feat_map):
    for i in xrange(row.shape[0]):
        if np.isreal(row[i]) and np.isnan(row[i]):
            pass
        else:
            row[i] = str(feat_map.loc[str(row.index[i]),'cnt_id'])+"\t"+str(row[i])
    return row

def get_discrete_value_with_feat_name_per_row(row, feat_map):
    for i in xrange(row.shape[0]):
        if np.isreal(row[i]) and np.isnan(row[i]):
            pass
        else:
            row[i] = str(feat_map.loc[
                         str(row.index[i])+"_"+str(row[i]).replace(" ",""), 'cnt_id'])\
                     +"\t1"
    return row

def get_contin_value_with_feat_name_per_row_in_libsvm_format(row, feat_map):
    for i in xrange(row.shape[0]):
        if np.isreal(row[i]) and np.isnan(row[i]):
            pass
        else:
            row[i] = str(feat_map.loc[str(row.index[i]), 'cnt_id'])+":"+str(row[i])
    return row

def get_discrete_value_with_feat_name_per_row_in_libsvm_format(row, feat_map):
    for i in xrange(row.shape[0]):
        if np.isreal(row[i]) and np.isnan(row[i]):
            pass
        else:
            row[i] = str(feat_map.loc[
                         str(row.index[i])+"_"+str(row[i]).replace(" ",""), 'cnt_id'])\
                     +":1"
    return row

def dump_df_to_path_omit_nan(df, path, sep="\t", add_time=True, time=time.strftime("%Y%m%d_%H%M%S")):
    if add_time:
        path = path + "." + time
    fo = open(path,'w')
    for i in xrange(df.shape[0]):
        is_first_col = True
        row = df.iloc[i,:]
        for item in set(row):
            if np.isreal(item) and np.isnan(item):
                continue
            if not is_first_col:
                fo.write(sep)
            else:
                is_first_col = False
            fo.write(item)
        fo.write("\n")
    fo.close()
    print "Dump to path %s" % path

def dump_col(col,path):
    col.to_csv(path, header=None, index=None)

def log(msg):
    print str(os.getpid())+'-[' + time.asctime( time.localtime(time.time()) ) + "] " + msg
    sys.stdout.flush()

default_contin_feat_means_path = "default_contin_feat_means.pkl"

default_feat_mean_raw_trainset = "dac_sample/train_0"

def update_default_contin_feat_means_path(feat_mean_raw_trainset=default_feat_mean_raw_trainset,dump_path=default_contin_feat_means_path):
    log("Start to load and split dataset: " + feat_mean_raw_trainset)
    labels, contin_df, discrete_df = get_labels_contin_discrete_feats(feat_mean_raw_trainset)
    log("line "+str(labels.shape[0]))
    log("Sep!")
    log("Dump Contin Feat Means to " + dump_path)
    dump_contin_feat_means(contin_df, dump_path)


def get_value_counts_arr(discrete_df):
    arr = []
    for col in xrange(discrete_df.shape[1]):
        value_counts = discrete_df.iloc[:, col].value_counts()
        value_counts.name = discrete_df.iloc[:, col].name
        arr.append(value_counts)
    return arr

def add_name_to_index_of_value_counts_arr(value_counts):
    value_counts.index = [value_counts.name+"_"+idx.replace(" ","") for idx in value_counts.index]
    return value_counts

def combine_value_counts_arr(value_counts_arr):
    return pd.concat(value_counts_arr)

def fill_cnt_id(df, offset=0):
    cnt_id = pd.Series(range(offset, offset + df.shape[0], 1), index=df.index)
    return pd.concat([df, cnt_id],axis=1)

def get_feat_map(discrete_df, offset=0):
    value_counts_arr = get_value_counts_arr(discrete_df)
    value_counts_arr = [add_name_to_index_of_value_counts_arr(value_counts) for value_counts in value_counts_arr]
    value_counts = combine_value_counts_arr(value_counts_arr)
    feat_map = fill_cnt_id(value_counts, offset=offset)
    feat_map.columns = ['freq', 'cnt_id']
    return feat_map

def get_contin_feat_map(contin_df, offset=0):
    return pd.DataFrame(range(offset,offset+len(contin_df.columns),1),
                        index=contin_df.columns, columns=['cnt_id'])

def test_get_feat_map():
    print os.getcwd()
    raw_trainset = "dac_sample/dac_sample.csv"
    log("Start to load and split dataset: " + raw_trainset)
    labels, contin_df, discrete_df = get_labels_contin_discrete_feats(raw_trainset)

    labels = labels[:10000]
    contin_df = contin_df[:10000]
    discrete_df = discrete_df[:10000]

    print get_feat_map(discrete_df)

def process_contin_df(contin_df, path_prefix="discrete", n_interval=1000, n_split=2, min_freq=10):
    log("Start to process contin df")
    bins_arr = []
    for col in xrange(contin_df.shape[1]):
        log("Handling col %d"%col)
        uniq_bins = np.unique(algos.quantile(contin_df.iloc[:, col], np.linspace(0,1,n_interval)))
        bins_arr.append(uniq_bins)
        contin_df.iloc[:, col] = pd.tools.tile._bins_to_cuts(contin_df.iloc[:, col],uniq_bins,include_lowest=True)
        log("Cut by bins")
    joblib.dump(bins_arr, "%s.bins_arr.pkl"%path_prefix)
    del bins_arr
    log("Start to process cut contin df as discrete_df")
    process_discrete_df(contin_df, path_prefix, n_split, min_freq)
    log("End to process contin df")

def process_discrete_df(discrete_df, path_prefix="discrete", n_split=2, min_freq=4000, too_few_as_nan=False):
    log("Start to process discrete df")
    n_col = discrete_df.shape[1]

    too_few_str = "TOO_FEW"
    nan_str = "-"
    if too_few_as_nan:
        too_few_str = nan_str

    value_counts_arr = []
    log("Start to filter too few")
    too_few_keys_arr = []
    for col in xrange(n_col):
        log("Handling col %d"%col)
        value_counts = discrete_df.iloc[:, col].value_counts()
        value_counts_arr.append(value_counts)
        too_few_keys = list(value_counts[value_counts<min_freq].index)
        enough_keys = list(value_counts[value_counts>=min_freq].index)
        map_pairs = dict(zip(too_few_keys+enough_keys,[too_few_str]*len(too_few_keys)+enough_keys))
        too_few_keys_arr.append(too_few_keys)
        discrete_df.iloc[:, col] = discrete_df.iloc[:, col].map(map_pairs)
    joblib.dump(value_counts_arr, "%s.discrete_value_counts_arr.pkl"%(path_prefix))
    del value_counts_arr
    joblib.dump(too_few_keys_arr, "%s.too_few_keys_arr.pkl"%path_prefix)
    del too_few_keys_arr
    gc.collect()
    log("End to filter too few")
    #To handle NaN Here
    #discrete_df.fillnan("NAN")

    discrete_dfs = np.array_split(discrete_df, n_split, axis=1)

    log("Start to dump filtered discrete data")
    for i in xrange(n_split):
        joblib.dump(discrete_dfs[i], "%s.filtered_discrete%d_in_%d.pkl"%(path_prefix, i, n_split))
    del discrete_dfs
    del discrete_df
    gc.collect()

    log("Start to Label & One hot encode")

    one_hot_encoders = []
    for i in xrange(n_split):
        log("Handling split %d"%i)
        label_encoders = []
        df = joblib.load("%s.filtered_discrete%d_in_%d.pkl"%(path_prefix, i, n_split))
        df = df.fillna(nan_str)

        for col in xrange(df.shape[1]):
            log("Handling split %d, col %d"%(i, col))
            lbl_enc = LabelEncoder()
            df.iloc[:,col] = lbl_enc.fit_transform(df.iloc[:,col])
            label_encoders.append(lbl_enc)
        joblib.dump(label_encoders, "%s.discrete_label_encoders%d.pkl"%(path_prefix, i))
        del label_encoders
        gc.collect()

        log("One hot encoding split %d, col %d"%(i, col))
        enc = OneHotEncoder(sparse=True, dtype=np.float32)
        df  = enc.fit_transform(df)
        log("End to fit&transform")
        joblib.dump(df, "%s.one_hot_discrete%d_in_%d.pkl"%(path_prefix, i, n_split))
        del df
        gc.collect()
        one_hot_encoders.append(enc)

    joblib.dump(one_hot_encoders, "%s.one_hot_encoders.pkl"%path_prefix)
    del one_hot_encoders
    gc.collect()
    log("End to Label & One hot encode")
    log("End to process discrete df")


def preprocess(raw_trainset, is_test=False, n_split=2, discrete_min_freq=4000, n_contin_intervals=1000, contin_min_freq=10):
    log("Start to load and split dataset: " + raw_trainset)
    labels, contin_df, discrete_df = get_labels_contin_discrete_feats(raw_trainset)
    log("line "+str(labels.shape[0]))
    log("Sep!")

    if is_test:
        labels = labels[:100]
        contin_df = contin_df[:100]
        discrete_df = discrete_df[:100]

    cur_time = time.strftime("%Y%m%d_%H%M%S")
    path_prefix = "%s.%s"%(raw_trainset, cur_time)

    dump_col(labels,path_prefix+'.labels.txt')

    del labels
    log("To Cache Contin df")
    joblib.dump(contin_df, "%s.contin_df.pkl"%path_prefix)
    log("Contin df Cached")
    del contin_df
    log("Start to process discrete df")
    discrete_path_prefix = "%s.discrete_%d"%(path_prefix,discrete_min_freq)
    process_discrete_df(discrete_df, path_prefix=discrete_path_prefix, n_split=n_split, min_freq=discrete_min_freq)
    del discrete_df
    log("End process discrete df")
    log("Load contin df cache")
    contin_df = joblib.load("%s.contin_df.pkl"%path_prefix)
    log("Start to process contin df")
    contin_discrete_path_prefix = "%s.contin_%d_%d"%(path_prefix, n_contin_intervals, contin_min_freq)
    process_contin_df(contin_df, path_prefix=contin_discrete_path_prefix, n_interval=n_contin_intervals, n_split=n_split, min_freq=contin_min_freq)
    log("End process contin df")
    log("Start to hstack discrete contin sparse one hot mat with differenct split")
    log("loading discrete")
    discrete_sparse_one_hot_mats = [joblib.load("%s.one_hot_discrete%d_in_%d.pkl"%(
        discrete_path_prefix, i, n_split)) for i in xrange(n_split)]
    log("loading contin")
    contin_discrete_sparse_one_hot_mats = [joblib.load("%s.one_hot_discrete%d_in_%d.pkl"%(
        contin_discrete_path_prefix, i, n_split)) for i in xrange(n_split)]
    log("hstacking")
    whole_mat = sparse.hstack(discrete_sparse_one_hot_mats+contin_discrete_sparse_one_hot_mats, format="csr")
    del discrete_sparse_one_hot_mats
    del contin_discrete_sparse_one_hot_mats
    gc.collect()
    log("Start to dump whole matrix")
    joblib.dump(whole_mat, "%s.discrete_%d_contin_%d_%d.whole_one_hot_csr.pkl"
                %(path_prefix, discrete_min_freq, n_contin_intervals, contin_min_freq))
    del whole_mat
    gc.collect()
    log("End to hstack discrete contin sparse one hot mat with differenct split")

def process(raw_trainset, data_format="default", is_test=False, n_process = 1, contin_feat_means_path=default_contin_feat_means_path):
    log("Start to load and split dataset: " + raw_trainset)
    labels, contin_df, discrete_df = get_labels_contin_discrete_feats(raw_trainset)
    log("line "+str(labels.shape[0]))
    log("Sep!")

    if is_test:
        labels = labels[:100]
        contin_df = contin_df[:100]
        discrete_df = discrete_df[:100]

    item_sep = "\t"
    if data_format == "default":
        get_discrete_row_func = get_discrete_value_with_feat_name_per_row
        get_contin_row_func = get_contin_value_with_feat_name_per_row
    elif data_format == "libsvm":
        get_discrete_row_func = get_discrete_value_with_feat_name_per_row_in_libsvm_format
        get_contin_row_func = get_contin_value_with_feat_name_per_row_in_libsvm_format
        item_sep = " "
    else:
        raise AttributeError("Get wrong data format "+str(data_format))

    dump_col(labels,raw_trainset+'.labels.txt')
    del labels
    log("Label Dumped")

    cur_time = time.strftime("%Y%m%d_%H%M%S")

    #discrete_df = discrete_df.apply(get_discrete_col_replacing_too_few_feats, axis=0)
    discrete_df = preprocess.apply_by_multiprocessing(
        discrete_df, get_discrete_col_replacing_too_few_feats, axis = 0, processes=n_process)
    log("After get_discrete_col_replacing_too_few_feats")
    #discrete_df = discrete_df.apply(get_discrete_row_func, axis=1)
    discrete_feat_map = get_feat_map(discrete_df)
    discrete_df = preprocess.apply_by_multiprocessing(
        discrete_df, get_discrete_row_func, axis = 1, processes=n_process, args=(discrete_feat_map,))
    log("After discrete_df.apply(get_discrete_value_with_feat_name_per_row, axis=1)")
    dump_df_to_path_omit_nan(discrete_df,raw_trainset+'.discrete_feats.%s.txt'%(data_format+"_format"), sep=item_sep, time=cur_time)
    del discrete_df

    #discrete_contin_df = contin_df.fillna(load_contin_feat_means(default_contin_feat_means_path))\
    #                              .apply(get_discrete_col_from_contin_col, axis=0)
    # should fill nan with means
    #contin_feat_means = load_contin_feat_means(contin_feat_means_path)
    #contin_df = contin_df.fillna(contin_feat_means)
    discrete_contin_df = preprocess.apply_by_multiprocessing(
        contin_df, get_discrete_col_from_contin_col, axis=0, processes=n_process)
    #discrete_contin_df = discrete_contin_df.apply(get_discrete_row_func, axis=1)
    discrete_contin_feat_map = get_feat_map(discrete_contin_df, discrete_feat_map.shape[0])

    discrete_contin_df = preprocess.apply_by_multiprocessing(
        discrete_contin_df, get_discrete_row_func, axis=1, processes=n_process, args=(discrete_contin_feat_map,))
    log("After discrete_contin_df.apply(get_discrete_value_with_feat_name_per_row, axis=1)")
    dump_df_to_path_omit_nan(
        discrete_contin_df,raw_trainset+'.discrete_contin_feats.%s.txt'%(data_format+"_format"),sep=item_sep, time=cur_time)
    del discrete_contin_df

    #contin_df = contin_df.apply(get_contin_row_func, axis=1)
    contin_feat_map = get_contin_feat_map(contin_df, discrete_feat_map.shape[0])
    contin_df = preprocess.apply_by_multiprocessing(
        contin_df,  get_contin_row_func, axis=1, processes=n_process, args=(contin_feat_map,))
    log("After contin_df = contin_df.apply(get_contin_value_with_feat_name_per_row, axis=1)")
    dump_df_to_path_omit_nan(contin_df,raw_trainset+'.contin_feats.%s.txt'%(data_format+"_format"),sep=item_sep, time=cur_time)
    del contin_df

def process_with_paths(paths, data_format="default",is_test=False):
    for path in paths:
        process(path,data_format,is_test)

if __name__ == "__main__":
    if False:
        test_get_feat_map()
    if True:
        path = "dac_sample/dac_sample.csv"
        if True:
            contin_feat_means_path = path+".contin_means"
            update_default_contin_feat_means_path(path,contin_feat_means_path)
            process(path,"libsvm", is_test=False, n_process=1, contin_feat_means_path=contin_feat_means_path)
        else:
            process(path,data_format="libsvm",is_test=True)