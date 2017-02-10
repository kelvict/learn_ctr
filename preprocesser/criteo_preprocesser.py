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
from util import preprocess as preprocess_util
import gc
from scipy import sparse
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import xgboost as xgb

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

        log("One hot encoding split %d"%(i))
        enc = OneHotEncoder(sparse=True, dtype=np.float32)
        df  = enc.fit_transform(df)
        log("End to fit&transform")
        joblib.dump(df, "%s.one_hot_discrete%d_in_%d.pkl"%(path_prefix, i, n_split))
        del df
        gc.collect()
        one_hot_encoders.append(enc)

    field_sizes = preprocess_util.get_field_sizes_from_one_hot_encoders(one_hot_encoders)
    joblib.dump(field_sizes, "%s.field_sizes.pkl"%path_prefix)
    del field_sizes

    joblib.dump(one_hot_encoders, "%s.one_hot_encoders.pkl"%path_prefix)
    del one_hot_encoders
    gc.collect()
    log("End to Label & One hot encode")
    log("End to process discrete df")


def preprocess(raw_trainset, is_test=False, n_split=2, discrete_min_freq=4000, n_contin_intervals=1000,
               contin_min_freq=10, split_by_field=False, add_xgb_feat=False, drop_contin_feat=False,
               xgb_model_load_path=""):
    log("Start to load and split dataset: " + raw_trainset)
    labels, contin_df, discrete_df = get_labels_contin_discrete_feats(raw_trainset)
    log("line "+str(labels.shape[0]))
    log("Sep!")

    cur_time = time.strftime("%Y%m%d_%H%M%S")
    path_prefix = "%s.%s"%(raw_trainset, cur_time)

    n_discrete_split = n_split if not split_by_field else discrete_df.shape[1]
    n_contin_split = n_split if not split_by_field else contin_df.shape[1]

    dump_col(labels,path_prefix+'.labels.txt')

    log("To Cache Contin df")
    joblib.dump(contin_df, "%s.contin_df.pkl"%path_prefix)
    log("Contin df Cached")
    del contin_df

    log("Start to process discrete df")
    discrete_path_prefix = "%s.discrete_%d"%(path_prefix,discrete_min_freq)
    process_discrete_df(discrete_df, path_prefix=discrete_path_prefix, n_split=n_discrete_split, min_freq=discrete_min_freq)
    del discrete_df
    gc.collect()
    log("End process discrete df")

    if not drop_contin_feat:
        log("Load contin df cache")
        contin_df = joblib.load("%s.contin_df.pkl"%path_prefix)
        log("Start to process contin df")
        contin_discrete_path_prefix = "%s.contin_%d_%d"%(path_prefix, n_contin_intervals, contin_min_freq)
        process_contin_df(contin_df, path_prefix=contin_discrete_path_prefix, n_interval=n_contin_intervals, n_split=n_contin_split, min_freq=contin_min_freq)
        del contin_df
        gc.collect()
        log("End process contin df")

    if add_xgb_feat:
        log("Load contin df cache")
        contin_df = joblib.load("%s.contin_df.pkl"%path_prefix)
        log("Start to process xgb feat")
        make_gbdt_features(contin_df, labels,
                           split_by_field=split_by_field, path_prefix=path_prefix)
        del contin_df
        del labels
        gc.collect()
        log("End process xgb feat")

    disc_field_sizes = joblib.load("%s.field_sizes.pkl"%(discrete_path_prefix))
    if not drop_contin_feat:
        contin_field_sizes = joblib.load("%s.field_sizes.pkl"%(contin_discrete_path_prefix))
    else:
        contin_field_sizes = []
    if add_xgb_feat:
        xgb_field_sizes = joblib.load("%s.xgb.field_sizes.pkl"%path_prefix)
    else:
        xgb_field_sizes = []

    all_field_size = disc_field_sizes + contin_field_sizes + xgb_field_sizes
    joblib.dump(all_field_size, "%s.all_field_sizes.pkl"%path_prefix)
    log("Start to hstack discrete contin sparse one hot mat with differenct split")
    log("loading discrete")
    discrete_sparse_one_hot_mats = [joblib.load("%s.one_hot_discrete%d_in_%d.pkl"%(
        discrete_path_prefix, i, n_discrete_split)) for i in xrange(n_discrete_split)]


    if not drop_contin_feat:
        log("loading contin")
        contin_discrete_sparse_one_hot_mats = [joblib.load("%s.one_hot_discrete%d_in_%d.pkl"%(
        contin_discrete_path_prefix, i, n_contin_split)) for i in xrange(n_contin_split)]
    else:
        contin_discrete_sparse_one_hot_mats = []
    if add_xgb_feat:
        xgb_one_hot_mats = joblib.load("%s.xgb.one_hot_mats.pkl"%path_prefix)
    else:
        xgb_one_hot_mats = []
    log("Hstacking discrete mat %d contin discrete mat %d xgb mat %d"
        %(len(discrete_sparse_one_hot_mats), len(contin_discrete_sparse_one_hot_mats), len(xgb_one_hot_mats)))
    disc_mats_len = len(discrete_sparse_one_hot_mats)
    contin_mats_len = len(contin_discrete_sparse_one_hot_mats)
    xgb_mats_len = len(xgb_one_hot_mats)

    all_split_by_field_mats = discrete_sparse_one_hot_mats\
                              +contin_discrete_sparse_one_hot_mats\
                              +xgb_one_hot_mats
    del discrete_sparse_one_hot_mats
    del contin_discrete_sparse_one_hot_mats
    del xgb_one_hot_mats
    gc.collect()

    whole_prefix = "%s.discrete_%d_%d_contin_%d_%d_%d_xgb_%d"\
                   %(path_prefix, discrete_min_freq, disc_mats_len,
                  n_contin_intervals, contin_min_freq, contin_mats_len,
                  xgb_mats_len)

    log("all_split_by_field_mats_len: %s"%(str(len(all_split_by_field_mats))))
    joblib.dump(all_split_by_field_mats, "%s.all_split_by_field_coo_mats.pkl"
                %(whole_prefix))
    log("COO Mats to CSR Mats")
    all_split_by_field_mats = [mat.tocsr() for mat in all_split_by_field_mats]
    joblib.dump(all_split_by_field_mats, "%s.all_split_by_field_csr_mats.pkl"
                %(whole_prefix))
    del all_split_by_field_mats
    gc.collect()
    log("Loading all COO Mats")
    all_split_by_field_mats = joblib.load("%s.all_split_by_field_coo_mats.pkl"
                %(whole_prefix))
    whole_mat = sparse.hstack(all_split_by_field_mats)
    del all_split_by_field_mats
    gc.collect()
    log("Dumping COO: %s"%str(whole_mat.shape))
    joblib.dump(whole_mat, "%s.whole_one_hot_coo.pkl"
                %(whole_prefix))
    log("To CSR")
    whole_mat = whole_mat.tocsr()
    log("Start to dump whole matrix")
    joblib.dump(whole_mat, "%s.whole_one_hot_csr.pkl"
                %(whole_prefix))
    del whole_mat
    gc.collect()
    log("End to hstack discrete contin sparse one hot mat with differenct split")

default_xgb_params = {
    'max_depth': 6,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.8,
    'objective':'binary:logistic',
    'gamma': 0.1,
}

def make_gbdt_features(df, labels, xgb_params=default_xgb_params, split_by_field=False, path_prefix="", model_load_path=""):
    n_round = 40
    train_data = xgb.DMatrix(df.values, labels,missing=np.nan)
    if len(model_load_path)==0:
        bst = xgb.train(xgb_params, train_data, n_round)
        joblib.dump(bst,"%s.xgb_model.pkl"%path_prefix)
    else:
        bst = joblib.load(model_load_path)
    leaf_idxs = bst.predict(train_data, pred_leaf=True)
    onehot_encs = []
    field_sizes = []
    mats = []
    if split_by_field:
        leaf_idxs_every_col = np.hsplit( leaf_idxs ,leaf_idxs.shape[1])
        for i in xrange(len(leaf_idxs_every_col)):
            log("encoding col %d"%i)
            col = leaf_idxs_every_col[i]
            lbl_enc = LabelEncoder()
            onehot_enc = OneHotEncoder()

            one_hot_leaf_idxs = onehot_enc.fit_transform(np.atleast_2d(lbl_enc.fit_transform(col)).T)
            onehot_encs.append(onehot_enc)
            mats.append(one_hot_leaf_idxs)
            field_sizes.append(one_hot_leaf_idxs.shape[1])
    else:
        lbl_enc = LabelEncoder()
        onehot_enc = OneHotEncoder()
        one_hot_leaf_idxs = onehot_enc.fit_transform(lbl_enc.fit_transform(leaf_idxs))
        onehot_encs.append(onehot_enc)
        mats.append(one_hot_leaf_idxs)
        field_sizes = preprocess_util.get_field_sizes_from_one_hot_encoders([onehot_enc])
    joblib.dump(onehot_encs,"%s.xgb.one_hot_encoders.pkl"%(path_prefix))
    joblib.dump(field_sizes,"%s.xgb.field_sizes.pkl"%(path_prefix))
    joblib.dump(mats,"%s.xgb.one_hot_mats.pkl"%(path_prefix))

def test_gbdt_features():
    path = "/Users/kelvin_zhang/Project/graduate_design/dataset/ctr/criteo/dac_sample/dac_sample.csv"
    labels, contin_df, discrete_df = get_labels_contin_discrete_feats(
        filepath="/Users/kelvin_zhang/Project/graduate_design/dataset/ctr/criteo/dac_sample/dac_sample.csv")
    make_gbdt_features(contin_df, labels,split_by_field=True,path_prefix=path)

if __name__ == "__main__":
    test_gbdt_features()