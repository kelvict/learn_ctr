#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#

import cPickle as pkl

import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix

from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

import util

import json

DTYPE = tf.float32

STDDEV = 1e-3
MINVAL = -1e-2
MAXVAL = 1e-2

def create_conf(path, model_name="lr", model_params=None, **kwargs):
    if model_params is None:
        model_params = {
            "learning_rate":0.01,
            "opt_algo":"gd"
        }
    conf = {
        "model_name":model_name,
        "model_params": model_params,
        "should_dump_model":True,
        "model_dump_path":"dataset/ctr/criteo/..model.pkl",
        "trainset_csr_pkl_path":"dataset/ctr/criteo/....pkl",
        "labels_pkl_path":"labels.pkl",
        "field_sizes_pkl_path":"field_size.pkl",
        "n_epoch":10,
        "batch_size":256,
        "train_set_percent":0.75,
        "should_early_stop":True,
        "early_stop_interval":10,
        "shuffle_trainset":True,
        "shuffle_seed":0,
    }
    for key in kwargs:
        conf[key] = kwargs[key]
    fo = open(path, 'w')
    json.dump(conf, fo, indent=True)
    fo.close()
    return conf

def read_data(file_name, input_dim):
    X = []
    y = []
    with open(file_name) as fin:
        for line in fin:
            fields = line.strip().split()
            y_i = int(fields[0])
            X_i = map(lambda x: int(x.split(':')[0]), fields[1:])
            y.append(y_i)
            X.append(X_i)
    X = np.array(X) - 1
    y = np.reshape(np.array(y), (-1, 1))
    X = libsvm_2_coo(X, (X.shape[0], input_dim)).tocsr()
    return X, y

def shuffle(data):
    X, y = data
    ind = np.arange(X.shape[0])
    for i in range(7):
        np.random.shuffle(ind)
    return X[ind], y[ind]


def libsvm_2_coo(libsvm_data, shape):
    coo_rows = np.zeros_like(libsvm_data)
    coo_rows = (coo_rows.transpose() + range(libsvm_data.shape[0])).transpose()
    coo_rows = coo_rows.flatten()
    coo_cols = libsvm_data.flatten()
    coo_data = np.ones_like(coo_rows)
    return coo_matrix((coo_data, (coo_rows, coo_cols)), shape=shape)


def csr_2_input(csr_mat):
    #处理csr_mat为[csr_mat]的情况
    if not isinstance(csr_mat, list):
        coo_mat = csr_mat.tocoo()
        indices = np.vstack((coo_mat.row, coo_mat.col)).transpose()
        values = csr_mat.data
        shape = csr_mat.shape
        return indices, values, shape
    else:
        inputs = []
        for csr_i in csr_mat:
            inputs.append(csr_2_input(csr_i))
        return inputs


def slice(csr_data, start=0, size=-1):
    if not isinstance(csr_data[0], list):
        if size == -1 or start + size >= csr_data[0].shape[0]:
            slc_data = csr_data[0][start:]
            slc_labels = csr_data[1][start:]
        else:
            slc_data = csr_data[0][start:start + size]
            slc_labels = csr_data[1][start:start + size]
    else:
        if size == -1 or start + size >= csr_data[0][0].shape[0]:
            slc_data = []
            for d_i in csr_data[0]:
                slc_data.append(d_i[start:])
            slc_labels = csr_data[1][start:]
        else:
            slc_data = []
            for d_i in csr_data[0]:
                slc_data.append(d_i[start:start + size])
            slc_labels = csr_data[1][start:start + size]
    return csr_2_input(slc_data), slc_labels


def split_data_by_field(data, field_offsets):
    fields = []
    for i in range(len(field_offsets) - 1):
        start_ind = field_offsets[i]
        end_ind = field_offsets[i + 1]
        field_i = data[0][:, start_ind:end_ind]
        fields.append(field_i)
    fields.append(data[0][:, field_offsets[-1]:])
    return fields, data[1]

def train(model, trainset_csr_pkl_path, labels_pkl_path, n_epoch=5,
          batch_size=256, train_set_percent = 0.75,
          should_split_by_field=False, field_sizes=None,
          should_early_stop=True, early_stop_interval=10, should_dump_model=False,
          model_dump_path=""):
    util.log.log("Start to train model")
    util.log.log("Loading trainset and labels")
    dataset = joblib.load(trainset_csr_pkl_path)
    labels = joblib.load(labels_pkl_path)
    train_set_size = int(train_set_percent * dataset.shape[0])
    train_set = dataset[:train_set_size]
    train_labels = labels[:train_set_size]
    test_set = dataset[train_set_size:]
    test_labels = labels[train_set_size:]

    train_data = (train_set, train_labels)
    util.log.log("Shuffling train_data")
    train_data = util.train.shuffle(train_data)
    test_data = (test_set, test_labels)

    if field_sizes is not None:
        feat_idxs = util.preprocess.get_field_idxs_from_field_size(field_sizes)
        if should_split_by_field:
            util.log.log("Spliting Data by field")
            train_data = util.train.split_data_by_field(train_data, feat_idxs)
            test_data = util.train.split_data_by_field(test_data,feat_idxs)

    history_infos = []
    history_test_auc = []
    for i in xrange(n_epoch):
        util.log.log("Train in epoch %d"%i)
        fetches = [model.optimizer, model.loss]
        losses = []
        if batch_size > 0:
            losses = []
            n_iter = train_data[0].shape[0] / batch_size
            for j in range(n_iter):
                X, y = util.train.slice(train_data, j * batch_size, batch_size)
                _, loss = model.run(fetches, X, y)
                losses.append(loss)
        elif batch_size == -1:
            X, y = util.train.slice(train_data)
            _, loss = model.run(fetches, X, y)
            losses = [loss]
        train_preds = model.run(model.y_prob, util.train.slice(train_data)[0])
        test_preds = model.run(model.y_prob, util.train.slice(train_data)[0])

        train_score = roc_auc_score(train_data[1], train_preds)
        test_score = roc_auc_score(test_data[1], test_preds)
        util.log.log("[%d]\tloss:%f\ttrain-auc:%f\teval-auc:%f"%(i, np.means(losses), train_score, test_score))
        history_infos.append({
            "losses":losses,
            "avg-loss":np.means(losses),
            "train-auc":train_score,
            "test-auc":test_score
        })
        history_test_auc.append(test_score)
        best_test_auc_epoch = np.argmax(history_test_auc)
        if should_early_stop and i - best_test_auc_epoch >= early_stop_interval and abs(history_test_auc[-1]-history_test_auc[best_test_auc_epoch]) < 1e-5:
            print "Early stop\nbest iteration:\n[%d]\teval-auc: %f"%(best_test_auc_epoch, history_test_auc[best_test_auc_epoch])
            break
    if should_dump_model:
        model.dump(model_dump_path)


def init_var_map(init_vars, init_path=None):
    if init_path is not None:
        load_var_map = pkl.load(open(init_path, 'rb'))
        print 'load variable map from', init_path, load_var_map.keys()
    var_map = {}
    for var_name, var_shape, init_method, dtype in init_vars:
        if init_method == 'zero':
            var_map[var_name] = tf.Variable(tf.zeros(var_shape, dtype=dtype), dtype=dtype)
        elif init_method == 'one':
            var_map[var_name] = tf.Variable(tf.ones(var_shape, dtype=dtype), dtype=dtype)
        elif init_method == 'normal':
            var_map[var_name] = tf.Variable(tf.random_normal(var_shape, mean=0.0, stddev=STDDEV, dtype=dtype),
                                            dtype=dtype)
        elif init_method == 'tnormal':
            var_map[var_name] = tf.Variable(tf.truncated_normal(var_shape, mean=0.0, stddev=STDDEV, dtype=dtype),
                                            dtype=dtype)
        elif init_method == 'uniform':
            var_map[var_name] = tf.Variable(tf.random_uniform(var_shape, minval=MINVAL, maxval=MAXVAL, dtype=dtype),
                                            dtype=dtype)
        elif isinstance(init_method, int) or isinstance(init_method, float):
            var_map[var_name] = tf.Variable(tf.ones(var_shape, dtype=dtype) * init_method)
        elif init_method in load_var_map:
            if load_var_map[init_method].shape == tuple(var_shape):
                var_map[var_name] = tf.Variable(load_var_map[init_method])
            else:
                print 'BadParam: init method', init_method, 'shape', var_shape, load_var_map[init_method].shape
        else:
            print 'BadParam: init method', init_method
    return var_map


def activate(weights, activation_function):
    if activation_function == 'sigmoid':
        return tf.nn.sigmoid(weights)
    elif activation_function == 'softmax':
        return tf.nn.softmax(weights)
    elif activation_function == 'relu':
        return tf.nn.relu(weights)
    elif activation_function == 'tanh':
        return tf.nn.tanh(weights)
    elif activation_function == 'elu':
        return tf.nn.elu(weights)
    elif activation_function == 'none':
        return weights
    else:
        return weights


def get_optimizer(opt_algo, learning_rate, loss):
    if opt_algo == 'adaldeta':
        return tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adam':
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'ftrl':
        return tf.train.FtrlOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'padagrad':
        return tf.train.ProximalAdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'pgd':
        return tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    else:
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


def gather_2d(params, indices):
    shape = tf.shape(params)
    flat = tf.reshape(params, [-1])
    flat_idx = indices[:, 0] * shape[1] + indices[:, 1]
    flat_idx = tf.reshape(flat_idx, [-1])
    return tf.gather(flat, flat_idx)


def gather_3d(params, indices):
    shape = tf.shape(params)
    flat = tf.reshape(params, [-1])
    flat_idx = indices[:, 0] * shape[1] * shape[2] + indices[:, 1] * shape[2] + indices[:, 2]
    flat_idx = tf.reshape(flat_idx, [-1])
    return tf.gather(flat, flat_idx)


def gather_4d(params, indices):
    shape = tf.shape(params)
    flat = tf.reshape(params, [-1])
    flat_idx = indices[:, 0] * shape[1] * shape[2] * shape[3] + \
               indices[:, 1] * shape[2] * shape[3] + indices[:, 2] * shape[3] + indices[:, 3]
    flat_idx = tf.reshape(flat_idx, [-1])
    return tf.gather(flat, flat_idx)


def max_pool_2d(params, k):
    _, indices = tf.nn.top_k(params, k, sorted=False)
    shape = tf.shape(indices)
    r1 = tf.reshape(tf.range(shape[0]), [-1, 1])
    r1 = tf.tile(r1, [1, k])
    r1 = tf.reshape(r1, [-1, 1])
    indices = tf.concat(1, [r1, tf.reshape(indices, [-1, 1])])
    return tf.reshape(gather_2d(params, indices), [-1, k])


def max_pool_3d(params, k):
    _, indices = tf.nn.top_k(params, k, sorted=False)
    shape = tf.shape(indices)
    r1 = tf.reshape(tf.range(shape[0]), [-1, 1])
    r2 = tf.reshape(tf.range(shape[1]), [-1, 1])
    r1 = tf.tile(r1, [1, k * shape[1]])
    r2 = tf.tile(r2, [1, k])
    r1 = tf.reshape(r1, [-1, 1])
    r2 = tf.tile(tf.reshape(r2, [-1, 1]), [shape[0], 1])
    indices = tf.concat(1, [r1, r2, tf.reshape(indices, [-1, 1])])
    return tf.reshape(gather_3d(params, indices), [-1, shape[1], k])


def max_pool_4d(params, k):
    _, indices = tf.nn.top_k(params, k, sorted=False)
    shape = tf.shape(indices)
    r1 = tf.reshape(tf.range(shape[0]), [-1, 1])
    r2 = tf.reshape(tf.range(shape[1]), [-1, 1])
    r3 = tf.reshape(tf.range(shape[2]), [-1, 1])
    r1 = tf.tile(r1, [1, shape[1] * shape[2] * k])
    r2 = tf.tile(r2, [1, shape[2] * k])
    r3 = tf.tile(r3, [1, k])
    r1 = tf.reshape(r1, [-1, 1])
    r2 = tf.tile(tf.reshape(r2, [-1, 1]), [shape[0], 1])
    r3 = tf.tile(tf.reshape(r3, [-1, 1]), [shape[0] * shape[1], 1])
    indices = tf.concat(1, [r1, r2, r3, tf.reshape(indices, [-1, 1])])
    return tf.reshape(gather_4d(params, indices), [-1, shape[1], shape[2], k])
