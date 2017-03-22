#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#

import cPickle as pkl

import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix

from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
from sklearn.externals import joblib
from sklearn.utils import shuffle as sklearn_shuffle

import pandas as pd
import util
from util.optimizer import RadamOptimizer, NadamOptimizer
import json
import gc

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
        "testset_csr_pkl_path":"dataset/ctr/criteo/....pkl",
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

def train(model, trainset_csr_pkl_path, labels_pkl_path=None, testset_csr_pkl_path=None, n_epoch=5,
          batch_size=256, train_set_percent = 0.75,
          should_split_by_field=False, field_sizes_pkl_path=None,
          should_early_stop=True, early_stop_interval=10, batch_eval_interval=50, should_dump_model=False,
          model_dump_path="", shuffle_trainset=True, eval_interval=1, train_log_path="", ctr_or_recommend=True,
          predict_batch_size=10000, min_rec_pred=1, max_rec_pred=5,**kwargs):
    util.log.log("Start to train model")
    util.log.log("Loading trainset and labels")
    if testset_csr_pkl_path is None:
        dataset = joblib.load(trainset_csr_pkl_path)
        labels = pd.read_csv(labels_pkl_path, header=None)
        train_set_size = int(train_set_percent * labels.shape[0])
        util.log.log("Start to split trainset and testset")
        if not isinstance(dataset, list):
            train_set = dataset[:train_set_size]
            test_set = dataset[train_set_size:]
        else:
            train_set = [field[:train_set_size] for field in dataset]
            test_set = [field[train_set_size:] for field in dataset]
        util.log.log("Start to split trainset and testset labels")
        train_labels = labels[:train_set_size]

        test_labels = labels[train_set_size:]
        if not ctr_or_recommend:
            train_labels = np.clip(train_labels, min_rec_pred, max_rec_pred)
            test_labels = np.clip(test_labels, min_rec_pred, max_rec_pred)
        train_data = (train_set, train_labels)
        test_data = (test_set, test_labels)
    else:
        train_data = joblib.load(trainset_csr_pkl_path)
        test_data = joblib.load(testset_csr_pkl_path)
    util.log.log("Handling field size")
    field_sizes = joblib.load(field_sizes_pkl_path) \
        if field_sizes_pkl_path is not None else None
    if field_sizes is not None:
        if should_split_by_field and not isinstance(train_data[0], list):
            field_idxs = util.preprocess.get_field_idxs_from_field_size(field_sizes)
            util.log.log("Spliting Data by field")
            train_data = util.train.split_data_by_field(train_data, field_idxs)
            test_data = util.train.split_data_by_field(test_data,field_idxs)

    history_infos = []
    history_eval_scores = []
    best_eval_score = -1
    train_score = 999
    test_score = 999
    best_batch_eval_score = -1
    for i in xrange(n_epoch):
        util.log.log("Train in epoch %d"%i)
        fetches = [model.optimizer, model.loss]
        losses = []
        if batch_size > 0:
            losses = []
            inst_size = train_data[0].shape[0] \
                if not isinstance(train_data[0], list) else train_data[0][0].shape[0]
            n_iter = inst_size / batch_size
            if n_iter != float(inst_size) / batch_size:
                n_iter = n_iter + 1
            if shuffle_trainset:
                shuffle_idxs = sklearn_shuffle(range(n_iter))
            for j in xrange(n_iter):
                if j%10000 == 0:
                    util.log.log("Train in epoch %d iter %d"%(i, j))
                idx = j
                if shuffle_trainset:
                    idx = shuffle_idxs[j]
                X, y = util.train.slice(train_data, idx * batch_size,
                                        min(batch_size,
                                            inst_size - idx * batch_size))
                _, loss = model.run(fetches, X, y)
                if batch_eval_interval > 0 and j%batch_eval_interval == batch_eval_interval/2:
                    train_preds = predict(model, train_data, predict_batch_size)
                    test_preds = predict(model, test_data, predict_batch_size)
                    if ctr_or_recommend:
                        train_score = roc_auc_score(train_data[1], train_preds)
                        test_score = roc_auc_score(test_data[1], test_preds)
                        if best_batch_eval_score == -1 or test_score < best_batch_eval_score:
                            best_batch_eval_score = test_score
                        train_loss = log_loss(train_data[1], train_preds)
                        test_loss = log_loss(test_data[1], test_preds)
                        util.log.log("[%d-%d]\tavg-loss:%f\ttrain-auc:%f\teval-auc:%f\ttrain-loss:%f\teval-loss:%f\tmin-eval-auc:%f"
                                     %(i, j, np.mean(losses), train_score, test_score, train_loss, test_loss, best_batch_eval_score))
                        print "[%d-%d]\tavg-loss:%f\ttrain-auc:%f\teval-auc:%f\ttrain-loss:%f\teval-loss:%f\tmin-eval-auc:%f"\
                              %(i, j, np.mean(losses), train_score, test_score, train_loss, test_loss, best_batch_eval_score)
                    else:
                        train_preds = np.clip(train_preds, min_rec_pred, max_rec_pred)
                        test_preds = np.clip(test_preds, min_rec_pred, max_rec_pred)
                        train_score = np.sqrt(mean_squared_error(train_data[1], train_preds))
                        test_score = np.sqrt(mean_squared_error(test_data[1], test_preds))
                        if best_batch_eval_score == -1 or test_score < best_batch_eval_score:
                            best_batch_eval_score = test_score
                        util.log.log("[%d-%d]\tavg-loss:%f\ttrain-rmse:%f\teval-rmse:%f\tmin-eval-rmse:%f"
                                     %(i, j, np.mean(losses), train_score, test_score, best_batch_eval_score))
                        print "[%d-%d]\tavg-loss:%f\ttrain-rmse:%f\teval-rmse:%f\tmin-eval-rmse:%f"\
                              %(i, j, np.mean(losses), train_score, test_score, best_batch_eval_score)

                losses.append(loss)
        elif batch_size == -1:
            X, y = util.train.slice(train_data)
            _, loss = model.run(fetches, X, y)
            losses = [loss]
        if ( i + 1 ) % eval_interval == 0:
            util.log.log("Evaluate in epoch %d"%i)
            train_preds = predict(model, train_data, predict_batch_size)
            util.log.log("Predict Test Set")
            test_preds = predict(model, test_data, predict_batch_size)
            util.log.log("Cal Evaluation")
            if ctr_or_recommend:
                train_score = roc_auc_score(train_data[1], train_preds)
                test_score = roc_auc_score(test_data[1], test_preds)
                if best_eval_score == -1 or test_score < best_eval_score:
                    best_eval_score = test_score
                train_loss = log_loss(train_data[1], train_preds)
                test_loss = log_loss(test_data[1], test_preds)
                util.log.log("[%d]\tavg-loss:%f\ttrain-auc:%f\teval-auc:%f\ttrain-loss:%f\teval-loss:%f\tmin-eval-auc:%f"
                             %(i, np.mean(losses), train_score, test_score, train_loss, test_loss, best_eval_score))
                print "[%d]\tavg-loss:%f\ttrain-auc:%f\teval-auc:%f\ttrain-loss:%f\teval-loss:%f\tmin-eval-auc:%f"\
                      %(i, np.mean(losses), train_score, test_score, train_loss, test_loss, best_eval_score)
            else:
                train_preds = np.clip(train_preds, min_rec_pred, max_rec_pred)
                test_preds = np.clip(test_preds, min_rec_pred, max_rec_pred)
                train_score = np.sqrt(mean_squared_error(train_data[1], train_preds))
                test_score = np.sqrt(mean_squared_error(test_data[1], test_preds))
                if best_eval_score == -1 or test_score < best_eval_score:
                    best_eval_score = test_score
                util.log.log("[%d]\tavg-loss:%f\ttrain-rmse:%f\teval-rmse:%f\tmin-eval-rmse:%f"
                             %(i, np.mean(losses), train_score, test_score, best_eval_score))
                print "[%d]\tavg-loss:%f\ttrain-rmse:%f\teval-rmse:%f\tmin-eval-rmse:%f"\
                      %(i, np.mean(losses), train_score, test_score, best_eval_score)
        else:
            if ctr_or_recommend:
                train_score = -1
                test_score = -1
                train_loss = -1
                test_loss = -1
            else:
                pass
        if ctr_or_recommend:
            history_infos.append({
                "losses":losses,
                "avg-loss":np.mean(losses),
                "train-auc":train_score,
                "test-auc":test_score,
                "train-loss":train_loss,
                "test-loss":test_loss
            })
        else:
            history_infos.append({
                "losses":losses,
                "avg-loss":np.mean(losses),
                "train-rmse":train_score,
                "test-rmse":test_score,
            })
        history_eval_scores.append(test_score)
        if ctr_or_recommend:
            best_test_auc_epoch = np.argmax(history_eval_scores)
        else:
            best_test_auc_epoch = np.argmin(history_eval_scores)
        if should_early_stop and i - best_test_auc_epoch >= early_stop_interval:
            print "Early stop\nbest iteration:\n[%d]\teval-auc: %f"%(best_test_auc_epoch, history_eval_scores[best_test_auc_epoch])
            break
    if should_dump_model:
        model.dump(model_dump_path)
    if len(train_log_path) != 0:
        json_log = {
            "conf": kwargs,
            "eval_log": history_infos,
            "best_eval_score":best_eval_score,
            "best_batch_eval_score": best_batch_eval_score
        }

        param_str = ""
        if kwargs['model_name'] == "biasedMF":
            param_str += "."+str(kwargs['model_params']['embd_size'])
            param_str += "."+str(kwargs['model_params']['learning_rate']).replace('.','p')
            param_str += "."+str(kwargs['model_params']['reg_rate']).replace('.','p')
        else:
            param_str += "."+"_".join([str(l) for l in kwargs['model_params']['layer_sizes'][1:]])
            param_str += "."+str(kwargs['model_params']['layer_acts'][2])
            param_str += "."+str(kwargs['model_params']['learning_rate']).replace('.','p')
            param_str += "."+str(kwargs['model_params']['kernel_l2']).replace('.','p')
            param_str += "."+str(kwargs['model_params']['layer_keeps'][1]).replace('.','p')

        param_str += "."+str(trainset_csr_pkl_path.split('/')[2])
        if field_sizes is None:
            param_str += "."+str(1)
        else:
            param_str += "."+str(len(field_sizes))
        if not trainset_csr_pkl_path.endswith(".pkl"):
            param_str += "."+trainset_csr_pkl_path[-5:].replace('.','p')
        train_log_path += param_str
        train_log_path += "."+str(min(best_eval_score, best_batch_eval_score if best_batch_eval_score != -1 else best_eval_score)).replace('.','p')
        train_log_path += "."+str(test_score).replace('.','p')
        fo = open(train_log_path, "w")
        json.dump(json_log, fo, indent=True, default=util.json_util.json_numpy_serialzer)
        fo.close()
        util.log.log("log json in %s"%train_log_path)

def predict(model, eval_data, batch_size=100000):
    preds = []
    inst_size = eval_data[0].shape[0] \
        if not isinstance(eval_data[0], list) else eval_data[0][0].shape[0]
    n_iter = inst_size / batch_size
    if float(n_iter) != float(inst_size) / batch_size:
        n_iter = n_iter + 1
    for j in xrange(n_iter):
        #util.log.log("Predict in iter %d"%(j))
        X, y = util.train.slice(eval_data, j * batch_size,
                                min(batch_size, inst_size - j * batch_size))
        #util.log.log("Sliced in iter %d"%(j))
        pred = model.run(model.y_prob, X)
        preds.append(pred)
    util.log.log("Stack Prediction Result")
    preds = np.vstack(preds)
    return preds

def init_var_map(init_vars, init_path=None):
    if init_path is not None:
        load_var_map = pkl.load(open(init_path, 'rb'))
        print 'load variable map from', init_path, load_var_map.keys()
    var_map = {}
    #print init_vars
    for var_name, var_shape, init_method, dtype in init_vars:
        #print var_name, var_shape, init_method, dtype
        if init_method == 'zero':
            var_map[var_name] = tf.Variable(tf.zeros(var_shape, dtype=dtype), dtype=dtype)
        elif init_method == 'one':
            var_map[var_name] = tf.Variable(tf.ones(var_shape, dtype=dtype), dtype=dtype)
        elif init_method == 'normal':
            var_map[var_name] = tf.Variable(tf.random_normal(var_shape, mean=0.0, stddev=STDDEV, dtype=dtype),
                                            dtype=dtype)
        elif init_method == "normal_one":
            var_map[var_name] = tf.Variable(tf.random_normal(var_shape, mean=1.0, stddev=STDDEV, dtype=dtype),
                                            dtype=dtype)
        elif init_method == "tnormal_user_product_one":
            const_mat = np.zeros(shape=var_shape, dtype=dtype)
            if var_shape[0] > 1:
                const_mat[1][0] = 1
            if var_shape[1] > 1:
                const_mat[0][1] = 1
            const_mat = tf.constant(const_mat,dtype=dtype)
            init_val = tf.truncated_normal(var_shape, mean=0.0, stddev=STDDEV, dtype=dtype)
            init_val = const_mat+init_val
            var_map[var_name] = tf.Variable(init_val, dtype=dtype)
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
    #There are bugs in nadam & radam
    """
    elif opt_algo == 'nadam':
        return NadamOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'radam':
        return RadamOptimizer(learning_rate).minimize(loss)
    """
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
