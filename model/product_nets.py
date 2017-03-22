import cPickle as pkl

import tensorflow as tf

from util import train as train_util
from base_model import BaseModel
dtype = train_util.DTYPE
import math

class LR(BaseModel):
    default_params = {
        'input_dim': "To fill input_dim", #To Fill
        'opt_algo': 'gd',
        'learning_rate': 0.01,
        'l2_weight': 0,
        'random_seed': 0,
        "init_path":None,
    }
    def __init__(self, input_dim=None, output_dim=1, init_path=None, opt_algo='gd', learning_rate=1e-2,
                 l2_weight=0, random_seed=None):
        init_vars = [('w', [input_dim, output_dim], 'tnormal', dtype),
                     ('b', [output_dim], 'zero', dtype)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            self.vars = train_util.init_var_map(init_vars, init_path)

            w = self.vars['w']
            b = self.vars['b']
            logits = tf.sparse_tensor_dense_matmul(self.X, w) + b
            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits, self.y)) + \
                        l2_weight * tf.nn.l2_loss(w)
            self.optimizer = train_util.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        if X is not None:
            feed_dict[self.X] = X
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path

class FM(BaseModel):
    default_params = {
        'input_dim': "To fill input_dim", # To Fill
        'factor_order': 10,
        'opt_algo': 'adam',
        'learning_rate': 0.01,
        'l2_w': 0.01,
        'l2_v': 0.001,
        "init_path":None,
    }
    def __init__(self, input_dim=None, output_dim=1, factor_order=10, init_path=None, opt_algo='gd', learning_rate=1e-2,
                 l2_w=0, l2_v=0, random_seed=None):
        init_vars = [('w', [input_dim, output_dim], 'tnormal', dtype),
                     ('v', [input_dim, factor_order], 'tnormal', dtype),
                     ('b', [output_dim], 'zero', dtype)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            self.vars = train_util.init_var_map(init_vars, init_path)

            w = self.vars['w']
            v = self.vars['v']
            b = self.vars['b']
            X_square = tf.SparseTensor(self.X.indices, tf.square(self.X.values), self.X.shape)
            p = 0.5 * tf.reshape(
                tf.reduce_sum(
                    tf.square(tf.sparse_tensor_dense_matmul(self.X, v)) -
                    tf.sparse_tensor_dense_matmul(X_square, tf.square(v)),
                    1),
                [-1, output_dim])
            logits = tf.sparse_tensor_dense_matmul(self.X, w) + b + p
            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits, self.y)) + \
                        l2_w * tf.nn.l2_loss(w) + \
                        l2_v * tf.nn.l2_loss(v)
            self.optimizer = train_util.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        if X is not None:
            feed_dict[self.X] = X
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path


class FNN(BaseModel):
    default_params = {
        'layer_sizes': ["To fill field_sizes", 10, 1],
        'layer_acts': [None, None, None],
        'layer_keeps': [1, 1, 1],
        'opt_algo': 'gd',
        'learning_rate': 1,
        'layer_l2': [0.001, 0.001, 0.001],
        'random_seed': 0,
        "init_path":None,
    }
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, init_path=None,
                 opt_algo='gd', learning_rate=1e-2, random_seed=None):
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w1', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal',))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.vars = train_util.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            l = tf.nn.dropout(
                train_util.activate(
                    tf.concat(1, [
                        tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) + b0[i]
                        for i in range(num_inputs)]),
                    layer_acts[0]),
                layer_keeps[0])

            for i in range(1, len(layer_sizes) - 1):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(
                    train_util.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    layer_keeps[i])

            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(l, self.y))
            if layer_l2 is not None:
                for i in range(num_inputs):
                    self.loss += layer_l2[0] * tf.nn.l2_loss(w0[i])
                for i in range(1, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    # bi = self.vars['b%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            self.optimizer = train_util.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        if X is not None:
            for i in range(len(X)):
                feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path


class CCPM(BaseModel):
    default_params = {
        'layer_sizes': ["To fill field_sizes", 10, 5, 3], # To fill field sizes
        'layer_acts': ['relu', 'relu', 'relu'],
        'layer_keeps': [1, 1, 1],
        'opt_algo': 'gd',
        'learning_rate': 1e-2,
        'random_seed': 0,
        "init_path":None,
    }
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, init_path=None, opt_algo='gd',
                 learning_rate=1e-2, random_seed=None):
        init_vars = []
        num_inputs = len(layer_sizes[0])
        embedding_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = embedding_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('f1', [embedding_order, layer_sizes[2], 1, 2], 'tnormal', dtype))
        init_vars.append(('f2', [embedding_order, layer_sizes[3], 2, 2], 'tnormal', dtype))
        init_vars.append(('w1', [2 * 3 * embedding_order, 1], 'tnormal', dtype))
        init_vars.append(('b1', [1], 'zero', dtype))

        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.vars = train_util.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            l = tf.nn.dropout(
                train_util.activate(
                    tf.concat(1, [
                        tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) + b0[i]
                        for i in range(num_inputs)]),
                    layer_acts[0]),
                layer_keeps[0])
            l = tf.transpose(tf.reshape(l, [-1, num_inputs, embedding_order, 1]), [0, 2, 1, 3])
            f1 = self.vars['f1']
            l = tf.nn.conv2d(l, f1, [1, 1, 1, 1], 'SAME')
            l = tf.transpose(
                train_util.max_pool_4d(
                    tf.transpose(l, [0, 1, 3, 2]),
                    num_inputs / 2),
                [0, 1, 3, 2])
            f2 = self.vars['f2']
            l = tf.nn.conv2d(l, f2, [1, 1, 1, 1], 'SAME')
            l = tf.transpose(
                train_util.max_pool_4d(
                    tf.transpose(l, [0, 1, 3, 2]), 3),
                [0, 1, 3, 2])
            l = tf.nn.dropout(
                train_util.activate(
                    tf.reshape(l, [-1, embedding_order * 3 * 2]),
                    layer_acts[1]),
                layer_keeps[1])
            w1 = self.vars['w1']
            b1 = self.vars['b1']
            l = tf.nn.dropout(
                train_util.activate(
                    tf.matmul(l, w1) + b1,
                    layer_acts[2]),
                layer_keeps[2])

            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(l, self.y))
            self.optimizer = train_util.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        if X is not None:
            for i in range(len(X)):
                feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path


class PNN1(BaseModel):
    default_params = {
        'layer_sizes': ["To fill field_sizes", 10, 1],
        'layer_acts': [None, None, None],
        'layer_keeps': [1, 1, 1],
        'short_cuts':[None, None, None],
        'opt_algo': 'gd',
        'learning_rate': 1,
        'layer_l2': [0.001, 0.001, 0.001],
        'kernel_l2': 0.001,
        'random_seed': 0,
        "init_path":None,
        "p_mode":0
    }
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, short_cuts=None, layer_l2=None, kernel_l2=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None, p_mode=0):
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w1', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        print "Running in p_mode %d"%p_mode
        if p_mode == 0:
            init_vars.append(('k1', [num_inputs, layer_sizes[2]], 'tnormal', dtype))
        elif p_mode == 1:
            init_vars.append(('k1', [num_inputs * num_inputs, layer_sizes[2]], 'tnormal', dtype))
        else:
            init_vars.append(('k1', [num_inputs, layer_sizes[2]], 'tnormal', dtype))

        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            layers = []

            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.vars = train_util.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            l = tf.nn.dropout(
                train_util.activate(
                    tf.concat(1, [
                        tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) + b0[i] #transform to embedding
                        for i in range(num_inputs)]),
                    layer_acts[0]),
                layer_keeps[0])
            layers.append(l)

            w1 = self.vars['w1']
            k1 = self.vars['k1']
            b1 = self.vars['b1']
            if p_mode == 0:
                p = tf.reduce_sum(
                    tf.reshape(
                        tf.matmul(
                            tf.reshape(
                                tf.transpose(
                                    tf.reshape(l, [-1, num_inputs, factor_order]),
                                    [0, 2, 1]),
                                [-1, num_inputs]),
                            k1),
                        [-1, factor_order, layer_sizes[2]]),
                    1)
            elif p_mode == 1:
                #slower inner product
                feat_emb_mat = tf.reshape(l, [-1, num_inputs, factor_order])
                feat_sim_vec = tf.reshape(tf.batch_matmul(
                    feat_emb_mat,tf.transpose(feat_emb_mat,[0,2,1])),[-1, num_inputs * num_inputs])
                p = tf.matmul(feat_sim_vec,k1)
            else:
                p = tf.reduce_sum(
                    tf.reshape(
                        tf.matmul(
                            tf.reshape(
                                tf.transpose(
                                    tf.reshape(l, [-1, num_inputs, factor_order]),
                                    [0, 2, 1]),
                                [-1, num_inputs]),
                            k1),
                        [-1, factor_order, layer_sizes[2]]),
                    1)
            l = tf.nn.dropout(
                train_util.activate(
                    tf.matmul(l, w1) + b1 + p,
                    layer_acts[1]),
                layer_keeps[1])
            layers.append(l)

            for i in range(2, len(layer_sizes) - 1):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(
                    train_util.activate(
                        tf.matmul(l, wi) + bi if short_cuts[i] is None
                        else tf.matmul(l, wi) + bi + layers[short_cuts[i]],
                        layer_acts[i]),
                    layer_keeps[i])
                layers.append(l)
                print i, layers
            self.y_prob = tf.sigmoid(l)
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(l, self.y))
            if layer_l2 is not None:
                for i in range(num_inputs):
                    self.loss += layer_l2[0] * tf.nn.l2_loss(w0[i])
                for i in range(1, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    # bi = self.vars['b%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            if kernel_l2 is not None:
                self.loss += kernel_l2 * tf.nn.l2_loss(k1)
            self.optimizer = train_util.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        if X is not None:
            for i in range(len(X)):
                feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path

class RecIPNN(BaseModel):
    default_params = {
        'layer_sizes': ["To fill field_sizes", 10, 1],
        'layer_acts': [None, None, None],
        'layer_keeps': [1, 1, 1],
        'short_cuts':[None, None, None],
        'opt_algo': 'gd',
        'learning_rate': 1,
        'layer_l2': [0.001, 0.001, 0.001],
        'kernel_l2': 0.001,
        'random_seed': 0,
        "init_path":None,
        "init_with_normal_one":True,
        "p_mode":1,
        "drop_embedding_layer":False,
        "add_svd_score":False,
        "add_u_auto_rec":False,
        "add_i_auto_rec":False,
        "add_user_product_bias":True,
        "prev_item_vec_cnt":0
    }
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, short_cuts=None, layer_l2=None, kernel_l2=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None,
                 p_mode=1, add_svd_score=False, init_with_normal_one=True, drop_embedding_layer=False,
                 add_u_auto_rec=False, add_i_auto_rec=False, prev_item_vec_cnt=0, add_user_product_bias=True):
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]

        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        #add score bias for fields
        user_item_fields = [0,1]
        for i in user_item_fields:
            field_size = layer_sizes[0][i]
            init_vars.append(('field_score_b_%d'%i, [field_size, 1], 'tnormal', dtype))
        init_vars.append(('w1', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        if p_mode == 0:
            init_vars.append(('k1', [num_inputs, layer_sizes[2]], 'tnormal', dtype))
        elif p_mode == 1:
            init_vars.append(('k1', [num_inputs * num_inputs, layer_sizes[2]], 'tnormal', dtype))
        else:
            init_vars.append(('k1', [num_inputs, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output],
                              'normal_one' if init_with_normal_one else 'tnormal', dtype)) #critical modify tnormal_zero to normal_one
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))

        #AutoRec
        user_cnt = layer_sizes[0][0]
        item_cnt = layer_sizes[0][1]
        if add_u_auto_rec:
            scale = math.sqrt(6.0 / (user_cnt + layer_sizes[1]))
            WU0 = tf.Variable(tf.random_uniform([user_cnt, layer_sizes[1]], -scale, scale))
            bU0 = tf.Variable(tf.random_uniform([layer_sizes[1]], -scale, scale))
            WU1 = tf.Variable(tf.random_uniform([layer_sizes[1], user_cnt], -scale, scale))
            bU = tf.Variable(tf.random_uniform([user_cnt], -scale, scale))
        if add_i_auto_rec:
            scale = math.sqrt(6.0 / (item_cnt + layer_sizes[1]))
            WI0 = tf.Variable(tf.random_uniform([item_cnt, layer_sizes[1]], -scale, scale))
            bI0 = tf.Variable(tf.random_uniform([layer_sizes[1]], -scale, scale))
            WI1 = tf.Variable(tf.random_uniform([layer_sizes[1], item_cnt], -scale, scale))
            bI = tf.Variable(tf.random_uniform([item_cnt], -scale, scale))


        self.graph = tf.Graph()
        with self.graph.as_default():
            layers = []

            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.vars = train_util.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            #transform to embedding

            embd_vecs = [tf.div(tf.sparse_tensor_dense_matmul(self.X[i], w0[i]),
                                    tf.clip_by_value(tf.reshape(tf.sparse_reduce_sum(self.X[i],1),(-1,1)), 1.0, 999999.0))
                             + b0[i]
                            for i in range(num_inputs-prev_item_vec_cnt)]
            #embd_vecs = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) + b0[i] for i in range(num_inputs-prev_item_vec_cnt)]
            if prev_item_vec_cnt != 0:
                embd_vecs.extend([tf.div(tf.sparse_tensor_dense_matmul(self.X[i], w0[i]),
                                         tf.clip_by_value(tf.reshape(tf.sparse_reduce_sum(self.X[i],1),(-1,1)), 1.0, 999999.0)
                                         + b0[i])
                                  for i in range(num_inputs)[-prev_item_vec_cnt:]])
            l = tf.nn.dropout(
                train_util.activate(
                    tf.concat(1, embd_vecs),
                    layer_acts[0]),
                layer_keeps[0])
            layers.append(l)

            w1 = self.vars['w1']
            k1 = self.vars['k1']
            b1 = self.vars['b1']

            if p_mode == 0:
                feat_emb_mat = tf.reshape(l, [-1, num_inputs, factor_order])
                feat1_emb_mat = tf.slice(feat_emb_mat, [0, 0, 0], [-1, 1, factor_order])
                feat2_emb_mat = tf.slice(feat_emb_mat, [0, 1, 0], [-1, 1, factor_order])
                p = tf.reduce_sum(
                    tf.reshape(
                        tf.matmul(
                            tf.reshape(
                                tf.transpose(
                                    feat_emb_mat,
                                    [0, 2, 1]),
                                [-1, num_inputs]),
                            k1),
                        [-1, factor_order, layer_sizes[2]]),
                    1)
            elif p_mode == 1:
                #slower inner product
                feat_emb_mat = tf.reshape(l, [-1, num_inputs, factor_order])
                feat1_emb_mat = tf.slice(feat_emb_mat, [0, 0, 0], [-1, 1, factor_order])
                feat2_emb_mat = tf.slice(feat_emb_mat, [0, 1, 0], [-1, 1, factor_order])
                feat_sim_vec = tf.reshape(tf.batch_matmul(
                    feat_emb_mat,tf.transpose(feat_emb_mat,[0,2,1])),[-1, num_inputs * num_inputs])
                p = tf.matmul(feat_sim_vec,k1)
            else:
                feat_emb_mat = tf.reshape(l, [-1, num_inputs, factor_order])
                feat1_emb_mat = tf.slice(feat_emb_mat, [0, 0, 0], [-1, 1, factor_order])
                feat2_emb_mat = tf.slice(feat_emb_mat, [0, 1, 0], [-1, 1, factor_order])
                p = tf.reduce_sum(
                    tf.reshape(
                        tf.matmul(
                            tf.reshape(
                                tf.transpose(
                                    tf.reshape(l, [-1, num_inputs, factor_order]),
                                    [0, 2, 1]),
                                [-1, num_inputs]),
                            k1),
                        [-1, factor_order, layer_sizes[2]]),
                    1)
            #TODO
            #svd_score = tf.reduce_sum(tf.mul(tf.reshape(feat1_emb_mat,[-1,factor_order]), tf.reshape(feat2_emb_mat,[-1,factor_order])), 1)
            svd_score = tf.batch_matmul(feat1_emb_mat,tf.transpose(feat2_emb_mat,(0, 2, 1)))
            svd_score = tf.reshape(svd_score,[-1,1])
            print "svd_score: ",svd_score
            embed_layer = tf.matmul(l, w1) + b1 + p

            l = tf.nn.dropout(
                train_util.activate(
                    embed_layer,
                    layer_acts[1]),
                layer_keeps[1])
            layers.append(l)

            for i in range(2, len(layer_sizes) - 1):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(
                    train_util.activate(
                        tf.matmul(l, wi) + bi if short_cuts[i] is None
                        else tf.matmul(l, wi) + bi + layers[short_cuts[i]],
                        layer_acts[i]),
                    layer_keeps[i])
                layers.append(l)
                print i, layers
            self.score_bias = tf.zeros([1,1])
            if add_user_product_bias:
                for i in user_item_fields:
                    self.score_bias = tf.add(self.score_bias, tf.sparse_tensor_dense_matmul(self.X[i], self.vars['field_score_b_%d'%i]))

            if add_svd_score:
                if drop_embedding_layer:
                    self.y_prob = tf.add(self.score_bias, svd_score)
                else:
                    self.y_prob = tf.add(tf.add(l, self.score_bias), svd_score)
            else:
                self.y_prob = tf.add(l, self.score_bias)
            print "y_prob: ", self.y_prob
            self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.y,self.y_prob))))
            #TODO to test down side one is bes
            #Problem is in the learning rate!!!
            #self.loss = (tf.nn.l2_loss(tf.sub(self.y,self.y_prob))/500.0)
            if layer_l2 is not None:
                for i in range(num_inputs):
                    self.loss += layer_l2[0] * tf.nn.l2_loss(w0[i])
                if not drop_embedding_layer:
                    for i in range(1, len(layer_sizes) - 1):
                        wi = self.vars['w%d' % i]
                        # bi = self.vars['b%d' % i]
                        self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            if kernel_l2 is not None:
                if not drop_embedding_layer:
                    self.loss += kernel_l2 * tf.nn.l2_loss(k1)
            self.optimizer = train_util.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        if X is not None:
            for i in range(len(X)):
                feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path

class biasedMF(BaseModel):
    default_params = {
        'layer_sizes':[[1,1]],
        'embd_size':50,
        'opt_algo': 'gd',
        'reg_rate':0.025,
        'learning_rate': 0.03,
        'random_seed': 0,
        "log_exp_product": False,
        "init_path": ""
    }
    def __init__(self, layer_sizes, embd_size=50, opt_algo="gd", reg_rate=0.05, learning_rate=0.03, random_seed=0, log_exp_product=False, init_path=None):
        feature_sizes = layer_sizes[0]
        init_vars = []
        init_vars.append(('b0', [feature_sizes[0],1], 'zero', dtype)) #zero no problem?
        init_vars.append(('b1', [feature_sizes[1],1], 'zero', dtype))
        init_vars.append(('bg', [1,1], 'zero', dtype))
        init_vars.append(('w0', [feature_sizes[0], embd_size], 'tnormal', dtype))
        init_vars.append(('w1', [feature_sizes[1], embd_size], 'tnormal', dtype))
        if log_exp_product:
            init_vars.append(('w2', [2 * embd_size, 1], 'tnormal', dtype))

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = [tf.sparse_placeholder(dtype), tf.sparse_placeholder(dtype)]
            self.y = tf.placeholder(dtype)
            self.vars = train_util.init_var_map(init_vars, init_path=init_path)

            w0 = self.vars['w0']
            w1 = self.vars['w1']
            b0 = self.vars['b0']
            b1 = self.vars['b1']
            bg = self.vars['bg']

            embd0 = tf.sparse_tensor_dense_matmul(self.X[0], w0)
            embd1 = tf.sparse_tensor_dense_matmul(self.X[1], w1)
            bias0 = tf.sparse_tensor_dense_matmul(self.X[0], b0)
            bias1 = tf.sparse_tensor_dense_matmul(self.X[1], b1)
            if log_exp_product:
                w2 = self.vars['w2']
                self.y_prob = tf.exp(tf.matmul(tf.concat(1, [embd0, embd1]), w2))
            else:
                self.y_prob = tf.reshape(tf.reduce_sum(tf.mul(embd0, embd1), 1), [-1, 1])
            print self.y_prob
            self.y_prob = tf.add(self.y_prob, bg)
            print self.y_prob
            self.y_prob = tf.add(self.y_prob, bias0)
            print self.y_prob
            self.y_prob = tf.add(self.y_prob, bias1)
            print self.y_prob
            regularizer = tf.add(tf.nn.l2_loss(embd0), tf.nn.l2_loss(embd1), name='svd_regularizer')
            self.loss = tf.sqrt(tf.nn.l2_loss(tf.subtract(self.y_prob, self.y)))
            penalty = tf.constant(reg_rate, dtype=tf.float32, shape=[], name="l2")
            self.loss = tf.add(self.loss, tf.mul(regularizer,penalty))
            self.optimizer = train_util.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        if X is not None:
            for i in range(len(X)):
                feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path

class PNN2(BaseModel):
    default_params = {
        'layer_sizes': ["To fill field sizes", 10, 1],
        'layer_acts': [None, None, None],
        'layer_keeps': [1, 1, 1],
        'opt_algo': 'gd',
        'learning_rate': 1,
        'layer_l2': [0.001, 0.001, 0.001],
        'kernel_l2': 0.001,
        'random_seed': 0
    }
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, kernel_l2=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w1', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('k1', [factor_order * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal',))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)] #one hot feat per field
            self.y = tf.placeholder(dtype)
            self.vars = train_util.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)] #one hot to embedding transformer
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            l = tf.nn.dropout(
                train_util.activate(
                    tf.concat(1, [ #concat to #field * emb_dim 39 * 10
                        tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) + b0[i] # transform to embedding [#field, emb_dim] [39, 10]
                        for i in range(num_inputs)]),
                    layer_acts[0]),
                layer_keeps[0])
            w1 = self.vars['w1']
            k1 = self.vars['k1']
            b1 = self.vars['b1']
            z = tf.reduce_sum(tf.reshape(l, [-1, num_inputs, factor_order]), 1) # sum all embed of embed dim -> (emb_dim,)
            p = tf.reshape(
                tf.batch_matmul(
                    tf.reshape(z, [-1, factor_order, 1]),
                    tf.reshape(z, [-1, 1, factor_order])),
                [-1, factor_order * factor_order])
            l = tf.nn.dropout(
                train_util.activate(
                    tf.matmul(l, w1) + tf.matmul(p, k1) + b1,
                    layer_acts[1]),
                layer_keeps[1])

            for i in range(2, len(layer_sizes) - 1):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(
                    train_util.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    layer_keeps[i])

            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(l, self.y))
            if layer_l2 is not None:
                for i in range(num_inputs):
                    self.loss += layer_l2[0] * tf.nn.l2_loss(w0[i])
                for i in range(1, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    # bi = self.vars['b%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            if kernel_l2 is not None:
                self.loss += kernel_l2 * tf.nn.l2_loss(k1)
            self.optimizer = train_util.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        if X is not None:
            for i in range(len(X)):
                feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path