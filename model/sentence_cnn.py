#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#

from base_model import BaseModel
import tensorflow as tf
from util.train import init_var
class SentenceCNN(BaseModel):
	def __init__(self, seq_len, vocab_size, embd_size, filter_sizes, filter_nums_per_size, full_conn_layer_sizes=[100,100], l2_reg=0.0, dropout_keep_rate=0.5, is_classfication_or_regression=True, n_class=2, **kwargs):
		self.__dict__.update(kwargs)
		local_vars = locals()
		[setattr(self, key, local_vars[key]) for key in local_vars if key not in ["self", "kwargs"]]

		#Init Input Output Place Holder
		self.X_text = tf.placeholder(tf.int32, [None, seq_len], "X_text")
		self.y = tf.placeholder(tf.float32, [None, n_class], "y")

		self.l2_loss = tf.constant(0.0)

		#Init Embd Layer
		with tf.device("/cpu:0"), tf.name_scope("embedding"):
			self.W_e = tf.Variable(tf.random_uniform([vocab_size, embd_size]),name="W_e")
			self.embd_layer = tf.nn.embedding_lookup(self.W_e, self.X_text, name="chars embedding")
			self.expanded_embd_layer = tf.expand_dims(self.embd_layer, -1)

		#Conv Pool Layer
		pool_layers = []
		for i, filter_size in enumerate(filter_sizes):
			with tf.name_scope("conv_pool_%d"%i):
				conv_filter = tf.Variable(init_var("tnormal", (filter_size, embd_size, 1, filter_nums_per_size[i])))
				feat_map = tf.nn.conv2d(self.expanded_embd_layer, conv_filter, [1, 1, 1, 1], padding="VALID", name="conv_layer")
				b = tf.Variable(init_var("tnormal", [filter_nums_per_size[i]]))
				act_feat_map = tf.nn.relu(tf.nn.bias_add(feat_map, b),name="relu")
				pool_layer = tf.nn.max_pool(act_feat_map, [1, seq_len-(filter_size-1), 1, 1])
				pool_layers.append(pool_layer)
		self.pool_layer = tf.concat(pool_layers, 3, name="concat pool layer")
		total_filter_num = sum(filter_nums_per_size)
		self.pool_layer = tf.reshape(self.pool_layer, [-1, total_filter_num])

		with tf.name_scope("conv_pool_dropout"):
			self.pool_layer = tf.nn.dropout(self.pool_layer, dropout_keep_rate)

		#Full Conn Layer
		hidden_layer = self.pool_layer
		last_layer_size = total_filter_num
		for i, layer_size in enumerate(full_conn_layer_sizes):
			with tf.name_scope("full_conn_%d"%i):
				W = tf.Variable(init_var("tnormal", (last_layer_size, layer_size)), dtype=tf.float32, name="W")
				b = tf.Variable(init_var("tnormal", layer_size), dtype=tf.float32, name="b")
				hidden_layer = tf.nn.relu(tf.nn.bias_add(tf.matmul(hidden_layer, W), b))
				last_layer_size = layer_size

		#Output
		with tf.name_scope("output"):
			W = tf.get_variable("W", (last_layer_size, n_class), initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(init_var("tnormal", n_class), dtype=tf.float32, name="b")

			tf.exp(tf.nn.bias_add(tf.matmul(hidden_layer, W), b))
			self.l2_loss += tf.nn.l2_loss(W, "W_l2_loss")
			self.l2_loss += tf.nn.l2_loss(b, "b_l2_loss")
			self.scores = tf.nn.bias_add(tf.matmul(hidden_layer, W), b, name="score")
			if is_classfication_or_regression:
				self.preds = tf.argmax(self.scores, 1, name="prediction")
			else:
				self.preds = self.scores

		#Cal Loss
		with tf.name_scope("loss"):
			if is_classfication_or_regression:
				losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y)
				self.pure_loss = tf.reduce_mean(losses)
			else:
				self.pure_loss = tf.reduce_mean(tf.square(self.scores - self.y))
			self.loss = self.pure_loss + l2_reg * self.l2_loss

		if is_classfication_or_regression:
			with tf.name_scope("accuracy"):
				self.accuracy = tf.reduce_mean(
					tf.cast(tf.equal(self.preds, tf.argmax(self.y, 1)), tf.float32), name="accuracy")
		else:
			with tf.name_scope("rmse"):
				self.rmse = tf.sqrt(self.pure_loss)

	def run(self, fetches, X=None, y=None):
		feed_dict = {}
		if X is not None:
			if isinstance(X, dict):
				if "text" in X:
					feed_dict[self.X_text] = X["text"]
				else:
					raise NameError("Unknown Text Key")
			else:
				feed_dict[self.X_text] = X
		if y is not None:
			feed_dict[self.y] = y
		self.sess.run(fetches, feed_dict)
	def dump(self, model_path):
		pass

if __name__ == "__main__":
	a = 3
	obj = SentenceCNN(1,2,3,4,5,6,qq=1)
	print obj.filter_size