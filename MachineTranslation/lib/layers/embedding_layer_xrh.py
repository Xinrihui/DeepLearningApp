#!/usr/bin/python
# -*- coding: UTF-8 -*-

#  适用于 tensorflow >= 2.0, keras 被直接集成到 tensorflow 的内部

from tensorflow.keras.layers import Layer

import tensorflow as tf

import numpy as np


class SharedEmbedding(Layer):
    """
    共享权重的 Embedding

    Author: xrh
    Date: 2022-1-5

    ref:
    1. https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer
    2. https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup
    3. https://github.com/tensorflow/models/tree/master/official/nlp/modeling

    """
    def __init__(self, n_h, n_vocab):
        """

        :param n_h:  隐藏层的维度
        :param n_vocab: 词表的大小
        """

        super(SharedEmbedding, self).__init__()
        self.n_h = n_h
        self.n_vocab = n_vocab

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_h': self.n_h,
            'n_vocab': self.n_vocab,

        })
        return config

    def build(self, input_shape):

        self.shared_weights = self.add_weight(
            name='shared_weights',
            shape=(self.n_vocab, self.n_h),
            initializer=tf.random_normal_initializer(
              mean=0., stddev=self.n_h**-0.5),
            trainable=True,
        )  # 必须加上 name, 否则模型无法 checkpoint

    def call(self, inputs, mode="embedding"):
        """

        :param inputs:
        :param mode:
            (1) embedding
            (2) linear
        :return:
        """
        if mode == "embedding":
            return self.call_embedding(inputs)
        elif mode == "linear":
            return self.call_linear(inputs)
        else:
            raise ValueError("the value of mode is {}, which is illegal.".format(mode))


    def call_embedding(self, inputs):
        """

        :param inputs: 输入的 tensor, shape (N_batch, seq_length)
        :return:
            out, shape (N_batch, seq_length, n_h)

        """

        out = tf.nn.embedding_lookup(params=self.shared_weights, ids=inputs)

        return out

    def call_linear(self, inputs):
        """
        输出 线性层(Dense) 的结果

        :param inputs: 输入的 tensor, shape (N_batch, seq_length, n_h)
        :return:
            out shape (N_batch, seq_length, n_vocab)
        """

        N_batch = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]

        x = tf.reshape(inputs, (-1, self.n_h))
        logits = tf.matmul(x, self.shared_weights, transpose_b=True)  # 解决 OOM 问题

        out = tf.reshape(logits, [N_batch, seq_length, self.n_vocab])

        return out




class Test:

    def test_SharedEmbedding(self):

        n_vocab = 50
        n_h = 20

        embed_layer = SharedEmbedding(n_h, n_vocab)

        batch_tokens = np.random.randint(10, size=(4, 6))
        out_embed = embed_layer(inputs=batch_tokens)
        print('out_embed shape: ', out_embed.shape)

        batch_h = tf.random.normal(shape=[4, 6, n_h])
        out_liner = embed_layer(batch_h, mode='linear')
        print('out_liner shape: ', out_liner.shape)



if __name__ == '__main__':

    test = Test()

    test.test_SharedEmbedding()

