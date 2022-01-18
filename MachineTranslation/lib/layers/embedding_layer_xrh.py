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
    3. Attention Is All You Need

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
            'V': self.V,
            'b': self.b,

        })
        return config

    def build(self, input_shape):

        self.V = self.add_weight(
            name='V',
            shape=(self.n_vocab, self.n_h),
            initializer="random_normal",
            trainable=True,
        )

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            name='b',
            initial_value=b_init(shape=(self.n_vocab,), dtype='float32'),
            trainable=True)

    def call(self, inputs):
        """
        输出 embedding 的结果

        :param inputs: 输入的 tensor, shape (N_batch, seq_length)
        :return:
            out, shape (N_batch, seq_length, n_h)

        """

        out = tf.nn.embedding_lookup(params=self.V, ids=inputs)

        return out

    def call_liner(self, inputs):
        """
        输出 线性层(Dense) 的结果

        :param inputs: 输入的 tensor, shape (N_batch, seq_length, n_h)
        :return:
            out shape (N_batch, seq_length, n_vocab)
        """

        out = tf.matmul(inputs, tf.transpose(self.V, perm=[1, 0])) + self.b
        # inputs shape (N_batch, seq_length, n_h) , V.T shape (n_h, n_vocab)

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
        out_liner = embed_layer.call_liner(batch_h)
        print('out_liner shape: ', out_liner.shape)



if __name__ == '__main__':

    test = Test()

    test.test_SharedEmbedding()

