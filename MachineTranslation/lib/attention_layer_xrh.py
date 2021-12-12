#!/usr/bin/python
# -*- coding: UTF-8 -*-

#  适用于 tensorflow >= 2.0, keras 被直接集成到 tensorflow 的内部
#  ref: https://keras.io/about/

from tensorflow.keras.layers import Layer, Dense

import tensorflow as tf

import numpy as np

class DotAttention(Layer):
    """
    对齐函数为 dot 的 attention

    ref:
    1.Effective Approaches to Attention-based Neural Machine Translation
    2.https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention

    """
    def __init__(self):
        super().__init__()

        self.attention = tf.keras.layers.Attention()  # 对齐函数 dot

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'attention': self.attention,
        })
        return config

    def call(self, query, value, mask):
        """

        query 和  value 的 n_h 必须相同

        :param query: 上一层解码器的输出序列 shape (N_batch, target_length, n_h)
        :param value: 编码器的输出序列 shape (N_batch, source_length, n_h)
        :param mask: 编码器的 mask shape (N_batch, source_length)
        :return:
            context [c1,... ] shape (N_batch, target_length, n_h)

        """

        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask

        context_vector, attention_weights = self.attention(
            inputs=[query, value],
            mask=[query_mask, value_mask],
            return_attention_scores=True,
        )

        return context_vector, attention_weights


class GeneralAttention(Layer):
    """
    对齐函数为 general 的 attention

    ref:
    1.Effective Approaches to Attention-based Neural Machine Translation
    2.https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention

    """
    def __init__(self, n_h, initializer=None):

        super().__init__()

        self.W_a = Dense(n_h, use_bias=False, kernel_initializer=initializer)

        self.attention = tf.keras.layers.Attention()  # 对齐函数 dot

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'attention': self.attention,
        })
        return config

    def call(self, query, value, mask):
        """

        :param query: 上一层解码器的输出序列 shape (N_batch, target_length, m)
        :param value: 编码器的输出序列 shape (N_batch, source_length, n_h)
        :param mask: 编码器的 mask shape (N_batch, source_length)
        :return:
            context [c1,... ] shape (N_batch, target_length, n_h)

        """

        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask

        query_W_a = self.W_a(query)

        context_vector, attention_weights = self.attention(
            inputs=[query_W_a, value],
            mask=[query_mask, value_mask],
            return_attention_scores=True,
        )

        return context_vector, attention_weights



class ConcatAttention(Layer):
    """
    对齐函数为 concat 的 attention

    ref:
    1.Effective Approaches to Attention-based Neural Machine Translation
    2.Neural machine translation by jointly learning to align and translate
    3.https://www.tensorflow.org/api_docs/python/tf/keras/layers/AdditiveAttention

    """

    def __init__(self, n_h, initializer=None):
        super().__init__()

        self.W1 = Dense(n_h, use_bias=False, kernel_initializer=initializer)
        self.W2 = Dense(n_h, use_bias=False, kernel_initializer=initializer)

        self.attention = tf.keras.layers.AdditiveAttention()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'W1': self.W1,
            'W2': self.W2,
            'attention': self.attention,
        })
        return config

    def call(self, query, value, mask):
        """

        :param query: 上一层解码器的输出序列 shape (N_batch, target_length, n_h)
        :param value: 编码器的输出序列 shape (N_batch, source_length, n_h)
        :param mask: 编码器的 mask shape (N_batch, source_length)
        :return:
            context [c1,... ] shape (N_batch, target_length, n_h)

        """

        w1_query = self.W1(query)

        w2_key = self.W2(value)

        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask

        context_vector, attention_weights = self.attention(
            inputs=[w1_query, value, w2_key],
            mask=[query_mask, value_mask],
            return_attention_scores=True,
        )

        return context_vector, attention_weights

class Test:

    def test_DotAttention(self):

        attention_layer = DotAttention()

        out_target = tf.random.normal(shape=[4, 1, 100])

        out_source = tf.random.normal(shape=[4, 6, 100])

        example_tokens = np.random.randint(10, size=(4, 6))

        context_vector, attention_weights = attention_layer(
            query=out_target,
            value=out_source,
            mask=(example_tokens != 0))

        print(f'Attention result shape: (batch_size, query_seq_length, units):{context_vector.shape}')
        print(f'Attention weights shape: (batch_size, query_seq_length, value_seq_length):{attention_weights.shape}')

    def test_GeneralAttention(self):

        n_h = 50
        attention_layer = GeneralAttention(n_h)

        out_decoder = tf.random.normal(shape=[4, 1, 100])

        out_encoder = tf.random.normal(shape=[4, 6, n_h])

        example_tokens = np.random.randint(10, size=(4, 6))

        context_vector, attention_weights = attention_layer(
            query=out_decoder,
            value=out_encoder,
            mask=(example_tokens != 0))

        print(f'Attention result shape: (batch_size, query_seq_length, units):           {context_vector.shape}')
        print(f'Attention weights shape: (batch_size, query_seq_length, value_seq_length): {attention_weights.shape}')

    def test_ConcatAttention(self):

        n_h = 50
        attention_layer = GeneralAttention(n_h)

        out_decoder = tf.random.normal(shape=[4, 1, 100])

        out_encoder = tf.random.normal(shape=[4, 6, n_h])

        example_tokens = np.random.randint(10, size=(4, 6))

        context_vector, attention_weights = attention_layer(
            query=out_decoder,
            value=out_encoder,
            mask=(example_tokens != 0))

        print(f'Attention result shape: (batch_size, query_seq_length, units):           {context_vector.shape}')
        print(f'Attention weights shape: (batch_size, query_seq_length, value_seq_length): {attention_weights.shape}')


if __name__ == '__main__':

    test = Test()

    # test.test_DotAttention()

    # test.test_GeneralAttention()

    test.test_ConcatAttention()
