#!/usr/bin/python
# -*- coding: UTF-8 -*-

#  适用于 tensorflow >= 2.0, keras 被直接集成到 tensorflow 的内部

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


class MultiHeadAttention(Layer):
    """
    transformer 中的多头注意力机制

    1.将多个注意力层的输出结果进行连接

    ref:
    1. Attention Is All You Need
    2. https://tensorflow.google.cn/text/tutorials/transformer

    """

    def __init__(self, d_model, num_heads):
        """

        :param d_model: 模型整体的隐藏层的维度
        :param num_heads: 并行注意力层的个数(头数)
        """

        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0  #

        self.depth = d_model // self.num_heads  # 每一个注意力层('头')的隐藏层维度

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        将张量切分为多层, 以适用于多层的注意力层

        将张量的最后一个维度 扩展为 (num_heads, depth)

        :param x: 输入张量 shape (batch_size, seq_len, d_model)
        :param batch_size:
        :return:

        输出张量 shape (batch_size, num_heads, seq_len, depth)

        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        # x shape (batch_size, seq_len, num_heads, depth)

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask):
        """
        缩放后的对齐函数为 dot 的 attention

        :param q: shape (..., seq_len_q, depth) ... 代表可以有多个维度
        :param k: shape (..., seq_len_k, depth)
        :param v: shape (..., seq_len_v, depth)
        :param mask:
             浮点类型的张量, 可以被广播为 (..., seq_len_q, seq_len_k)

        :return:
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # shape (..., seq_len_q, seq_len_k)

        # 对 matmul_qk 进行缩放
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # 对 scaled_attention_logits 加上 mask
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # softmax 归一化作用于张量的最后一个维度(轴) 上
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # shape (..., seq_len_q, seq_len_k)
        # seq_len_q == seq_len_k

        output = tf.matmul(attention_weights, v)
        # output shape (..., seq_len_q, depth)

        return output, attention_weights

    def call(self, v, k, q, mask):
        """

        :param v: shape (batch_size, seq_len, d_model)
        :param k: shape (batch_size, seq_len, d_model)
        :param q: shape (batch_size, seq_len, d_model)
        :param mask: shape (batch_size, 1, 1, seq_len)

        :return: shape (batch_size, seq_len_q, d_model)

        """

        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # shape (batch_size, seq_len, d_model)
        k = self.wk(k)  # shape (batch_size, seq_len, d_model)
        v = self.wv(v)  # shape (batch_size, seq_len, d_model)

        q_heads = self.split_heads(q, batch_size)  # shape (batch_size, num_heads, seq_len_q, depth)
        k_heads = self.split_heads(k, batch_size)  # shape (batch_size, num_heads, seq_len_k, depth)
        v_heads = self.split_heads(v, batch_size)  # shape (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q=q_heads, k=k_heads, v=v_heads, mask=mask)
        # scaled_attention shape  (batch_size, num_heads, seq_len_q, depth)
        # attention_weights shape  (batch_size, num_heads, seq_len_q, seq_len_k)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

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

    def test_MultiHeadAttentionn(self):

        temp_mha = MultiHeadAttention(d_model=512, num_heads=8)

        y = tf.random.uniform((2, 60, 512))  # (batch_size, encoder_sequence, d_model)

        out, attention_weights = temp_mha(v=y, k=y, q=y, mask=None)

        print(f'Attention result shape:  (batch_size, seq_len, d_model): {out.shape}')
        print(f'Attention weights shape: (batch_size, num_heads, seq_len, seq_len): {attention_weights.shape}')


if __name__ == '__main__':

    test = Test()

    # test.test_DotAttention()

    # test.test_GeneralAttention()

    # test.test_ConcatAttention()

    test.test_MultiHeadAttentionn()