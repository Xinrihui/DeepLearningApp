#!/usr/bin/python
# -*- coding: UTF-8 -*-

#  适用于 tensorflow >= 2.0, keras 被直接集成到 tensorflow 的内部

from tensorflow.keras.layers import Embedding, \
    Dropout, Activation, LayerNormalization

from tensorflow.keras.models import Model

from tqdm import tqdm

import time


from lib.layers.attention_layer_xrh import *

from lib.models.position_encode_xrh import *

from lib.utils.mask_xrh import *

from lib.models.optimizer_xrh import *



class PointWiseFeedForward(Layer):
    """
    带有 1个隐藏层的 MLP

    """

    def __init__(self, d_model, dff):
        super(PointWiseFeedForward, self).__init__()
        self.fc1 = Dense(dff, activation='relu')
        self.fc2 = Dense(d_model)

    def get_config(self):
        config = super().get_config().copy()

        config.update({
            'fc1': self.fc1,
            'fc2': self.fc2,
        })
        return config

    def call(self, x):
        out_fc1 = self.fc1(x)
        out = self.fc2(out_fc1)

        return out


class EncoderLayer(Layer):
    """
    单层的编码器层

    """

    def __init__(self, d_model, num_heads, dff, dropout_rates):
        """

        :param d_model: 模型整体的隐藏层的维度
        :param num_heads: 并行注意力层的个数(头数)
        :param dff: Position-wise Feed-Forward Networks 的中间层的维度
        :param dropout_rates: dropout 的弃置率
        """

        super(EncoderLayer, self).__init__()

        self.MultiHead = MultiHeadAttention(d_model, num_heads)  # 命名参考论文
        self.ffn = PointWiseFeedForward(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout_rates[0])
        self.dropout2 = Dropout(dropout_rates[0])

    def get_config(self):
        config = super().get_config().copy()

        config.update({
            'MultiHead': self.MultiHead,
            'ffn': self.ffn,

            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,

            'dropout1': self.dropout1,
            'dropout2': self.dropout2,

        })
        return config

    def call(self, x, training, padding_mask):
        """

        :param x: 输入的 tensor
        :param training: 是否为训练模式
        :param padding_mask: 序列的填充 mask
        :return:
        """

        out_multihead, _ = self.MultiHead(x, x, x, padding_mask)  # shape (N_batch, input_seq_len, d_model)
        out_dropout1 = self.dropout1(out_multihead, training=training)
        out1 = self.layernorm1(x + out_dropout1)  # shape (N_batch, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # shape (N_batch, input_seq_len, d_model)
        out_dropout2 = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + out_dropout2)  # shape (N_batch, input_seq_len, d_model)

        return out2


class DecoderLayer(Layer):
    """
    单层的解码器层

    """

    def __init__(self, d_model, num_heads, dff, dropout_rates):
        """

        :param d_model: 模型整体的隐藏层的维度
        :param num_heads: 并行注意力层的个数(头数)
        :param dff: Position-wise Feed-Forward Networks 的中间层的维度
        :param dropout_rates: dropout 的弃置率
        """

        super(DecoderLayer, self).__init__()

        self.MultiHead1 = MultiHeadAttention(d_model, num_heads)
        self.MultiHead2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = PointWiseFeedForward(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout_rates[0])
        self.dropout2 = Dropout(dropout_rates[0])
        self.dropout3 = Dropout(dropout_rates[0])

    def get_config(self):
        config = super().get_config().copy()

        config.update({
            'MultiHead1': self.MultiHead1,
            'MultiHead2': self.MultiHead2,
            'ffn': self.ffn,

            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'layernorm3': self.layernorm2,

            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
            'dropout3': self.dropout3

        })
        return config

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """

        :param x: 输入的 tensor shape (N_batch, target_seq_len, d_model)
        :param encoder_output: 编码器的输出序列 shape (N_batch, input_seq_len, d_model)
        :param training: 是否为训练模式
        :param look_ahead_mask: 避免看到未来的序列 mask
        :param padding_mask: 序列的填充 mask
        :return:
        """

        out_multihead1, attention_weights_block1 = self.MultiHead1(v=x, k=x, q=x, mask=look_ahead_mask)
        # out_multihead1 shape (N_batch, target_seq_len, d_model)
        out_dropout1 = self.dropout1(out_multihead1, training=training)
        out1 = self.layernorm1(out_dropout1 + x)

        out_multihead2, attention_weights_block2 = self.MultiHead2(
            v=encoder_output, k=encoder_output, q=out1, mask=padding_mask)
        # out_multihead2 shape (N_batch, target_seq_len, d_model)
        out_dropout2 = self.dropout2(out_multihead2, training=training)
        out2 = self.layernorm2(out_dropout2 + out1)
        # out2 shape (N_batch, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # shape (N_batch, target_seq_len, d_model)
        out_dropout3 = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out_dropout3 + out2)  # (N_batch, target_seq_len, d_model)

        return out3, attention_weights_block1, attention_weights_block2
