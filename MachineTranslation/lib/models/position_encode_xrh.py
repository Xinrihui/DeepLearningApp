#!/usr/bin/python
# -*- coding: UTF-8 -*-

#  适用于 tensorflow >= 2.0, keras 被直接集成到 tensorflow 的内部

import tensorflow as tf

import numpy as np


class SinusoidalPE:
    """
    手工设计的使用正弦曲线的位置编码 (Positional Encoding)

    ref:
    1. Attention Is All You Need

    """

    def __init__(self, d_model, maximum_position=10000):
        """

        :param d_model: 位置编码向量的维度
        :param maximum_position: 可能的最大位置
        """
        self.maximum_position = maximum_position
        self.d_model = d_model

        self.pos_encoding = self.process()

    def get_angles(self, pos, i):

        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(self.d_model))
        return pos * angle_rates

    def process(self):

        angle_rads = self.get_angles(np.arange(self.maximum_position)[:, np.newaxis],
                                  np.arange(self.d_model)[np.newaxis, :])

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, :]

        return tf.cast(pos_encoding, dtype=tf.float32)


class Test:

    def test_SinusoidalPE(self):

        maximum_position = 6
        d_model = 4

        PE = SinusoidalPE(maximum_position=maximum_position, d_model=d_model)


        angle_rads = PE.get_angles(np.arange(maximum_position)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :])

        # angle_rads: 所有 position 的弧度
        print(np.shape(angle_rads.T))
        print(angle_rads.T)

        pos_encoding = PE.pos_encoding.numpy()

        pos_encoding = pos_encoding.transpose(0, 2, 1)

        print(pos_encoding.shape)
        print(pos_encoding)


if __name__ == '__main__':

    test = Test()

    test.test_SinusoidalPE()

