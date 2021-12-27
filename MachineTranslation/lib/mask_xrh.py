#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf


def create_padding_mask(seq, _null):
    seq = tf.cast(tf.math.equal(seq, _null), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # shape (N_batch, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # shape (seq_len, seq_len)

def create_masks(source, target, _null_source, _null_target):
    """

    :param source: 源序列 shape (N_batch, source_seq_len)
    :param target: 目标序列 shape (N_batch, target_seq_len)
    :param _null_source:
    :param _null_target:

    :return:
    """

    # 根据输入序列生成 编码器的 mask, Attention 模块不对输入序列中填充 null 的时间步分配注意力
    encoder_padding_mask = create_padding_mask(source, _null_source)
    # 在解码器中的 cross attention 中也使用此 mask

    # 根据输出序列生成 解码器的 mask, 此 mask 避免 Attention 模块看到未来的时间步
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])

    # 根据输出序列生成 解码器的 mask, Attention 模块不对输出序列中填充 null 的时间步分配注意力
    dec_target_padding_mask = create_padding_mask(target, _null_target)

    # 两个 mask 求交集
    look_ahead_mask_final = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return encoder_padding_mask, look_ahead_mask_final
