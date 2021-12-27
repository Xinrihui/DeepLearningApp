#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf


class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    带 warmup 的学习率衰减策略


    ref:
    1. Attention Is All You Need
    2. https://tensorflow.google.cn/text/tutorials/transformer

    """

    def __init__(self, d_model, warmup_steps=4000):
        super(WarmupSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
