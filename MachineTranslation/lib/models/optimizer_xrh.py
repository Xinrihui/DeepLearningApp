#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf


class LegacyWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    带 warmup 的学习率衰减策略, 参考 tensor2tensor 中的 "legacy" 策略

    ref:
    1. Attention Is All You Need
    2. https://tensorflow.google.cn/text/tutorials/transformer
    3. https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/learning_rate.py

    """
    def __init__(self, d_model, warmup_steps=4000):
        super(LegacyWarmupSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):

        arg1 = tf.math.rsqrt(step)  # rsqrt 等价于 **(-0.5)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class RsqrtDecayWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    带 warmup 的学习率衰减策略, 参考 tensor2tensor 中的 " constant * linear_warmup * rsqrt_decay " 策略

    ref:
    1. Attention Is All You Need
    2. https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/learning_rate.py

    """
    def __init__(self, d_model, warmup_steps=16000, learning_rate=0.1):
        super(RsqrtDecayWarmupSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
        self.learning_rate = tf.cast(learning_rate, tf.float32)

    def __call__(self, step):

        constant = self.learning_rate
        linear_warmup = tf.minimum(1.0, step / self.warmup_steps)
        rsqrt_decay = tf.math.rsqrt(tf.maximum(step, self.warmup_steps))
        ret = constant * linear_warmup * rsqrt_decay

        return ret