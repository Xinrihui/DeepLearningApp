#!/usr/bin/python
# -*- coding: UTF-8 -*-

#  适用于 tensorflow >= 2.0, keras 被直接集成到 tensorflow 的内部
#  ref: https://keras.io/about/

from tensorflow.keras.layers import Layer, Input, LSTM, TimeDistributed, Bidirectional, Dense, Lambda, Embedding, Dropout, \
    Concatenate, RepeatVector, Activation
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model
import tensorflow.keras as keras


from tensorflow.keras.models import Model

import tensorflow as tf

from tqdm import tqdm

class Seq2seq:
    """
    Seq2seq 模型  (v3-integrated)

    1. 解码采用一体化模型 (integrated model)的方式, 即将每一步的解码都在计算图中完成(时间步的循环控制写在计算图里面)

    2. 训练时采用静态图 (Session execution) 构建模型, 输入的 source 序列 和 输出的 target 序列必须为定长, 使用静态图可以节约显存并加速训练;

    3. 推理时采用动态图(Eager execution)构建模型, 使用动态图可以实现变长的解码

      (1) 每次推理 1 个 源序列, 在 decoder 预测出 <END> 时结束解码,

      (2) 每次推理 1 个 batch 的源序列, 目标序列的长度设置为源序列的长度


    Author: xrh
    Date: 2021-11-20

    ref:
    1. Sequence to Sequence Learning with Neural Networks
    2. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation
    3. Effective Approaches to Attention-based Neural Machine Translation

    """

    def __init__(self,  n_embedding, n_h, max_seq_length,
                 dropout_rates,
                 n_vocab_source, n_vocab_target, vocab_target,
                 _start_target, _null_target,
                 tokenizer_source=None, tokenizer_target=None,
                 reverse_source=True,
                 ):

        super().__init__()

        # 最大的序列长度
        self.max_seq_length = max_seq_length

        # 训练数据中源序列的长度
        self.source_length = self.max_seq_length

        # 训练数据中目标序列的长度
        self.target_length = self.max_seq_length - 1

        self.reverse_source = reverse_source

        # target 中代表 start 的标号
        self._start_target = _start_target

        # target 中代表 null 的标号
        self._null_target = _null_target

        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target

        # 建立编码器和解码器
        self.encoder = Encoder(n_embedding=n_embedding, n_h=n_h, n_vocab=n_vocab_source, dropout_rates=dropout_rates)

        self.train_decoder = TrianDecoder(n_embedding=n_embedding, n_h=n_h, n_vocab=n_vocab_target, target_length=self.target_length, dropout_rates=dropout_rates)

        self.infer_decoder = InferDecoder(train_decoder_obj=self.train_decoder, _start=self._start_target, vocab_target=vocab_target)

        # 建立训练计算图
        self.model_train = self.build_train_graph()

        # 损失函数对象
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')



    def build_train_graph(self):
        """
        将各个 网络层(layer) 拼接为训练计算图

        :return:
        """
        batch_source = Input(shape=(None,), name='batch_source')  # shape (N_batch, encoder_length)

        batch_target_in = Input(shape=(None,), name='batch_target_in')  # shape (N_batch, decoder_length)

        layer_state_list = self.encoder(batch_source)

        outputs_prob = self.train_decoder(batch_target_in, layer_state_list)

        model = Model(inputs=[batch_source, batch_target_in], outputs=outputs_prob)

        return model

    def _mask_loss_function(self, real, pred):
        """
        自定义的损失函数

        :param real: 标签值
        :param pred: 预测值
        :return:
        """
        mask = tf.math.logical_not(tf.math.equal(real, self._null_target))  # 输出序列中为空的不计入损失函数
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def _preprocess(self, batch_data):
        """
        对数据集的 一个批次的数据的预处理

        :param batch_data:
        :return:
        """

        batch_source = batch_data

        batch_source_vector = self.tokenizer_source(batch_source).to_tensor()

        if self.reverse_source:
            batch_source_vector = batch_source_vector[:, ::-1]

        return batch_source_vector


    # @tf.function
    def _test_step(self, batch_source, target_length):

        # batch_source  shape (N_batch, source_length)

        training = False

        layer_state_list = self.encoder(batch_source=batch_source, training=training)

        probs, preds, decode_text = self.infer_decoder(layer_state_list=layer_state_list,
                                   target_length=target_length, training=training)

        return probs, preds, decode_text



    def predict(self, source_dataset, target_length=None):
        """
        输出预测的单词序列

        :param source_dataset:
        :param target_length:
        :return:
        """

        seq_list = []

        # 遍历数据集
        for batch_data in tqdm(source_dataset):

            batch_source = self._preprocess(batch_data)

            if target_length is None:

                target_length = tf.shape(batch_source)[1]  # 源句子的长度决定了推理出的目标句子的长度

            _, _, decode_seq = self._test_step(batch_source, target_length)

            for seq in decode_seq:
                seq_list.append(seq)

        return seq_list

class Encoder(Layer):
    """
    基于 LSTM 的编码器层

    """

    def __init__(self, n_embedding, n_h, n_vocab, dropout_rates):

        super(Encoder, self).__init__()

        self.embedding_layer = Embedding(n_vocab, n_embedding)

        self.lstm_layer0 = LSTM(n_h, return_sequences=True, return_state=True)
        # self.dropout_layer0 = Dropout(dropout_rates[0])  # 神经元有 dropout_rates[0] 的概率被弃置


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embedding_layer': self.embedding_layer,
            'lstm_layer0': self.lstm_layer0,
            # 'dropout_layer0': self.dropout_layer0,
        })
        return config

    def call(self, batch_source, training=True):

        # batch_source shape (N_batch, source_length)

        source_embedding = self.embedding_layer(inputs=batch_source)  # shape (N_batch, encoder_length, n_embedding)

        layer_state_list = []

        # layer0
        out_lstm0, h0, c0 = self.lstm_layer0(
            inputs=source_embedding)  # out_lstm0 shape : (N_batch, source_length, n_h)

        layer_state_list.append((h0, c0))
        # dropout0 = self.dropout_layer0(inputs=out_lstm0, training=training)

        return layer_state_list


class TrianDecoder(Layer):
    """
    训练模式下的基于 LSTM 的解码器层,
    计算图中包含所有时间步, 但是不在计算图中手工展开时间步, 这样可以加速训练

    """

    def __init__(self, n_embedding, n_h, n_vocab, target_length, dropout_rates):

        super(TrianDecoder, self).__init__()

        self.target_length = target_length

        self.embedding_layer = Embedding(n_vocab, n_embedding)

        self.lstm_layer0 = LSTM(n_h, return_sequences=True, return_state=True)
        self.dropout_layer0 = Dropout(dropout_rates[0])  # 神经元有 dropout_rates[0] 的概率被弃置

        self.fc_layer = Dense(n_vocab)

        self.softmax_layer = Activation('softmax', dtype='float32')

    def get_config(self):

        config = super().get_config().copy()

        config.update({
            'target_length': self.target_length,
            'embedding_layer': self.embedding_layer,
            'lstm_layer0': self.lstm_layer0,
            'dropout_layer0': self.dropout_layer0,
            'fc_layer': self.fc_layer,
            'softmax_layer': self.softmax_layer,
        })
        return config

    def call(self, batch_target_in, layer_state_list, training=True):

        # batch_target_in shape (N_batch, target_length)

        batch_target_embbeding = self.embedding_layer(inputs=batch_target_in)
        # shape (N_batch, target_length, n_embedding)

        # 第 0 层的编码器LSTM 的隐藏层
        h0 = layer_state_list[0][0]  # shape: (N_batch, n_h)
        c0 = layer_state_list[0][1]  # shape: (N_batch, n_h)

        # outs_prob = []

        out_lstm0, h0, c0 = self.lstm_layer0(inputs=batch_target_embbeding, initial_state=[h0, c0])  # 输入 context 只有1个时间步

        out_dropout0 = self.dropout_layer0(out_lstm0, training=training)  # shape (N_batch, target_length, n_vocab)

        out = self.fc_layer(out_dropout0)  # shape (N_batch, target_length, n_vocab)

        outputs_prob = self.softmax_layer(out)

        return outputs_prob


class TrianDecoderUnroll(Layer):
    """
    训练模式下的基于 LSTM 的解码器层,
    在计算图中手工展开时间步

    """

    def __init__(self, n_embedding, n_h, n_vocab, target_length, dropout_rates):

        super(TrianDecoderUnroll, self).__init__()

        self.target_length = target_length

        self.embedding_layer = Embedding(n_vocab, n_embedding)

        self.lstm_layer0 = LSTM(n_h, return_sequences=True, return_state=True)
        self.dropout_layer0 = Dropout(dropout_rates[0])  # 神经元有 dropout_rates[0] 的概率被弃置

        self.fc_layer = Dense(n_vocab)

        self.softmax_layer = Activation('softmax', dtype='float32')

    def get_config(self):

        config = super().get_config().copy()

        config.update({
            'target_length': self.target_length,
            'embedding_layer': self.embedding_layer,
            'lstm_layer0': self.lstm_layer0,
            'dropout_layer0': self.dropout_layer0,
            'fc_layer': self.fc_layer,
            'softmax_layer': self.softmax_layer,
        })
        return config

    def call(self, batch_target_in, layer_state_list, training=True):

        # batch_target_in shape (N_batch, target_length)

        batch_target_embbeding = self.embedding_layer(inputs=batch_target_in)
        # shape (N_batch, target_length, n_embedding)

        # 第 0 层的编码器LSTM 的隐藏层
        h0 = layer_state_list[0][0]  # shape: (N_batch, n_h)
        c0 = layer_state_list[0][1]  # shape: (N_batch, n_h)

        # outs_prob = []
        outs_prob = tf.TensorArray(tf.float32, size=self.target_length, clear_after_read=False)

        # target_length = tf.shape(batch_target_in)[1]

        for t in range(self.target_length):  # 使用静态图必须为固定的长度
            # TODO: 若使用 tf.range() 会报错, 详见 logs/../bugs.md

            batch_token_embbeding = tf.expand_dims(batch_target_embbeding[:, t, :], axis=1)
            # Teacher Forcing: 每一个时间步的输入为真实的标签值而不是上一步预测的结果
            # batch_token_embbeding shape (N_batch, 1, n_embedding)

            context = batch_token_embbeding

            out_lstm0, h0, c0 = self.lstm_layer0(inputs=context, initial_state=[h0, c0])  # 输入 context 只有1个时间步

            out_dropout0 = self.dropout_layer0(h0, training=training)

            out = self.fc_layer(out_dropout0)  # shape (N_batch, n_vocab)

            out = self.softmax_layer(out)

            # outs_prob.append(out)  # shape (target_length, N_batch, n_vocab)

            outs_prob = outs_prob.write(t, out)

        outputs_prob = tf.transpose(outs_prob.stack(), perm=[1, 0, 2])  # shape (N_batch, target_length, n_vocab)

        return outputs_prob



class InferDecoder(Layer):
    """
    推理模式下的基于 LSTM 的解码器层
    由于每一步的输入是上一步的输出, 所以必须在计算图中手工展开时间步

    """

    def __init__(self, train_decoder_obj, _start, vocab_target):

        super(InferDecoder, self).__init__()

        self.train_decoder_obj = train_decoder_obj
        self._start = _start

        self.embedding_layer = self.train_decoder_obj.embedding_layer

        self.lstm_layer0 = self.train_decoder_obj.lstm_layer0
        self.dropout_layer0 = self.train_decoder_obj.dropout_layer0

        self.fc_layer = self.train_decoder_obj.fc_layer
        self.softmax_layer = self.train_decoder_obj.softmax_layer

        self.vocab_target = vocab_target

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'train_decoder_obj': self.train_decoder_obj,
            '_start': self._start,
            'embedding_layer': self.embedding_layer,
            'lstm_layer0': self.lstm_layer0,
            'dropout_layer0': self.dropout_layer0,
            'fc_layer': self.fc_layer,
            'vocab_target': self.vocab_target,
            'softmax_layer': self.softmax_layer,
        })
        return config

    def call(self, layer_state_list, target_length, training=False):
        """

        :param layer_state_list:
        :param target_length:
        :param training:
        :return: outputs_prob - shape (N_batch, target_length, n_vocab) 概率形式的推理结果
                outputs - shape (N_batch, target_length) 标号形式的推理结果
        """

        # 第 0 层的编码器LSTM 的隐藏层
        h0 = layer_state_list[0][0]  # shape: (N_batch, n_h)
        c0 = layer_state_list[0][1]  # shape: (N_batch, n_h)

        N_batch = tf.shape(h0)[0]
        batch_token = tf.ones((N_batch, 1)) * self._start  # (N_batch, 1)

        # outs_prob = []
        outs_prob = tf.TensorArray(tf.float32, size=target_length, clear_after_read=False)

        # outs = []
        outs = tf.TensorArray(tf.int64, size=target_length, clear_after_read=False)


        for t in tf.range(target_length):

            batch_token_embbeding = self.embedding_layer(batch_token)  # shape (N_batch, 1, n_embedding)

            context = batch_token_embbeding

            out_lstm0, h0, c0 = self.lstm_layer0(inputs=context, initial_state=[h0, c0])  # 输入 context 只有1个时间步

            out_dropout0 = self.dropout_layer0(h0, training=training)

            out = self.fc_layer(out_dropout0)  # shape (N_batch, n_vocab)

            out = self.softmax_layer(out)

            max_idx = tf.math.argmax(out, axis=1)  # shape (N_batch, )

            # print('max_idx', max_idx)

            batch_token = tf.expand_dims(max_idx, axis=1)  # shape (N_batch, 1)

            # outs_prob.append(out)  # shape (target_length, N_batch, n_vocab)
            outs_prob = outs_prob.write(t, out)

            # outs.append(max_idx)  # shape (target_length, N_batch)
            outs = outs.write(t, max_idx)

        outputs_prob = tf.transpose(outs_prob.stack(), perm=[1, 0, 2])  # 每一个时间步的概率列表 shape (N_batch, target_length, n_vocab)

        outputs = tf.transpose(outs.stack(), perm=[1, 0])  # 单词标号序列 shape (N_batch, target_length)

        decode_seq = self.vocab_target.map_id_to_word(outputs)  # 解码后的单词序列 shape (N_batch, target_length)

        decode_text = tf.strings.reduce_join(decode_seq, axis=1, separator=' ')  # 单词序列 join 成句子

        return outputs_prob, outputs, decode_text


