#!/usr/bin/python
# -*- coding: UTF-8 -*-

#  适用于 tensorflow >= 2.0, keras 被直接集成到 tensorflow 的内部

from tensorflow.keras.layers import Layer, Input, Dense, Embedding, \
    Dropout, Activation, LayerNormalization

from tensorflow.keras.utils import plot_model

from tensorflow.keras.models import Model

import tensorflow as tf

from tqdm import tqdm

import numpy as np
import time


from lib.attention_layer_xrh import *

from lib.position_encode_xrh import *

from lib.mask_xrh import *

from lib.optimizer_xrh import *

class TransformerSeq2seq:
    """

    Seq2seq with Transformer 模型


    1. 训练时用以下方式生成 计算图 (Graph):

     (1) 使用 Eager execution 构建模型, 调试完成后, 利用框架提供的 tf.function 从代码中自动抽取出计算图, 再喂入大量数据训练,
         显然最终也是做了 graph execution ,

    2. 推理时采用  Eager execution 构建模型：

      每次推理 1 个 batch 的源序列, 解码的时间步设置为固定长度(测试集中最长的序列的长度),
      对每一个源序列, 在 decoder 预测出 <END> 时就结束解码, 即剩余的时间步都填充 null

    3.关于 mask 机制
     (1) 对于填充 null 的时间步 (padding) 不计入损失函数中

    Author: xrh
    Date: 2021-12-15

    ref:
    1. Attention Is All You Need
    2. https://tensorflow.google.cn/text/tutorials/transformer

    """

    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rates, label_smoothing,
                 maximum_position_source, maximum_position_target,
                 max_seq_length,
                 n_vocab_source, n_vocab_target,
                 _null_source, _start_target, _null_target, _end_target,
                 tokenizer_source, tokenizer_target,
                 reverse_source=True,
                 build_mode='Eager',
                 ):
        """

        :param num_layers: 堆叠的编码器的层数
        :param d_model: 模型整体的隐藏层的维度
        :param num_heads: 并行注意力层的个数(头数)
        :param dff: Position-wise Feed-Forward 的中间层的维度
        :param dropout_rates: dropout 的弃置率
        :param label_smoothing: 标签平滑
        :param maximum_position_source: 源句子的可能最大长度
        :param maximum_position_target: 目标句子的可能最大长度
        :param max_seq_length: 最大的序列长度
        :param n_vocab_source: 源语言的词表大小
        :param n_vocab_target: 目标语言的词表大小
        :param _null_source: 源序列的填充标号
        :param _start_target: 目标序列的开始标号
        :param _null_target: 目标序列的填充标号
        :param _end_target: 目标序列的结束标号
        :param tokenizer_source: 源语言的分词器
        :param tokenizer_target: 目标语言的分词器
        :param reverse_source: 是否将源序列倒置
        :param build_mode: 建立训练计算图的方式
        """

        super().__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rates = dropout_rates
        self.maximum_position_source = maximum_position_source
        self.maximum_position_target = maximum_position_target

        # 最大的序列长度
        self.max_seq_length = max_seq_length

        # 训练数据中源序列的长度
        self.source_length = self.max_seq_length

        # 训练数据中目标序列的长度
        self.target_length = self.max_seq_length - 1

        self.n_vocab_source = n_vocab_source
        self.n_vocab_target = n_vocab_target

        self.reverse_source = reverse_source

        # source 中代表 null 的标号
        self._null_source = _null_source

        # target 中代表 start 的标号
        self._start_target = _start_target

        # target 中代表 null 的标号
        self._null_target = _null_target

        # target 中代表 end 的标号
        self._end_target = _end_target

        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target

        if build_mode == 'Eager':

            self.model_train = TrainModel(
                num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, dropout_rates=dropout_rates, label_smoothing=label_smoothing,
                n_vocab_source=n_vocab_source, n_vocab_target=n_vocab_target,
                _null_source=_null_source, _null_target=_null_target,
                maximum_position_source=maximum_position_source, maximum_position_target=maximum_position_target,
                tokenizer_source=tokenizer_source, tokenizer_target=tokenizer_target
                )

            self.encoder = self.model_train.encoder
            self.train_decoder = self.model_train.decoder

            self.infer_decoder = InferDecoder(train_decoder_obj=self.train_decoder,
                                              _start_target=self._start_target, _end_target=self._end_target,
                                              _null_target=self._null_target,
                                              tokenizer_target=self.tokenizer_target)

        else:
            raise Exception("Invalid param value, build_mode= ", build_mode)


    def _preprocess_infer(self, batch_data):
        """
        对数据集的 一个批次的数据的预处理

        :param batch_data:
        :return:
        """

        batch_source = batch_data

        batch_source_vector = self.tokenizer_source.tokenize(batch_source).to_tensor()

        return batch_source_vector


    # 1.调试的时候去掉  @tf.function 装饰器
    # 2.input_signature 规定了函数参数的类型, 在重复收到规定类型的输入不会重新构建计算图
    #   shape=[None, None] 表示张量的维度是2维, None 表示可以取任意值
    #   shape=None 表示标量
    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.int64, shape=[None, None]), tf.TensorSpec(dtype=tf.int32, shape=None)])
    def test_step(self, batch_source, target_length):
        """

        :param batch_source: shape (N_batch, source_length)
        :param target_length:
        :return:
        """
        training = False

        encoder_padding_mask = create_padding_mask(batch_source, self._null_source)

        encoder_output = self.encoder(x=batch_source, training=training, padding_mask=encoder_padding_mask)
        # encoder_output shape (N_batch, source_length, d_model)

        # outputs shape  (N_batch, target_length, n_vocab_target)
        preds, decode_text = self.infer_decoder(
            target_length=target_length, encoder_output=encoder_output, training=training,
            padding_mask=encoder_padding_mask)

        return preds, decode_text

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

            batch_source = self._preprocess_infer(batch_data)

            if target_length is None:
                target_length = tf.shape(batch_source)[1]  # 源句子的长度决定了推理出的目标句子的长度

            _, decode_text = self.test_step(batch_source, target_length)

            for sentence in decode_text:
                seq_list.append(sentence)

        return seq_list


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
        self.dropout2 = Dropout(dropout_rates[1])

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
        self.dropout2 = Dropout(dropout_rates[1])
        self.dropout3 = Dropout(dropout_rates[2])

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


class Encoder(Layer):
    """
    将多层的编码器层进行堆叠

    """

    def __init__(self, num_layers, d_model, num_heads, dff, n_vocab_source,
                 pos_encoding, dropout_rates):
        """

        :param num_layers: 堆叠的编码器的层数
        :param d_model: 模型整体的隐藏层的维度
        :param num_heads: 并行注意力层的个数(头数)
        :param dff: Position-wise Feed-Forward Networks 的中间层的维度
        :param n_vocab_source: 源语言的词表大小
        :param pos_encoding: 位置编码张量(包括所有位置)
        :param dropout_rates: dropout 的弃置率
        """
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embedding(n_vocab_source, d_model)
        self.pos_encoding = pos_encoding

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rates)
                           for _ in range(num_layers)]

        self.dropout = Dropout(dropout_rates[-1])

    def get_config(self):
        config = super().get_config().copy()

        config.update({

            'd_model': self.d_model,
            'num_layers': self.num_layers,

            'embedding': self.embedding,
            'pos_encoding': self.pos_encoding,

            'enc_layers': self.enc_layers,
            'dropout': self.dropout,

        })
        return config

    def call(self, x, training, padding_mask):
        """

        :param x: 输入的 tensor
        :param training: 是否为训练模式
        :param padding_mask: 序列的填充 mask
        :return:
        """

        seq_length = tf.shape(x)[1]

        out_embed = self.embedding(x)  # shape (N_batch, source_length, d_model)
        out_embed *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # 加入位置编码信息
        # 对 pos_encoding 根据序列长度进行截取
        out_embed += self.pos_encoding[:, :seq_length, :]

        x = self.dropout(out_embed, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, padding_mask)

        return x  # shape (N_batch, source_length, d_model)


class TrainDecoder(Layer):
    """
    将多层解码器层进行堆叠

    """

    def __init__(self, num_layers, d_model, num_heads, dff, n_vocab_target,
                 pos_encoding, dropout_rates):
        """

        :param num_layers: 堆叠的编码器的层数
        :param d_model: 模型整体的隐藏层的维度
        :param num_heads: 并行注意力层的个数(头数)
        :param dff: Position-wise Feed-Forward Networks 的中间层的维度
        :param n_vocab_target: 源语言的词表大小
        :param pos_encoding: 位置编码张量(包括所有位置)
        :param dropout_rates: dropout 的弃置率
        """

        super(TrainDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embedding(n_vocab_target, d_model)
        self.pos_encoding = pos_encoding

        self.dec_layer_list = [DecoderLayer(d_model, num_heads, dff, dropout_rates)
                               for _ in range(num_layers)]
        self.dropout = Dropout(dropout_rates[-1])

        self.fc = Dense(n_vocab_target)

        self.softmax = Activation('softmax', dtype='float32')



    def get_config(self):
        config = super().get_config().copy()

        config.update({

            'd_model': self.d_model,
            'num_layers': self.num_layers,

            'embedding': self.embedding,
            'pos_encoding': self.pos_encoding,

            'dec_layer_list': self.dec_layer_list,
            'dropout': self.dropout,
            'fc': self.fc,
            'softmax': self.softmax,

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

        seq_len = tf.shape(x)[1]

        attention_weights = {}

        out_embed = self.embedding(x)  # shape (N_batch, target_seq_len, d_model)
        out_embed *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # 加入位置编码信息
        # 对 pos_encoding 根据序列长度进行截取
        out_embed += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(out_embed, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layer_list[i](x, encoder_output, training,
                                                       look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        # x shape (N_batch, target_seq_len, d_model)

        out = self.fc(x)  # out shape (N_batch, target_seq_len, n_vocab_target)

        out_prob = self.softmax(out)  # shape (N_batch, target_seq_len, n_vocab_target)

        return out_prob, attention_weights


class InferDecoder(Layer):
    """
    将多层解码器层进行堆叠

    """

    def __init__(self, train_decoder_obj, _start_target, _end_target, _null_target, tokenizer_target):

        super(InferDecoder, self).__init__()

        self.train_decoder_obj = train_decoder_obj
        self._start_target = _start_target
        self._end_target = _end_target
        self._null_target = _null_target
        self.tokenizer_target = tokenizer_target

        self.d_model = self.train_decoder_obj.d_model
        self.num_layers = self.train_decoder_obj.num_layers

        self.embedding = self.train_decoder_obj.embedding
        self.pos_encoding = self.train_decoder_obj.pos_encoding

        self.dec_layer_list = self.train_decoder_obj.dec_layer_list

        self.dropout = self.train_decoder_obj.dropout

        self.fc = self.train_decoder_obj.fc

    def call(self, target_length, encoder_output, training,
             padding_mask):
        """

        :param target_length: 解码的目标长度
        :param encoder_output: 编码器的输出序列 shape (N_batch, input_seq_len, d_model)
        :param training: 是否为训练模式
        :param padding_mask: 序列的填充 mask
        :return:
        """

        N_batch = tf.shape(encoder_output)[0]

        start_token = tf.ones((N_batch,), dtype=tf.int64) * self._start_target  # (N_batch, )

        outs = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        outs = outs.write(0, start_token)  # shape (1, N_batch)

        done = tf.zeros((N_batch, ), dtype=tf.bool)  # 标记序列的解码可以结束

        for t in tf.range(target_length):  # 使用 tf.range 会触发 tf.autograph 将循环也构成计算图的一部分

            batch_tokens = tf.transpose(outs.stack())  # shape (N_batch, t+1)

            out_embed = self.embedding(batch_tokens)  # shape (N_batch, t+1, d_model)

            out_embed *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

            # 加入位置编码信息
            # 对 pos_encoding 根据序列长度进行截取
            out_embed += self.pos_encoding[:, :t + 1, :]  # shape (N_batch, t+1, d_model)

            x = self.dropout(out_embed, training=training)

            for i in range(self.num_layers):
                x, block1, block2 = self.dec_layer_list[i](x=x, encoder_output=encoder_output, training=training,
                                                           look_ahead_mask=None, padding_mask=padding_mask)

            x = self.fc(x)  # shape (N_batch, t+1, n_vocab_target)

            # 最后一个时间步的输出
            out = x[:, -1, :]  # (N_batch, vocab_size)

            max_idx = tf.math.argmax(out, axis=-1)  # shape (N_batch, )

            # print('max_idx', max_idx)

            # 若出现结束标记位, 则置此序列的状态为 '解码结束' (True)
            # 注意这里是 '或', 也就是只要出现一次结束标记位之后 done 数组中表示此序列的位一直为 True
            done = done | (max_idx == self._end_target)  # shape (N_batch, )
            # 若序列的状态被置为 '解码结束', 则 后面的时间步都填充 null 元素
            batch_token = tf.where(done, tf.constant(self._null_target, dtype=tf.int64), max_idx)  # shape (N_batch, )

            outs = outs.write(t+1, batch_token)  # shape (t+2, N_batch)

            if tf.reduce_all(done):
                break

        outputs = tf.transpose(outs.stack(), perm=[1, 0])  # 单词标号序列 shape (N_batch, target_length)

        outputs = outputs[:, 1:]  # 第1个时间步是开始标记可以忽略

        vectors = self.tokenizer_target.detokenize(outputs)
        text = tf.strings.reduce_join(vectors, separator=' ', axis=-1)

        return outputs, text


class TrainModel(Model):

    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rates, label_smoothing,
                 maximum_position_source, maximum_position_target,
                 n_vocab_source, n_vocab_target,
                 _null_source, _null_target,
                 tokenizer_source=None, tokenizer_target=None
                 ):
        """

        :param num_layers: 堆叠的编码器的层数
        :param d_model: 模型整体的隐藏层的维度
        :param num_heads: 并行注意力层的个数(头数)
        :param dff: Position-wise Feed-Forward Networks 的中间层的维度
        :param dropout_rates: dropout 的弃置率
        :param label_smoothing: 标签平滑
        :param maximum_position_source: 源句子的可能最大长度
        :param maximum_position_target: 目标句子的可能最大长度
        :param n_vocab_source: 源语言的词表大小
        :param n_vocab_target: 目标语言的词表大小
        :param _null_source: 源序列的填充标号
        :param _null_target: 目标序列的填充标号
        :param tokenizer_source: 源语言的分词器
        :param tokenizer_target: 目标语言的分词器

        """

        super().__init__()

        PE_source = SinusoidalPE(maximum_position=maximum_position_source, d_model=d_model)
        PE_target = SinusoidalPE(maximum_position=maximum_position_target, d_model=d_model)

        self._null_source = _null_source
        self._null_target = _null_target

        self.n_vocab_target = n_vocab_target

        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               n_vocab_source, PE_source.pos_encoding, dropout_rates)

        self.decoder = TrainDecoder(num_layers, d_model, num_heads, dff,
                                    n_vocab_target, PE_target.pos_encoding, dropout_rates)

        # 损失函数对象
        # self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

        self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction='none', label_smoothing=label_smoothing)


        self.loss_tracker = tf.keras.metrics.Mean(name='train_loss')
        self.accuracy_metric = tf.keras.metrics.Mean(name='train_accuracy')

        learning_rate = WarmupSchedule(d_model)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                             epsilon=1e-9)

    def call(self, inputs, training):
        """

        :param inputs:
        :param training:
        :return:
        """

        # 在第1个参数中包括所有的输入 tensor
        source, target_in = inputs

        encoder_padding_mask, look_ahead_mask = create_masks(source=source, target=target_in,
                                                              _null_source=self._null_source, _null_target=self._null_target)

        encoder_output = self.encoder(x=source, training=training, padding_mask=encoder_padding_mask)
        # encoder_output shape (N_batch, source_length, d_model)

        # decoder_output shape  (N_batch, target_length, n_vocab_target)
        decoder_output, attention_weights = self.decoder(
            x=target_in, encoder_output=encoder_output, training=training,
            look_ahead_mask=look_ahead_mask, padding_mask=encoder_padding_mask)

        return decoder_output, attention_weights

    def _mask_loss_function(self, y_true, y_pred):
        """
        考虑 mask 的损失函数

        :param y_true: 标签值
        :param y_pred: 预测值
        :return:
        """
        y_true_dense = tf.argmax(y_true, axis=-1)
        # y_true_dense = y_true

        mask = (y_true_dense != self._null_target)  # 输出序列中为空的不计入损失函数

        loss_ = self.loss_object(y_true, y_pred)

        mask = tf.cast(mask, dtype=loss_.dtype)

        loss_ *= mask

        return tf.reduce_mean(loss_)

    def _mask_accuracy_function(self, y_true, y_pred):
        """

        :param y_true: 标签值
        :param y_pred: 预测值
        :return:
        """
        y_true_dense = tf.argmax(y_true, axis=-1)
        # y_true_dense = y_true

        accuracies = tf.equal(y_true_dense, tf.argmax(y_pred, axis=-1))
        mask = tf.math.logical_not(tf.math.equal(y_true_dense, self._null_target))

        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)

        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    def _preprocess_train(self, source, target):
        """
        对数据集的 一个批次的数据的预处理

        :param source: shape (N_batch, )
        :param target: shape (N_batch, )
        :return:
        """

        source_vector = self.tokenizer_source.tokenize(source).to_tensor()
        target_vector = self.tokenizer_target.tokenize(target).to_tensor()

        target_in = target_vector[:, :-1]
        target_out = target_vector[:, 1:]

        target_out_one_hot = tf.one_hot(indices=target_out, depth=self.n_vocab_target,
                                       on_value=1, off_value=0, dtype=tf.int64,
                                       axis=-1)


        return (source_vector, target_in), target_out_one_hot

    # @tf.function 将 train_step 编译为 计算图，以便更快地执行;
    # input_signature 指定了 输入张量的 shape, 可以避免重复建立计算图,
    # 若不指定 shape, 一个 epoch 的最后一批数据的 N_batch 与之前不同, 会触发耗时的 trace 操作
    signature = [
        (
            tf.TensorSpec(shape=(None, ), dtype=tf.string),
            tf.TensorSpec(shape=(None, ), dtype=tf.string)
        )
    ]
    @tf.function(input_signature=signature)
    def train_step(self, data):

        source, target = data

        (batch_source_vector, batch_target_in), batch_target_out = self._preprocess_train(source, target)

        with tf.GradientTape() as tape:

            predictions, _ = self([batch_source_vector, batch_target_in],
                                         training=True)
            loss = self._mask_loss_function(batch_target_out, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.accuracy_metric.update_state(self._mask_accuracy_function(batch_target_out, predictions))

        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy_metric.result()}

    def fit_debug(self, x, epochs, validation_data=None, verbose=None, callbacks=None):
        """
        使用 model.fit 自动开启 graph execution, 因此无法进入调试模式查看 tensor 具体的值

        :param x:
        :param epochs:
        :param validation_data:
        :param verbose:
        :param callbacks:
        :return:
        """

        for epoch in range(epochs):
            start = time.time()

            self.loss_tracker.reset_states()
            self.accuracy_metric.reset_states()


            for (batch, batch_data) in enumerate(x):

                self.train_step(batch_data)

                if batch % 50 == 0:
                    print(
                        f'Epoch {epoch + 1} Batch {batch} Loss {self.loss_tracker.result():.4f} Accuracy {self.accuracy_metric.result():.4f}')


            print(f'Epoch {epoch + 1} Loss {self.loss_tracker.result():.4f} Accuracy {self.accuracy_metric.result():.4f}')

            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

    @tf.function(input_signature=signature)
    def test_step(self, data):

        source, target = data

        # tf.print(source.numpy())

        (batch_source_vector, batch_target_in), batch_target_out = self._preprocess_train(source, target)

        # tf.print(source.numpy())

        predictions, _ = self([batch_source_vector, batch_target_in],
                                     training=True)
        loss = self._mask_loss_function(batch_target_out, predictions)

        self.loss_tracker.update_state(loss)
        self.accuracy_metric.update_state(self._mask_accuracy_function(batch_target_out, predictions))

        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy_metric.result()}


    # @property
    # def metrics(self):
    #     # We list our `Metric` objects here so that `reset_states()` can be
    #     # called automatically at the start of each epoch
    #     # or at the start of `evaluate()`.
    #     # If you don't implement this property, you have to call
    #     # `reset_states()` yourself at the time of your choosing.
    #     return [self.loss_tracker, self.accuracy_metric]

class Test:

    def test_TrainModel(self):
        num_layers = 2
        n_h = 32
        num_heads = 4

        n_vocab_source = 50
        n_vocab_target = 50

        _null_source = 0
        _null_target = 0

        maximum_position_source = 1000
        maximum_position_target = 600

        dropout_rates = [0.1, 0.1, 0.1, 0.1, 0.1]

        model = TrainModel(
            num_layers=num_layers, d_model=n_h, num_heads=num_heads, dff=2048, dropout_rates=dropout_rates, label_smoothing=0.1,
            n_vocab_source=n_vocab_source, n_vocab_target=n_vocab_target,
            _null_source=_null_source, _null_target=_null_target,
            maximum_position_source=maximum_position_source, maximum_position_target=maximum_position_target,
            )

        N_batch = 4
        source_length = 6
        target_length = 5

        batch_source = np.random.randint(10, size=(N_batch, source_length))
        batch_target_in = np.random.randint(10, size=(N_batch, target_length))

        inputs_tuple = (batch_source, batch_target_in)

        outputs_prob, _ = model.call(inputs_tuple, training=True)

        print(tf.shape(outputs_prob))


if __name__ == '__main__':
    test = Test()

    test.test_TrainModel()
