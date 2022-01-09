#!/usr/bin/python
# -*- coding: UTF-8 -*-

#  适用于 tensorflow >= 2.0, keras 被直接集成到 tensorflow 的内部
#  ref: https://keras.io/about/

from tensorflow.keras.layers import Input, RNN, LSTM, Embedding, \
    Dropout, Activation, LSTMCell

from tensorflow.keras.models import Model

from tqdm import tqdm

from lib.layers.attention_layer_xrh import *


class AttentionSeq2seq:
    """

    Seq2seq with Attention 模型

    1. 解码采用一体化模型 (integrated model)的方式, 即将每一步的解码都在计算图中完成 (时间步的循环控制写在计算图里面)

    2. 训练时可以采用两种方式生成 计算图 (Graph):

     (1) 自己建立计算图 (Session execution) ;
         训练速度: RTX3090 588ms/step

     (2) 使用 Eager execution 构建模型, 调试完成后, 利用框架提供的 tf.function 从代码中自动抽取出计算图, 再喂入大量数据训练,
         虽然最终也是做了 graph execution , 但是每一个 iteration 执行速度不如 Session execution ;
         训练速度: RTX3090 693ms/step

    3. 推理时采用  Eager execution 构建模型：

      每次推理 1 个 batch 的源序列, 解码的时间步设置为固定长度(测试集中最长的序列的长度),
      对每一个源序列, 在 decoder 预测出 <END> 时就结束解码, 即剩余的时间步都填充 null

    4.使用多层的 LSTM 堆叠, 中间使用 dropout 连接, 并加入 attention 机制

    5. 实现了 2种 attention 机制:
      (1) dot-product / Luong / general  attention
      (2) additive / Bahdanau / concat attention

    6.关于 mask 机制
     (1) 对于填充 null 的时间步(padding)不计入损失函数中
     (2) 根据 输入序列生成 编码器的 mask, Attention 模块不对输入序列中填充 null 的时间步分配注意力

    Author: xrh
    Date: 2021-12-5

    ref:
    1. Sequence to Sequence Learning with Neural Networks
    2. Neural machine translation by jointly learning to align and translate
    3. Effective Approaches to Attention-based Neural Machine Translation
    4. https://nlp.stanford.edu/projects/nmt/
    5. https://www.tensorflow.org/text/tutorials/nmt_with_attention

    """

    def __init__(self, n_embedding, n_h, max_seq_length,
                 dropout_rates,
                 n_vocab_source, n_vocab_target,
                 _null_source, _start_target, _null_target, _end_target,
                 tokenizer_source=None, tokenizer_target=None,
                 reverse_source=True,
                 build_mode='Eager',
                 ):

        super().__init__()

        self.n_embedding = n_embedding
        self.n_h = n_h
        self.dropout_rates = dropout_rates

        self.n_vocab_source = n_vocab_source
        self.n_vocab_target = n_vocab_target

        print('model architecture param:')
        print('n_h:{}, n_embedding:{}, n_vocab_source:{}, n_vocab_target:{}'.format(self.n_h, self.n_embedding, self.n_vocab_source,
                                                                                    self.n_vocab_target))
        print('-------------------------')


        # 最大的序列长度
        self.max_seq_length = max_seq_length

        # 训练数据中源序列的长度
        self.source_length = self.max_seq_length

        # 训练数据中目标序列的长度
        self.target_length = self.max_seq_length - 1

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

        # 指定参数初始化器
        # self.initializer = tf.keras.initializers.RandomUniform(
        #     minval=-0.1, maxval=0.1
        # )

        self.initializer = tf.keras.initializers.GlorotUniform()


        if build_mode == 'Session':

            # 建立编码器和解码器
            self.encoder = Encoder(n_embedding=self.n_embedding, n_h=self.n_h, n_vocab=self.n_vocab_source,
                                   dropout_rates=self.dropout_rates, initializer=self.initializer)

            self.train_decoder = TrianDecoder(n_embedding=self.n_embedding, n_h=self.n_h, n_vocab=self.n_vocab_target,
                                              target_length=self.target_length, dropout_rates=self.dropout_rates, initializer=self.initializer)

            # 手工建立计算图
            self.model_train = self.build_train_graph()


            self.model_infer = InferModel(train_encoder=self.encoder,
                                          train_decoder=self.train_decoder,
                                          reverse_source=reverse_source,
                                          _null_source=_null_source,
                                          _start_target=_start_target, _end_target=_end_target,
                                          _null_target=_null_target,
                                          tokenizer_source=tokenizer_source, tokenizer_target=tokenizer_target)


        elif build_mode == 'Eager':

            self.model_train = TrainModel(n_embedding=self.n_embedding, n_h=self.n_h,
                                          target_length=self.target_length, _null_source=self._null_source,
                                          dropout_rates=self.dropout_rates,
                                          n_vocab_source=self.n_vocab_source, n_vocab_target=self.n_vocab_target)

            self.model_infer = InferModel(train_encoder=self.model_train.encoder,
                                          train_decoder=self.model_train.decoder,
                                          reverse_source=reverse_source,
                                          _null_source=_null_source,
                                          _start_target=_start_target, _end_target=_end_target,
                                          _null_target=_null_target,
                                          tokenizer_source=tokenizer_source, tokenizer_target=tokenizer_target)

        else:
            raise Exception("Invalid param value, build_mode= ", build_mode)



        # 损失函数对象
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

    def build_train_graph(self):
        """
        将各个 网络层(layer) 拼接为训练计算图

        :return:
        """

        batch_source = Input(shape=(None,), name='batch_source')  # shape (N_batch, encoder_length)

        batch_target_in = Input(shape=(None,), name='batch_target_in')  # shape (N_batch, decoder_length)

        mask_source = (batch_source != self._null_source)

        layer_state_list, out_encoder = self.encoder(batch_source)

        outputs_prob = self.train_decoder(batch_target_in=batch_target_in, mask_source=mask_source,
                                               layer_state_list=layer_state_list, out_encoder=out_encoder)

        model = Model(inputs=[batch_source, batch_target_in], outputs=outputs_prob)

        return model

    def _mask_loss_function(self, y_true, y_pred):
        """
        自定义的损失函数

        :param y_true: 标签值
        :param y_pred: 预测值
        :return:
        """
        mask = (y_true != self._null_target)  # 输出序列中为空的不计入损失函数

        loss_ = self.loss_object(y_true, y_pred)

        mask = tf.cast(mask, dtype=loss_.dtype)

        loss_ *= mask

        return tf.reduce_mean(loss_)






class Encoder(Layer):
    """
    基于 LSTM 的编码器层

    """

    def __init__(self, n_embedding, n_h, n_vocab, dropout_rates, initializer=None):

        super(Encoder, self).__init__()

        self.embedding_layer = Embedding(n_vocab, n_embedding)
        self.dropout_layer0 = Dropout(dropout_rates[0])

        self.lstm_layer1 = LSTM(n_h, return_sequences=True, return_state=True, kernel_initializer=initializer, recurrent_initializer=initializer)
        self.dropout_layer1 = Dropout(dropout_rates[1])  # 神经元有 dropout_rates[0] 的概率被弃置

        self.lstm_layer2 = LSTM(n_h, return_sequences=True, return_state=True, kernel_initializer=initializer, recurrent_initializer=initializer)
        self.dropout_layer2 = Dropout(dropout_rates[2])

        self.lstm_layer3 = LSTM(n_h, return_sequences=True, return_state=True, kernel_initializer=initializer, recurrent_initializer=initializer)
        self.dropout_layer3 = Dropout(dropout_rates[3])

        self.lstm_layer4 = LSTM(n_h, return_sequences=True, return_state=True, kernel_initializer=initializer, recurrent_initializer=initializer)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embedding_layer': self.embedding_layer,
            'dropout_layer0': self.dropout_layer0,

            'lstm_layer1': self.lstm_layer1,
            'dropout_layer1': self.dropout_layer1,

            'lstm_layer2': self.lstm_layer2,
            'dropout_layer2': self.dropout_layer2,

            'lstm_layer3': self.lstm_layer3,
            'dropout_layer3': self.dropout_layer3,

            'lstm_layer4': self.lstm_layer4,

        })
        return config

    def call(self, batch_source, training=True):
        """

        :param batch_source: shape (N_batch, source_length)
        :param training:
        :return:
        """

        source_embedding = self.embedding_layer(inputs=batch_source)  # shape (N_batch, encoder_length, n_embedding)
        dropout0 = self.dropout_layer0(inputs=source_embedding, training=training)

        layer_state_list = []
        layer_state_list.append((None, None))  # 空出第 0个元素

        # layer1
        out_lstm1, h1, c1 = self.lstm_layer1(inputs=dropout0)  # out_lstm1 shape : (N_batch, source_length, n_h)
        layer_state_list.append((h1, c1))
        dropout1 = self.dropout_layer1(inputs=out_lstm1, training=training)

        # layer2
        out_lstm2, h2, c2 = self.lstm_layer2(
            inputs=dropout1)  # out_lstm2 shape : (N_batch, source_length, n_h)

        layer_state_list.append((h2, c2))
        dropout2 = self.dropout_layer2(inputs=out_lstm2, training=training)

        # layer3
        out_lstm3, h3, c3 = self.lstm_layer3(
            inputs=dropout2)  # out_lstm3 shape : (N_batch, source_length, n_h)

        layer_state_list.append((h3, c3))
        dropout3 = self.dropout_layer3(inputs=out_lstm3, training=training)

        # layer4
        out_lstm4, h4, c4 = self.lstm_layer4(
            inputs=dropout3)  # out_lstm4 shape : (N_batch, source_length, n_h)

        layer_state_list.append((h4, c4))

        return layer_state_list, out_lstm4


class TrianDecoder(Layer):
    """
    训练模式下的基于 LSTM 的解码器层,
    计算图中包含所有时间步, 但是不在计算图中手工展开时间步, 这样可以加速训练(占用显存更少)

    """

    def __init__(self, n_embedding, n_h, n_vocab, target_length, dropout_rates, initializer=None):
        super(TrianDecoder, self).__init__()

        self.target_length = target_length

        self.embedding_layer = Embedding(n_vocab, n_embedding)
        self.dropout_layer0 = Dropout(dropout_rates[0])  # 神经元有 dropout_rates[0] 的概率被弃置

        self.lstm_cell1 = LSTMCell(n_h, kernel_initializer=initializer, recurrent_initializer=initializer)
        self.lstm_layer1 = RNN(self.lstm_cell1, return_sequences=True, return_state=True)
        self.dropout_layer1 = Dropout(dropout_rates[1])

        self.lstm_cell2 = LSTMCell(n_h, kernel_initializer=initializer, recurrent_initializer=initializer)
        self.lstm_layer2 = RNN(self.lstm_cell2, return_sequences=True, return_state=True)
        self.dropout_layer2 = Dropout(dropout_rates[2])

        self.lstm_cell3 = LSTMCell(n_h, kernel_initializer=initializer, recurrent_initializer=initializer)
        self.lstm_layer3 = RNN(self.lstm_cell3, return_sequences=True, return_state=True)
        self.dropout_layer3 = Dropout(dropout_rates[3])

        self.lstm_cell4 = LSTMCell(n_h, kernel_initializer=initializer, recurrent_initializer=initializer)
        self.lstm_layer4 = RNN(self.lstm_cell4, return_sequences=True, return_state=True)
        self.dropout_layer4 = Dropout(dropout_rates[4])

        self.attention_layer = DotAttention()
        self.fc_layer0 = Dense(n_h, activation=tf.math.tanh, use_bias=False, kernel_initializer=initializer)

        self.fc_layer1 = Dense(n_vocab, kernel_initializer=initializer)
        self.softmax_layer = Activation('softmax', dtype='float32')

    def get_config(self):
        config = super().get_config().copy()

        config.update({
            'target_length': self.target_length,

            'embedding_layer': self.embedding_layer,
            'dropout_layer0': self.dropout_layer0,

            'lstm_cell1': self.lstm_cell1,
            'lstm_layer1': self.lstm_layer1,
            'dropout_layer1': self.dropout_layer1,

            'lstm_cell2': self.lstm_cell2,
            'lstm_layer2': self.lstm_layer2,
            'dropout_layer2': self.dropout_layer2,

            'lstm_cell3': self.lstm_cell3,
            'lstm_layer3': self.lstm_layer3,
            'dropout_layer3': self.dropout_layer3,

            'lstm_cell4': self.lstm_cell4,
            'lstm_layer4': self.lstm_layer4,
            'dropout_layer4': self.dropout_layer4,

            'attention_layer': self.attention_layer,
            'fc_layer0': self.fc_layer0,
            'fc_layer1': self.fc_layer1,
            'softmax_layer': self.softmax_layer,
        })
        return config

    def call(self, batch_target_in,
             mask_source,
             layer_state_list, out_encoder,
             training=True):
        """

        :param batch_target_in: shape (N_batch, target_length)
        :param mask_source:
        :param layer_state_list:
        :param out_encoder: shape (N_batch, source_length, n_h)
        :param training:
        :return:
        """

        batch_target_embbeding = self.embedding_layer(inputs=batch_target_in)
        # shape (N_batch, target_length, n_embedding)

        # 第 1 层的编码器LSTM 的隐藏层
        h1 = layer_state_list[1][0]  # shape: (N_batch, n_h)
        c1 = layer_state_list[1][1]  # shape: (N_batch, n_h)

        # 第 2 层的编码器LSTM 的隐藏层
        h2 = layer_state_list[2][0]  # shape: (N_batch, n_h)
        c2 = layer_state_list[2][1]  # shape: (N_batch, n_h)

        # 第 3 层的编码器LSTM 的隐藏层
        h3 = layer_state_list[3][0]  # shape: (N_batch, n_h)
        c3 = layer_state_list[3][1]  # shape: (N_batch, n_h)

        # 第 4 层的编码器LSTM 的隐藏层
        h4 = layer_state_list[4][0]  # shape: (N_batch, n_h)
        c4 = layer_state_list[4][1]  # shape: (N_batch, n_h)

        out_dropout0 = self.dropout_layer0(batch_target_embbeding, training=training)
        # Teacher Forcing: 每一个时间步的输入为真实的标签值而不是上一步预测的结果

        out_lstm1, h1, c1 = self.lstm_layer1(inputs=out_dropout0, initial_state=[h1, c1])
        out_dropout1 = self.dropout_layer1(out_lstm1, training=training)

        out_lstm2, h2, c2 = self.lstm_layer2(inputs=out_dropout1, initial_state=[h2, c2])
        out_dropout2 = self.dropout_layer2(out_lstm2, training=training)

        out_lstm3, h3, c3 = self.lstm_layer3(inputs=out_dropout2, initial_state=[h3, c3])
        out_dropout3 = self.dropout_layer3(out_lstm3, training=training)

        out_lstm4, h4, c4 = self.lstm_layer4(inputs=out_dropout3, initial_state=[h4, c4])
        rnn_output = self.dropout_layer4(out_lstm4, training=training)

        # attention 机制
        context, weights = self.attention_layer(query=rnn_output, value=out_encoder, mask=mask_source)
        # context shape (N_batch, target_length, n_h)
        out_concat = tf.concat([context, rnn_output], axis=-1)  # shape (N_batch, target_length, n_h+n_h)
        out_attention = self.fc_layer0(out_concat)  # shape (N_batch, target_length, n_h)
        # TODO: 加入 attention_layer 后会产生警告: Error in PredictCost() for the op: op: "Softmax" attr, 原因未知

        outputs = self.fc_layer1(out_attention)  # shape (N_batch, target_length, n_vocab)

        outputs_prob = self.softmax_layer(outputs)  # shape (N_batch, target_length, n_vocab)

        return outputs_prob


class InferDecoder(Layer):
    """
    推理模式下的基于 LSTM 的解码器层

    """

    def __init__(self, train_decoder_obj, _start_target, _end_target, _null_target, tokenizer_target):

        super(InferDecoder, self).__init__()

        self.train_decoder_obj = train_decoder_obj
        self._start_target = _start_target
        self._end_target = _end_target
        self._null_target = _null_target

        self.embedding_layer = self.train_decoder_obj.embedding_layer
        self.dropout_layer0 = self.train_decoder_obj.dropout_layer0

        self.lstm_cell1 = self.train_decoder_obj.lstm_cell1
        self.dropout_layer1 = self.train_decoder_obj.dropout_layer1

        self.lstm_cell2 = self.train_decoder_obj.lstm_cell2
        self.dropout_layer2 = self.train_decoder_obj.dropout_layer2

        self.lstm_cell3 = self.train_decoder_obj.lstm_cell3
        self.dropout_layer3 = self.train_decoder_obj.dropout_layer3

        self.lstm_cell4 = self.train_decoder_obj.lstm_cell4
        self.dropout_layer4 = self.train_decoder_obj.dropout_layer4

        self.attention_layer = self.train_decoder_obj.attention_layer
        self.fc_layer0 = self.train_decoder_obj.fc_layer0

        self.fc_layer1 = self.train_decoder_obj.fc_layer1
        self.softmax_layer = self.train_decoder_obj.softmax_layer

        self.tokenizer_target = tokenizer_target

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'train_decoder_obj': self.train_decoder_obj,
            '_start_target': self._start_target,
            '_end_target': self._end_target,
            '_null_target': self._null_target,

            'embedding_layer': self.embedding_layer,
            'dropout_layer0': self.dropout_layer0,

            'lstm_cell1': self.lstm_cell1,
            'dropout_layer1': self.dropout_layer1,

            'lstm_cell2': self.lstm_cell2,
            'dropout_layer2': self.dropout_layer2,

            'lstm_cell3': self.lstm_cell3,
            'dropout_layer3': self.dropout_layer3,

            'lstm_cell4': self.lstm_cell4,
            'dropout_layer4': self.dropout_layer4,

            'attention_layer': self.attention_layer,
            'fc_layer0': self.fc_layer0,
            'fc_layer1': self.fc_layer1,
            'softmax_layer': self.softmax_layer,

            'tokenizer_target': self.tokenizer_target,
        })
        return config

    # @tf.function 生成静态图(graph) 加速计算
    def call(self, target_length, mask_source,
             layer_state_list, out_encoder,
             training=False):
        """
        :param target_length: 解码的目标长度
        :param mask_source: shape (N_batch, source_length)
        :param layer_state_list:
        :param out_encoder: shape (N_batch, source_length, n_h)
        :param training:

        :return: outputs_prob - shape (N_batch, target_length, n_vocab) 概率形式的推理结果
                 outputs - shape (N_batch, target_length) 标号形式的推理结果
                 decode_text - shape (N_batch,) 解码后的句子
        """

        # 第 1 层的编码器LSTM 的隐藏层
        h1 = layer_state_list[1][0]  # shape: (N_batch, n_h)
        c1 = layer_state_list[1][1]  # shape: (N_batch, n_h)

        # 第 2 层的编码器LSTM 的隐藏层
        h2 = layer_state_list[2][0]  # shape: (N_batch, n_h)
        c2 = layer_state_list[2][1]  # shape: (N_batch, n_h)

        # 第 3 层的编码器LSTM 的隐藏层
        h3 = layer_state_list[3][0]  # shape: (N_batch, n_h)
        c3 = layer_state_list[3][1]  # shape: (N_batch, n_h)

        # 第 4 层的编码器LSTM 的隐藏层
        h4 = layer_state_list[4][0]  # shape: (N_batch, n_h)
        c4 = layer_state_list[4][1]  # shape: (N_batch, n_h)

        N_batch = tf.shape(h1)[0]
        batch_token = tf.ones(N_batch, dtype=tf.int64) * self._start_target  # (N_batch, )

        done = tf.zeros(N_batch, dtype=tf.bool)  # 标记序列的解码可以结束

        outs = tf.TensorArray(tf.int64, size=target_length, clear_after_read=False)

        for t in tf.range(target_length):  # 使用 tf.range 会触发 tf.autograph 将循环也构成计算图的一部分

            batch_token_embbeding = self.embedding_layer(batch_token)  # shape (N_batch, n_embedding)

            out_dropout0 = self.dropout_layer0(batch_token_embbeding, training=training)

            out_lstm1, (h1, c1) = self.lstm_cell1(inputs=out_dropout0, states=[h1, c1])
            out_dropout1 = self.dropout_layer1(out_lstm1, training=training)

            out_lstm2, (h2, c2) = self.lstm_cell2(inputs=out_dropout1, states=[h2, c2])
            out_dropout2 = self.dropout_layer2(out_lstm2, training=training)

            out_lstm3, (h3, c3) = self.lstm_cell3(inputs=out_dropout2, states=[h3, c3])
            out_dropout3 = self.dropout_layer3(out_lstm3, training=training)

            out_lstm4, (h4, c4) = self.lstm_cell4(inputs=out_dropout3, states=[h4, c4])  # 输入 context 只有1个时间步

            out_dropout4 = self.dropout_layer4(out_lstm4, training=training)  # shape (N_batch, n_h)

            # attention 机制
            rnn_output = tf.expand_dims(out_dropout4, axis=1)  # shape (N_batch, 1, n_h)
            context, weights = self.attention_layer(query=rnn_output, value=out_encoder, mask=mask_source)
            # context shape (N_batch, 1, n_h)
            context = tf.squeeze(context, axis=1)  # shape (N_batch, n_h)
            out_concat = tf.concat([context, out_dropout4], axis=-1)  # shape (N_batch, n_h+n_h)
            out_attention = self.fc_layer0(out_concat)  # shape (N_batch, n_h)

            out = self.fc_layer1(out_attention)  # shape (N_batch, n_vocab)

            out = self.softmax_layer(out)

            max_idx = tf.math.argmax(out, axis=1)  # shape (N_batch, )

            # print('max_idx', max_idx)

            # 若出现结束标记位, 则置此序列的状态为 '解码结束' (True)
            # 注意这里是 '或', 也就是只要出现一次结束标记位之后 done 数组中表示此序列的位一直为 True
            done = done | (max_idx == self._end_target)
            # 若序列的状态被置为 '解码结束', 则 后面的时间步都填充 null 元素
            batch_token = tf.where(done, tf.constant(self._null_target, dtype=tf.int64), max_idx)

            outs = outs.write(t, batch_token)  # shape (target_length, N_batch)

            if tf.reduce_all(done):
                break

        outputs = tf.transpose(outs.stack(), perm=[1, 0])  # 单词标号序列 shape (N_batch, target_length)

        decode_seq = self.tokenizer_target.detokenize(outputs)  # 解码后的单词序列 shape (N_batch, target_length)

        decode_text = tf.strings.reduce_join(decode_seq, axis=1, separator=' ')  # 单词序列 join 成句子

        return outputs, decode_text


class TrainModel(Model):

    def __init__(self, n_embedding, n_h, target_length,
                 dropout_rates,
                 n_vocab_source, n_vocab_target,
                 _null_source
                 ):

        super(TrainModel, self).__init__(self)

        # source 中代表 null 的标号
        self._null_source = _null_source

        # 建立编码器和解码器
        self.encoder = Encoder(n_embedding=n_embedding, n_h=n_h, n_vocab=n_vocab_source, dropout_rates=dropout_rates)

        self.decoder = TrianDecoder(n_embedding=n_embedding, n_h=n_h, n_vocab=n_vocab_target,
                                                target_length=target_length, dropout_rates=dropout_rates)

    # 调试的时候去掉  @tf.function 装饰器
    @tf.function(input_signature=[
        (tf.TensorSpec(dtype=tf.int64, shape=[None, None]),
        tf.TensorSpec(dtype=tf.int64, shape=[None, None]))
    ])
    def call(self, inputs):
        """
        要使用 Model 自带的 fit 函数, call() 只能有 1个参数

        :param inputs_tuple:
        :return:
        """
        batch_source, batch_target_in = inputs

        # batch_source shape (N_batch, source_length)
        # batch_target_in shape (N_batch, target_length)

        mask_source = (batch_source != self._null_source)

        layer_state_list, out_encoder = self.encoder(batch_source=batch_source)

        outputs_prob = self.decoder(batch_target_in=batch_target_in, mask_source=mask_source,
                                          layer_state_list=layer_state_list, out_encoder=out_encoder)

        return outputs_prob

class InferModel(Model):

    def __init__(self, train_encoder, train_decoder, reverse_source, _null_source,  _start_target, _end_target, _null_target, tokenizer_source, tokenizer_target):

        super(InferModel, self).__init__()

        self.reverse_source = reverse_source

        self._null_source = _null_source

        self._start_target = _start_target
        self._end_target = _end_target
        self._null_target = _null_target

        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target

        self.encoder = train_encoder

        self.infer_decoder = InferDecoder(train_decoder_obj=train_decoder,
                                          _start_target=_start_target, _end_target=_end_target,
                                          _null_target=_null_target, tokenizer_target=tokenizer_target)

    def _preprocess_infer(self, batch_data):
        """
        对数据集的 一个批次的数据的预处理

        :param batch_data:
        :return:
        """

        batch_source = batch_data

        batch_source_vector = self.tokenizer_source.tokenize(batch_source).to_tensor()

        if self.reverse_source:
            batch_source_vector = batch_source_vector[:, ::-1]

        return batch_source_vector

    # 1.调试的时候去掉  @tf.function 装饰器
    # 2.input_signature 规定了函数参数的类型, 在重复收到规定类型的输入不会重新构建计算图
    #   shape=[None, None] 表示张量的维度是2维, None 表示可以取任意值
    #   shape=None 表示标量
    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.int64, shape=[None, None]), tf.TensorSpec(dtype=tf.int32, shape=None)])
    def call(self, batch_source, target_length):
        """

        :param batch_source: shape (N_batch, source_length)
        :param target_length:
        :return:
        """
        training = False

        mask_source = (batch_source != self._null_source)

        layer_state_list, out_encoder = self.encoder(batch_source=batch_source, training=training)

        preds, decode_text = self.infer_decoder(target_length=target_length, mask_source=mask_source,
                                             layer_state_list=layer_state_list, out_encoder=out_encoder,
                                             training=training)

        return preds, decode_text


    def test_step(self, data, target_length=None):

        batch_source = self._preprocess_infer(data)

        if target_length is None:
            target_length = tf.shape(batch_source)[1] + 50  # 源句子的长度决定了推理出的目标句子的长度

        _, decode_text = self(batch_source, target_length)

        return decode_text

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

            decode_text = self.test_step(batch_data, target_length)

            for sentence in decode_text:
                seq_list.append(sentence)

        return seq_list



class MaskedLoss(tf.keras.losses.Loss):

    def __init__(self, _null_target):

        # self.name = 'masked_loss'
        self._null_target = _null_target

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction='none')

    def call(self, y_true, y_pred):

        # Calculate the loss for each item in the batch.
        loss = self.loss(y_true, y_pred)

        # Mask off the losses on padding.
        mask = (y_true != self._null_target)

        mask = tf.cast(mask, dtype=loss.dtype)

        loss *= mask

        # Return the total.
        return tf.reduce_sum(loss)

class Test:

    def test_TrainModel(self):

        n_embedding = 32
        n_h = 32
        _null_source = 0

        target_length = 5
        dropout_rates = [0.2, 0.2, 0.2, 0.2, 0.2]

        n_vocab_source = 50
        n_vocab_target = 50

        model = TrainModel(n_embedding, n_h, target_length,
                 dropout_rates,
                 n_vocab_source, n_vocab_target, _null_source)

        N_batch = 4
        source_length = 6

        batch_source = np.random.randint(10, size=(N_batch, source_length))
        batch_target_in = np.random.randint(10, size=(N_batch, target_length))

        inputs_tuple = (batch_source, batch_target_in)

        outputs_prob = model.call(inputs_tuple)

        print(tf.shape(outputs_prob))


if __name__ == '__main__':

    test = Test()

    test.test_TrainModel()
