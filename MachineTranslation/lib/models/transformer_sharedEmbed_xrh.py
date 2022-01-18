#!/usr/bin/python
# -*- coding: UTF-8 -*-

#  适用于 tensorflow >= 2.0, keras 被直接集成到 tensorflow 的内部

from tensorflow.keras.layers import Embedding, \
    Dropout, Activation, LayerNormalization

from tensorflow.keras.models import Model

from tqdm import tqdm

import time


from lib.models.position_encode_xrh import *

from lib.layers.transformer_layer_xrh import *

from lib.layers.embedding_layer_xrh import *

from lib.utils.mask_xrh import *

from lib.models.optimizer_xrh import *

class TransformerSharedEmbed:
    """

    共享 Embedding 的 Transformer 模型

    1.编码器的 Embedding, 解码器的 Embedding , 和解码器的输出层共享权重矩阵 V


    Author: xrh
    Date: 2022-1-5

    ref:
    1. Attention Is All You Need
    2. https://tensorflow.google.cn/text/tutorials/transformer

    """

    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rates,
                  label_smoothing, warmup_steps,
                 maximum_position_source, maximum_position_target,
                 fixed_seq_length,
                 n_vocab_source, n_vocab_target,
                 _null_source, _start_target, _null_target, _end_target,
                 tokenizer_source, tokenizer_target,
                 build_mode='Eager',
                 ):
        """

        :param num_layers: 堆叠的编码器的层数
        :param d_model: 模型整体的隐藏层的维度
        :param num_heads: 并行注意力层的个数(头数)
        :param dff: Position-wise Feed-Forward 的中间层的维度
        :param dropout_rates: dropout 的弃置率
        :param label_smoothing: 标签平滑
        :param warmup_steps: 优化器学习率的预热步骤
        :param maximum_position_source: 源句子的可能最大长度
        :param maximum_position_target: 目标句子的可能最大长度
        :param fixed_seq_length: 将原始序列标记化为固定的长度
        :param n_vocab_source: 源语言的词表大小
        :param n_vocab_target: 目标语言的词表大小
        :param _null_source: 源序列的填充标号
        :param _start_target: 目标序列的开始标号
        :param _null_target: 目标序列的填充标号
        :param _end_target: 目标序列的结束标号
        :param tokenizer_source: 源语言的分词器
        :param tokenizer_target: 目标语言的分词器
        :param build_mode: 建立训练计算图的方式
        """

        super().__init__()

        print('model architecture param:')
        print('num_layers:{}, d_model:{}, num_heads:{}, dff:{}, n_vocab_source:{}, n_vocab_target:{}'.format(num_layers, d_model, num_heads,
                                                                                    dff, n_vocab_source, n_vocab_target))
        print('-------------------------')

        # 最大的序列长度
        self.fixed_seq_length = fixed_seq_length

        # 训练数据中源序列的长度
        self.source_length = self.fixed_seq_length

        # 训练数据中目标序列的长度
        self.target_length = self.fixed_seq_length - 1


        if build_mode == 'Eager':

            self.model_train = TrainModel(
                num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, dropout_rates=dropout_rates,
                label_smoothing=label_smoothing, warmup_steps=warmup_steps,
                n_vocab_source=n_vocab_source, n_vocab_target=n_vocab_target,
                _null_source=_null_source, _null_target=_null_target,
                maximum_position_source=maximum_position_source, maximum_position_target=maximum_position_target,
                )

            self.model_infer = InferModel(model_train=self.model_train,
                                          _null_source=_null_source,
                                          _start_target=_start_target, _end_target=_end_target, _null_target=_null_target,
                                          tokenizer_source=tokenizer_source, tokenizer_target=tokenizer_target)

        else:
            raise Exception("Invalid param value, build_mode= ", build_mode)




class Encoder(Layer):
    """
    将多层的编码器层进行堆叠

    """

    def __init__(self, num_layers, d_model, num_heads, dff, shared_embed_layer,
                 pos_encoding, dropout_rates):
        """

        :param num_layers: 堆叠的编码器的层数
        :param d_model: 模型整体的隐藏层的维度
        :param num_heads: 并行注意力层的个数(头数)
        :param dff: Position-wise Feed-Forward Networks 的中间层的维度
        :param shared_embed_layer: 共享的 embedding 层
        :param pos_encoding: 位置编码张量(包括所有位置)
        :param dropout_rates: dropout 的弃置率
        """
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = shared_embed_layer
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
            x = self.enc_layers[i](x=x, training=training, padding_mask=padding_mask)

        return x  # shape (N_batch, source_length, d_model)


class TrainDecoder(Layer):
    """
    将多层解码器层进行堆叠

    """

    def __init__(self, num_layers, d_model, num_heads, dff, shared_embed_layer,
                 pos_encoding, dropout_rates):
        """

        :param num_layers: 堆叠的编码器的层数
        :param d_model: 模型整体的隐藏层的维度
        :param num_heads: 并行注意力层的个数(头数)
        :param dff: Position-wise Feed-Forward Networks 的中间层的维度
        :param shared_embed_layer: 共享的 embedding 层
        :param pos_encoding: 位置编码张量(包括所有位置)
        :param dropout_rates: dropout 的弃置率
        """

        super(TrainDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = shared_embed_layer
        self.pos_encoding = pos_encoding

        self.dec_layer_list = [DecoderLayer(d_model, num_heads, dff, dropout_rates)
                               for _ in range(num_layers)]
        self.dropout = Dropout(dropout_rates[-1])

        self.fc = shared_embed_layer

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
            x, block1, block2 = self.dec_layer_list[i](x=x, encoder_output=encoder_output, training=training,
                                                       look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)

            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        # x shape (N_batch, target_seq_len, d_model)

        out = self.fc.call_liner(x)  # out shape (N_batch, target_seq_len, n_vocab_target)

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

        out_list = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True, clear_after_read=False)
        out_list = out_list.write(0, start_token)  # shape (1, N_batch)

        done = tf.zeros((N_batch, ), dtype=tf.bool)  # 标记序列的解码可以结束

        for t in tf.range(target_length):  # 使用 tf.range 会触发 tf.autograph 将循环也构成计算图的一部分

            batch_tokens = tf.transpose(out_list.stack())  # shape (N_batch, t+1)

            look_ahead_mask = create_target_look_ahead_mask(batch_tokens, self._null_target)

            out_embed = self.embedding(batch_tokens)  # shape (N_batch, t+1, d_model)

            out_embed *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

            # 加入位置编码信息
            # 对 pos_encoding 根据序列长度进行截取
            out_embed += self.pos_encoding[:, :t + 1, :]  # shape (N_batch, t+1, d_model)

            x = self.dropout(out_embed, training=training)

            for i in range(self.num_layers):
                x, block1, block2 = self.dec_layer_list[i](x=x, encoder_output=encoder_output, training=training,
                                                           look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)

            x = self.fc.call_liner(x)  # shape (N_batch, t+1, n_vocab_target)

            # 最后一个时间步的输出
            out = x[:, -1, :]  # (N_batch, vocab_size)

            max_idx = tf.math.argmax(out, axis=-1)  # shape (N_batch, )

            # print('max_idx', max_idx)

            # 若出现结束标记位, 则置此序列的状态为 '解码结束' (True)
            # 注意这里是 '或', 也就是只要出现一次结束标记位之后 done 数组中表示此序列的位一直为 True
            done = done | (max_idx == self._end_target)  # shape (N_batch, )
            # 若序列的状态被置为 '解码结束', 则 后面的时间步都填充 null 元素
            batch_token_t = tf.where(condition=done, x=tf.constant(self._null_target, dtype=tf.int64), y=max_idx)  # shape (N_batch, )

            out_list = out_list.write(t+1, batch_token_t)  # shape (t+2, N_batch)

            if tf.reduce_all(done):
                break

        outputs = tf.transpose(out_list.stack(), perm=[1, 0])  # 单词标号序列 shape (N_batch, target_length)

        outputs = outputs[:, 1:]  # 第1个时间步是开始标记可以忽略

        vectors = self.tokenizer_target.detokenize(outputs)
        text = tf.strings.reduce_join(vectors, separator=' ', axis=-1)

        return outputs, text


class TrainModel(Model):

    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rates,
                 label_smoothing, warmup_steps,
                 maximum_position_source, maximum_position_target,
                 n_vocab_source, n_vocab_target,
                 _null_source, _null_target,
                 ):
        """

        :param num_layers: 堆叠的编码器的层数
        :param d_model: 模型整体的隐藏层的维度
        :param num_heads: 并行注意力层的个数(头数)
        :param dff: Position-wise Feed-Forward Networks 的中间层的维度
        :param dropout_rates: dropout 的弃置率
        :param label_smoothing: 标签平滑
        :param warmup_steps: 优化器学习率的预热步骤
        :param maximum_position_source: 源句子的可能最大长度
        :param maximum_position_target: 目标句子的可能最大长度
        :param n_vocab_source: 源语言的词表大小
        :param n_vocab_target: 目标语言的词表大小
        :param _null_source: 源序列的填充标号
        :param _null_target: 目标序列的填充标号


        """

        super().__init__()

        PE_source = SinusoidalPE(maximum_position=maximum_position_source, d_model=d_model)
        PE_target = SinusoidalPE(maximum_position=maximum_position_target, d_model=d_model)

        self.label_smoothing = label_smoothing

        self._null_source = _null_source
        self._null_target = _null_target

        self.n_vocab_target = n_vocab_target

        # self.tokenizer_source = tokenizer_source
        # self.tokenizer_target = tokenizer_target

        self.shared_embed_layer = SharedEmbedding(n_h=d_model, n_vocab=n_vocab_target)

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               self.shared_embed_layer, PE_source.pos_encoding, dropout_rates)

        self.decoder = TrainDecoder(num_layers, d_model, num_heads, dff,
                                    self.shared_embed_layer, PE_target.pos_encoding, dropout_rates)

        # 损失函数对象
        if label_smoothing == 0:  # 不开启 label_smoothing
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
        else:
            self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction='none', label_smoothing=label_smoothing)

        self.loss_tracker = tf.keras.metrics.Mean(name='train_loss')
        self.accuracy_metric = tf.keras.metrics.Mean(name='train_accuracy')

        learning_rate = WarmupSchedule(d_model, warmup_steps)
        # learning_rate = 3e-4
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                             epsilon=1e-9)

    def get_config(self):
        config = super().get_config().copy()

        config.update({
            'shared_embed_layer': self.shared_embed_layer,
        })
        return config


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

        if self.label_smoothing == 0: # 不开启 label_smoothing
            y_true_dense = y_true
        else:
            y_true_dense = tf.argmax(y_true, axis=-1)

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

        if self.label_smoothing == 0:  # 不开启 label_smoothing
            y_true_dense = y_true
        else:
            y_true_dense = tf.argmax(y_true, axis=-1)

        accuracies = tf.equal(y_true_dense, tf.argmax(y_pred, axis=-1))

        mask = (y_true_dense != self._null_target)

        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)

        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    def _preprocess_train(self, source, target):
        """
        对数据集的 一个批次的数据的预处理

        :param source:
        :param target:
        :return:
        """

        source_vector = source  # shape (N_batch, source_length)
        target_vector = target  # shape (N_batch, target_length)

        target_in = target_vector[:, :-1]
        target_out = target_vector[:, 1:]

        if self.label_smoothing != 0:  # 开启 label_smoothing

            target_out = tf.one_hot(indices=target_out, depth=self.n_vocab_target,
                                           on_value=1, off_value=0, dtype=tf.int64, axis=-1)

        return (source_vector, target_in), target_out

    # @tf.function 将 train_step 编译为 计算图，以便更快地执行;
    # input_signature 指定了 输入张量的 shape, 可以避免重复建立计算图,
    # 若不指定 shape, 一个 epoch 的最后一批数据的 N_batch 与之前不同, 会触发耗时的 trace 操作
    signature = [
        (
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.int64)
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

        # trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
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
        checkpoint_path = "models/checkpoints"

        ckpt = tf.train.Checkpoint(model_train=self)

        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')


        for epoch in range(epochs):
            start = time.time()

            self.loss_tracker.reset_states()
            self.accuracy_metric.reset_states()

            for (batch, batch_data) in enumerate(x):

                res_dict = self.train_step(batch_data)

                if batch % 50 == 0:
                    print(
                        f'Epoch {epoch + 1} Batch {batch} Loss {res_dict["loss"]:.4f} Accuracy {res_dict["accuracy"]:.4f}')

            # if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

            print(f'Epoch {epoch + 1} Loss {res_dict["loss"]:.4f} Accuracy {res_dict["accuracy"]:.4f}')

            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

    @tf.function(input_signature=signature)
    def test_step(self, data):

        source, target = data

        # tf.print(source.numpy())

        (batch_source_vector, batch_target_in), batch_target_out = self._preprocess_train(source, target)

        # tf.print(source.numpy())

        predictions, _ = self([batch_source_vector, batch_target_in],
                                     training=False)

        loss = self._mask_loss_function(batch_target_out, predictions)

        self.loss_tracker.update_state(loss)
        self.accuracy_metric.update_state(self._mask_accuracy_function(batch_target_out, predictions))

        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy_metric.result()}


    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.accuracy_metric]


class InferModel(Model):

    def __init__(self, model_train, _null_source,  _start_target, _end_target, _null_target, tokenizer_source, tokenizer_target):

        super(InferModel, self).__init__()

        self._null_source = _null_source

        self._start_target = _start_target
        self._end_target = _end_target
        self._null_target = _null_target

        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target

        self.encoder = model_train.encoder

        self.infer_decoder = InferDecoder(train_decoder_obj=model_train.decoder,
                                          _start_target=_start_target, _end_target=_end_target,
                                          _null_target=_null_target,
                                          tokenizer_target=tokenizer_target)

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
    def call(self, batch_source, target_length):
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
            num_layers=num_layers, d_model=n_h, num_heads=num_heads, dff=2048, dropout_rates=dropout_rates,
            label_smoothing=0.1, warmup_steps=4000,
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

        model.save_weights('tmp/checkpoint')

if __name__ == '__main__':
    test = Test()

    test.test_TrainModel()
