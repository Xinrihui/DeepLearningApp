#!/usr/bin/python
# -*- coding: UTF-8 -*-

#  适用于 tensorflow >= 2.0, keras 被直接集成到 tensorflow 的内部
#  ref: https://keras.io/about/

from tensorflow.keras.layers import Layer, Input, LSTM, TimeDistributed, Bidirectional, Dense, Lambda, Embedding, Dropout, \
    Concatenate, RepeatVector
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import tensorflow.keras as keras

from tensorflow.keras.models import Model

from lib.evaluate_xrh import *
from lib.tf_data_tokenize_xrh import *

import time
import configparser
import json


class MachineTranslation:
    """

    基于 seq2seq 的神经机器翻译模型 (v2-integrated)

    1. 解码采用一体化模型 (integrated model)的方式, 即建立推理计算图, 将每一步的解码都在计算图中完成
    2. 实现了基于 tf.data 的数据预处理 pipline, 使用 TextVectorization 和 StringLookup 做句子的向量化和反向量化
    3. 采用动态图构建模型, 并从头编写训练循环;
    4. 在配置中心中维护超参数

    Author: xrh
    Date: 2021-11-20

    ref:
    1. Sequence to Sequence Learning with Neural Networks
    2. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation
    3.

    """

    def __init__(self,
                 current_config,
                 vocab_source, vocab_target,
                 tokenizer_source=None, tokenizer_target=None,
                 use_pretrain=False,
                 ):
        """
        模型初始化

        :param vocab_source: 源语言的词典对象
        :param vocab_target: 目标语言的词典对象

        :param tokenizer_source:
        :param tokenizer_target:

        :param use_pretrain: 使用训练好的模型

        """

        self.current_config = current_config

        self.n_h = int(current_config['n_h'])
        self.n_embedding = int(current_config['n_embedding'])

        self.target_length = int(current_config['max_seq_length'])

        self.dropout_rates = json.loads(current_config['dropout_rates'])

        _null_str = current_config['_null_str']
        _start_str = current_config['_start_str']
        _end_str = current_config['_end_str']
        _unk_str = current_config['_unk_str']

        self.vocab_source = vocab_source
        self.vocab_target = vocab_target

        # 源语言的词表大小
        self.n_vocab_source = self.vocab_source.n_vocab

        # 目标语言的词表大小
        self.n_vocab_target = self.vocab_target.n_vocab


        print('model architecture param:')
        print('n_h:{}, n_embedding:{}, n_vocab_source:{}, n_vocab_target:{}'.format(self.n_h, self.n_embedding, self.n_vocab_source,
                                                                                    self.n_vocab_target))
        print('-------------------------')

        self._null = int(self.vocab_source.map_word_to_id(_null_str))  # 空
        self._start = int(self.vocab_source.map_word_to_id(_start_str))  # 句子的开始
        self._end = int(self.vocab_source.map_word_to_id(_end_str))  # 句子的结束
        self._unk = int(self.vocab_source.map_word_to_id(_unk_str))  # 未登录词

        # vocab_source 和 vocab_target 的标号不同
        self._null_target = int(self.vocab_target.map_word_to_id(_null_str))  # 空
        self._start_target = int(self.vocab_target.map_word_to_id(_start_str))  # 句子的开始
        self._end_target = int(self.vocab_target.map_word_to_id(_end_str))  # 句子的结束
        self._unk_target = int(self.vocab_target.map_word_to_id(_unk_str))  # 未登录词

        self.model_path = current_config['model_path']

        # 构建模型
        self.model = Seq2seqModel(n_embedding=self.n_embedding, n_h=self.n_h, target_length=self.target_length,
                                  n_vocab_source=self.n_vocab_source, n_vocab_target=self.n_vocab_target,
                                  tokenizer_source=tokenizer_source, tokenizer_target=tokenizer_target,
                                  _start_target=self._start_target, _null_target=self._null_target,
                                  dropout_rates=self.dropout_rates)


        if use_pretrain:  # 载入训练好的模型

            # self.model = tf.saved_model.load(self.model_path)

            self.model.load_weights(self.model_path)


    def _shuffle_dataset(self, dataset, buffer_size, batch_size):
        """
        数据混洗(shuffle), 分批(batch)和预取(prefetch)

        :param dataset:
        :return:
        """

        # 解开 batch, 数据集的粒度变为行
        dataset = dataset.unbatch()

        # tf.data 的数据混洗,分批和预取,
        # shuffle 后不能进行任何的 map 操作, 因为会改变 batch 中的数据行的组合
        dataset = dataset.shuffle(buffer_size).batch(batch_size)

        final_dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return final_dataset

    def fit_tf_data(self, train_dataset, valid_dataset, valid_source_target_dict, epoch_num,
                    batch_size, buffer_size,
                    ):
        """
        使用内置方法训练模型

        :param train_dataset: 训练数据生成器
        :param valid_dataset: 验证数据生成器
        :param valid_source_target_dict: 验证数据的字典, 用于计算 bleu

        :param epoch_num: 模型训练的 epoch 个数,  一般训练集所有的样本模型都见过一遍才算一个 epoch
        :param batch_size: 选择 min-Batch梯度下降时, 每一次输入模型的样本个数
        :param buffer_size:  shuffle 的窗口大小

        :return:
        """

        train_dataset_prefetch = self._shuffle_dataset(train_dataset, buffer_size, batch_size)
        valid_dataset_prefetch = self._shuffle_dataset(valid_dataset, buffer_size, batch_size)

        checkpoint_models_path = self.current_config['checkpoint_models_path']

        # Callbacks
        # 在根目录下运行 tensorboard --logdir ./logs
        tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True,
                                                   write_images=True)

        # 早停: 在验证集上, 损失经过 patience 次的迭代后, 仍然没有下降则暂停训练
        early_stop = EarlyStopping('val_loss', patience=5)

        model_checkpoint_with_eval = CheckoutCallback(current_config=self.current_config,
                                                      model=self.model,
                                                      vocab_obj=self.vocab_target, valid_source_target_dict=valid_source_target_dict,
                                                      checkpoint_models_path=checkpoint_models_path)

        # Final callbacks
        # callbacks = [model_checkpoint_with_eval, early_stop, tensor_board]

        history = self.model.fit(
            epoch_num=epoch_num, train_dataset=train_dataset_prefetch, valid_dataset=valid_dataset_prefetch,
            callback_obj=model_checkpoint_with_eval
            )

        # 将训练好的模型保存到文件
        # self.model.save(self.model_path)



    def inference(self, batch_source_dataset, target_length):
        """
        使用训练好的模型进行推理

        :param batch_source_dataset:

        :return:
        """

        # batch_source_dataset shape (N_batch, encoder_length)

        preds = self.model.predict(batch_source_dataset, target_length)

        decode_result = self.vocab_target.map_id_to_word(preds)

        decode_result = tf.strings.reduce_join(decode_result, axis=1,
                                               separator=' ')

        candidates = [sentence.numpy().decode('utf-8').strip() for sentence in decode_result]

        return candidates

class CheckoutCallback:
    """
    回调函数, 实现在每一次 epoch 后 checkout 训练好的模型,
    并且计算在验证集上的 bleu 分数

    """

    def __init__(self, current_config,
                 model, vocab_obj, valid_source_target_dict,
                 checkpoint_models_path):

        self.model = model
        self.vocab_obj = vocab_obj

        self.batch_source_dataset, self.references = self.prepare_valid_data(batch_size=int(current_config['batch_size']),
                                                                           valid_source_target_dict=valid_source_target_dict)

        self.target_length = int(current_config['max_seq_length'])

        self.evaluate_obj = Evaluate(
            with_unk=True,
            _null_str=current_config['_null_str'],
            _start_str=current_config['_start_str'],
            _end_str=current_config['_end_str'],
            _unk_str=current_config['_unk_str'])

        self.checkpoint_models_path = checkpoint_models_path

    def prepare_valid_data(self, batch_size, valid_source_target_dict):
        """
        返回 图片的 embedding 向量 和 图片对应的 caption

        :param batch_size:
        :param valid_source_target_dict:
        :return:
        """

        source_list = list(valid_source_target_dict.keys())

        print('valid source seq num :{}'.format(len(source_list)))

        references = [valid_source_target_dict[source] for source in source_list]

        source_dataset = tf.data.Dataset.from_tensor_slices(source_list)

        batch_source_dataset = source_dataset.batch(batch_size)

        return batch_source_dataset, references

    def inference_bleu(self):
        """
        使用验证数据集进行推理, 并计算 bleu

        :return:
        """

        # batch_source_dataset shape (N_batch, encoder_length)

        preds = self.model.predict(self.batch_source_dataset, self.target_length)

        decode_result = self.vocab_obj.map_id_to_word(preds)

        decode_result = tf.strings.reduce_join(decode_result, axis=1,
                                               separator=' ')

        candidates = [sentence.numpy().decode('utf-8').strip() for sentence in decode_result]

        bleu_score, _ = self.evaluate_obj.evaluate_bleu(self.references, candidates)

        print()
        print('bleu_score:{}'.format(bleu_score))

    def on_epoch_end(self, epoch, logs=None):

        # checkout 模型
        fmt = os.path.join(self.checkpoint_models_path, 'model.%02d-%.4f')

        self.model.save_weights(fmt % (epoch, logs['val_loss']))  # 保存模型的参数

        # tf.saved_model.save(self.model, fmt % (epoch, logs['val_loss']))  # 保存整个模型

        # 计算 bleu 分数
        self.inference_bleu()


class Seq2seqModel(Model):
    """
    seq2seq 模型

    1. 解码采用一体化模型的方式, 即建立推理计算图, 将每一步的解码都在计算图中完成
    2. 采用动态图构建模型, 并从头编写训练循环

    Author: xrh
    Date: 2021-11-20

    """

    def __init__(self,  n_embedding, n_h, target_length,
                 n_vocab_source, n_vocab_target,
                 _start_target, _null_target,
                 tokenizer_source=None, tokenizer_target=None,
                 dropout_rates=(0.2, 0.2, 0.2)):

        super().__init__()

        # target 中代表 null 的标号
        self._start_target = _start_target

        # target 中代表 null 的标号
        self._null_target = _null_target

        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target

        # 建立编码器和解码器
        self.encoder = Encoder(n_embedding=n_embedding, n_h=n_h, n_vocab=n_vocab_source, dropout_rates=dropout_rates)

        self.train_decoder = TrianDecoder(n_embedding=n_embedding, n_h=n_h, n_vocab=n_vocab_target, target_length=target_length, dropout_rates=dropout_rates)

        self.infer_decoder = InferDecoder(train_decoder_obj=self.train_decoder, _start=self._start_target, target_length=target_length)

        # 损失对象
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')

        # 优化器
        self.optimizer = keras.optimizers.RMSprop()

        # 评价指标
        self.train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        self.val_acc_metric = keras.metrics.SparseCategoricalAccuracy()



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

        return mask, tf.reduce_mean(loss_)

    def _preprocess(self, batch_data):
        """
        对数据集的 一个批次的数据的预处理

        :param batch_data:
        :return:
        """

        batch_source, batch_target = batch_data

        batch_source_vector = self.tokenizer_source(batch_source).to_tensor()

        batch_target_vector = self.tokenizer_target(batch_target).to_tensor()

        batch_target_in = batch_target_vector[:, :-1]
        batch_target_out = batch_target_vector[:, 1:]

        return (batch_source_vector, batch_target_in), batch_target_out

    # @tf.function 可以加速训练
    # @tf.function
    def _train_step(self, inputs):

        training = True

        (batch_source, batch_target_in), batch_target_out = inputs
        # batch_source  shape (N_batch, source_length)
        # batch_target_in shape (N_batch, target_length)
        # batch_target_out shape (N_batch, target_length)

        target_length = tf.shape(batch_target_in)[1]

        with tf.GradientTape() as tape:

            layer_state_list = self.encoder(batch_source=batch_source, training=training)

            probs = self.train_decoder(batch_target_in=batch_target_in,
                                       layer_state_list=layer_state_list, training=training)
            # probs shape (N_batch, target_length, n_vocab)

            target_mask, loss = self._mask_loss_function(batch_target_out, probs)

            # 求所有没有被填充的位置的损失的平均和
            # average_loss = loss / tf.reduce_sum(target_mask)

            average_loss = loss

        # Apply an optimization step
        variables = self.trainable_variables
        gradients = tape.gradient(average_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return average_loss, probs

    # @tf.function
    def _test_step(self, batch_source, target_length):

        # batch_source  shape (N_batch, source_length)

        training = False

        layer_state_list = self.encoder(batch_source=batch_source, training=training)

        probs, preds = self.infer_decoder(layer_state_list=layer_state_list,
                                   target_length=target_length, training=training)

        return probs, preds


    def fit(self, epoch_num, train_dataset, valid_dataset, callback_obj=None):

        logs_list = []  # 记录每一个 epoch 的日志

        for epoch in range(epoch_num):

            print('----------------------------------')
            print("Epoch %d/%d" % (epoch, epoch_num))
            start_time = time.time()

            logs = {}  # 当前 epoch 的日志

            epoch_train_loss = 0

            # 遍历训练数据集的 所有batch
            for batch_data in tqdm(train_dataset):

                features, labels = self._preprocess(batch_data)

                batch_loss, probs = self._train_step((features, labels))

                # print('iteration loss: %.2f' % (batch_loss,))

                epoch_train_loss += batch_loss

                # Update training metric.
                self.train_acc_metric.update_state(labels, probs)

            # 本次 epoch 的训练结束
            # print("\nEpoch: %d end" % (epoch,))

            train_acc = self.train_acc_metric.result()

            print("train loss: %.4f , train acc: %.4f" % (epoch_train_loss, float(train_acc)))

            # 重置验证指标
            self.train_acc_metric.reset_states()

            epoch_valid_loss = 0

            # 遍历验证数据集
            for batch_data in tqdm(valid_dataset):

                features, labels = self._preprocess(batch_data)

                (batch_source, batch_target_in) = features

                target_length = tf.shape(batch_target_in)[1]

                probs, _ = self._test_step(batch_source, target_length)

                target_mask, batch_loss = self._mask_loss_function(labels, probs)

                epoch_valid_loss += batch_loss

                # Update val metrics
                self.val_acc_metric.update_state(labels, probs)


            val_acc = self.val_acc_metric.result()
            self.val_acc_metric.reset_states()

            print("valid loss: %.4f , valid acc: %.4f" % (epoch_valid_loss, float(val_acc)))

            cost_time = time.time() - start_time

            print("Time taken: %.2fs" % cost_time)


            # 记录日志
            logs['epoch'] = epoch
            logs['train_loss'] = epoch_train_loss
            logs['train_acc'] = train_acc
            logs['val_loss'] = epoch_valid_loss
            logs['val_acc'] = val_acc
            logs['time'] = cost_time

            logs_list.append(logs)

            # callback
            if callback_obj is not None:
                callback_obj.on_epoch_end(epoch=epoch, logs=logs)

        return logs_list

    def predict(self, source_dataset, target_length):

        preds_list = []

        # 遍历数据集
        for batch_data in tqdm(source_dataset):

            batch_source = batch_data
            batch_source = self.tokenizer_source(batch_source).to_tensor()

            _, preds = self._test_step(batch_source, target_length)

            for pred in preds:
                preds_list.append(pred)

            # preds_list.append(list(preds.numpy()))

        # batch_num, N_batch, seq_length = tf.shape(preds_list)

        # res = tf.reshape(preds_list, [-1, target_length])

        return preds_list

class Encoder(Layer):
    """
    基于 LSTM 的编码器层

    """

    def __init__(self, n_embedding, n_h, n_vocab, dropout_rates=(0.2, 0.2, 0.2)):

        super(Encoder, self).__init__()

        self.embedding_layer = Embedding(n_vocab, n_embedding)

        self.lstm_layer0 = LSTM(n_h, return_sequences=True, return_state=True, name='lstm_layer1')
        # self.dropout_layer1 = Dropout(dropout_rates[0], name='dropout1')  # 神经元有 0.1 的概率被弃置

        # self.lstm_layer2 = LSTM(n_h, return_sequences=True, return_state=True, name='lstm_layer1')
        # self.dropout_layer2 = Dropout(dropout_rates[0], name='dropout1')  # 神经元有 0.1 的概率被弃置


    # def get_config(self):
    #     config = super().get_config().copy()
    #     config.update({
    #         'embedding_layer': self.embedding_layer,
    #         'lstm_layer0': self.lstm_layer0
    #
    #     })
    #     return config

    def call(self, batch_source, training=True):

        # batch_source shape (N_batch, source_length)

        source_embedding = self.embedding_layer(inputs=batch_source)  # shape (N_batch, encoder_length, n_embedding)

        layer_state_list = []

        out_lstm0, state_h0, state_c0 = self.lstm_layer0(
            inputs=source_embedding)  # out_lstm0 shape : (N_batch, source_length, n_h)

        layer_state_list.append((state_h0, state_c0))

        return layer_state_list


class TrianDecoder(Layer):
    """
    训练模式下的基于 LSTM 的解码器层

    """

    def __init__(self, n_embedding, n_h, n_vocab, target_length, dropout_rates=(0.2, 0.2, 0.2)):

        super(TrianDecoder, self).__init__()

        self.target_length = target_length

        self.embedding_layer = Embedding(n_vocab, n_embedding)

        self.lstm_layer0 = LSTM(n_h, return_sequences=True, return_state=True, name='lstm_layer1')

        self.out_dropout_layer = Dropout(dropout_rates[-1])

        self.fc_layer = Dense(n_vocab, activation='softmax')


    # def get_config(self):
    #
    #     config = super().get_config().copy()
    #
    #     config.update({
    #         'target_length': self.target_length,
    #         'embedding_layer': self.embedding_layer,
    #         'lstm_layer0': self.lstm_layer0,
    #         'out_dropout_layer': self.out_dropout_layer,
    #         'fc_layer': self.fc_layer,
    #     })
    #     return config

    def call(self, batch_target_in, layer_state_list, training=True):

        # batch_target_in shape (N_batch, target_length)

        batch_target_embbeding = self.embedding_layer(inputs=batch_target_in)
        # shape (N_batch, target_length, n_embedding)

        h0 = layer_state_list[0][0]  # shape: (N_batch, n_h)
        c0 = layer_state_list[0][1]  # shape: (N_batch, n_h)

        outs_prob = []

        target_length = tf.shape(batch_target_in)[1]  # 使用动态图时可以支持

        for t in tf.range(target_length):

            batch_token_embbeding = tf.expand_dims(batch_target_embbeding[:, t, :], axis=1)
            # Teacher Forcing: 每一个时间步的输入为真实的标签值而不是上一步预测的结果
            # batch_token_embbeding shape (N_batch, 1, n_embedding)

            context = batch_token_embbeding

            out_lstm0, h0, c0 = self.lstm_layer0(inputs=context, initial_state=[h0, c0])  # 输入 context 只有1个时间步

            h_dropout = self.out_dropout_layer(h0, training=training)

            out = self.fc_layer(h_dropout)  # shape (N_batch, n_vocab)

            outs_prob.append(out)  # shape (target_length, N_batch, n_vocab)

        outputs_prob = tf.transpose(outs_prob, perm=[1, 0, 2])  # shape (N_batch, target_length, n_vocab)

        return outputs_prob


class InferDecoder(Layer):
    """
    推理模式下的基于 LSTM 的解码器层

    """

    def __init__(self, train_decoder_obj, target_length, _start):

        super(InferDecoder, self).__init__()

        self.train_decoder_obj = train_decoder_obj
        self._start = _start

        self.target_length = target_length

        self.embedding_layer = self.train_decoder_obj.embedding_layer

        self.lstm_layer0 = self.train_decoder_obj.lstm_layer0
        self.out_dropout_layer = self.train_decoder_obj.out_dropout_layer

        self.fc_layer = self.train_decoder_obj.fc_layer

    # def get_config(self):
    #     config = super().get_config().copy()
    #     config.update({
    #
    #         'train_decoder_obj': self.train_decoder_obj,
    #         '_start': self._start,
    #         'target_length': self.target_length,
    #         'embedding_layer': self.embedding_layer,
    #         'lstm_layer0': self.lstm_layer0,
    #         'out_dropout_layer': self.out_dropout_layer,
    #         'fc_layer': self.fc_layer,
    #     })
    #     return config

    def call(self, layer_state_list, target_length, training=False):
        """

        :param layer_state_list:
        :param target_length:
        :param training:
        :return: outputs_prob - shape (N_batch, target_length, n_vocab) 概率形式的推理结果
                outputs - shape (N_batch, target_length) 标号形式的推理结果
        """

        h0 = layer_state_list[0][0]  # shape: (N_batch, n_h)
        c0 = layer_state_list[0][1]  # shape: (N_batch, n_h)

        N_batch = tf.shape(h0)[0]
        batch_token = tf.ones((N_batch, 1)) * self._start  # (N_batch, 1)

        outs_prob = []

        outs = []

        for t in tf.range(target_length):

            batch_token_embbeding = self.embedding_layer(batch_token)  # shape (N_batch, 1, n_embedding)

            context = batch_token_embbeding

            out_lstm0, h0, c0 = self.lstm_layer0(inputs=context, initial_state=[h0, c0])  # 输入 context 只有1个时间步

            h_dropout = self.out_dropout_layer(h0, training=training)

            out = self.fc_layer(h_dropout)  # shape (N_batch, n_vocab)

            max_idx = tf.math.argmax(out, axis=1)  # shape (N_batch, )

            # print('max_idx', max_idx)

            batch_token = tf.expand_dims(max_idx, axis=1)  # shape (N_batch, 1)

            outs_prob.append(out)  # shape (target_length, N_batch, n_vocab)

            outs.append(max_idx)  # shape (target_length, N_batch)

        outputs_prob = tf.transpose(outs_prob, perm=[1, 0, 2])  # shape (N_batch, target_length, n_vocab)

        outputs = tf.transpose(outs, perm=[1, 0])  # shape (N_batch, target_length)

        return outputs_prob, outputs


class Test_WMT14_Eng_Ge_Dataset:

    def test_training(self, config_path='lib/config.ini', tag='DEFAULT'):

        # 0. 读取配置文件

        config = configparser.ConfigParser()
        config.read(config_path, 'utf-8')
        current_config = config[tag]

        print('current tag:{}'.format(tag))

        # 1. 数据集的预处理, 运行 tf_data_tokenize_xrh.py 中的 DataPreprocess -> do_mian()
        dataset_obj = WMT14_Eng_Ge_Dataset(base_dir=current_config['base_dir'],
                                           cache_data_folder=current_config['cache_data_folder'], mode='train')


        # 2. 训练模型

        trainer = MachineTranslation(
            current_config=current_config,
            vocab_source=dataset_obj.vocab_source,
            vocab_target=dataset_obj.vocab_target,
            tokenizer_source=dataset_obj.tokenizer_source, tokenizer_target=dataset_obj.tokenizer_target,
            use_pretrain=False
        )
        # use_pretrain=True: 在已有的模型参数基础上, 进行更进一步的训练

        batch_size = int(current_config['batch_size'])
        buffer_size = int(current_config['buffer_size'])
        epoch_num = int(current_config['epoch_num'])

        trainer.fit_tf_data(train_dataset=dataset_obj.train_dataset, valid_dataset=dataset_obj.valid_dataset,
                            valid_source_target_dict=dataset_obj.valid_source_target_dict,
                            epoch_num=epoch_num, batch_size=batch_size, buffer_size=buffer_size)

    def test_evaluating(self, config_path='lib/config.ini', tag='DEFAULT'):

        # 0. 读取配置文件
        config = configparser.ConfigParser()
        config.read(config_path, 'utf-8')
        current_config = config[tag]

        # 1. 数据集的预处理, 运行 tf_data_tokenize_xrh.py 中的 DataPreprocess -> do_mian()
        dataset_obj = WMT14_Eng_Ge_Dataset(base_dir=current_config['base_dir'],
                                           cache_data_folder=current_config['cache_data_folder'], mode='infer')

        batch_size = int(current_config['batch_size'])

        # 2.模型推理

        infer = MachineTranslation(
            current_config=current_config,
            vocab_source=dataset_obj.vocab_source,
            vocab_target=dataset_obj.vocab_target,
            tokenizer_source=dataset_obj.tokenizer_source, tokenizer_target=dataset_obj.tokenizer_target,
            use_pretrain=True
        )

        source_list = list(dataset_obj.test_source_target_dict.keys())

        print('valid source seq num :{}'.format(len(source_list)))

        references = [dataset_obj.test_source_target_dict[source] for source in source_list]

        source_dataset = tf.data.Dataset.from_tensor_slices(source_list)

        batch_source_dataset = source_dataset.batch(batch_size)

        candidates = infer.inference(batch_source_dataset, target_length=int(current_config['test_max_seq_length']))

        print('\ncandidates:')
        for i in range(0, 10):
            print('[{}] {}'.format(i, candidates[i]))

        print('\nreferences:')
        for i in range(0, 10):
            print('[{}] {}'.format(i, references[i]))


        evaluate_obj = Evaluate(
            with_unk=True,
            _null_str=current_config['_null_str'],
            _start_str=current_config['_start_str'],
            _end_str=current_config['_end_str'],
            _unk_str=current_config['_unk_str'])


        bleu_score, _ = evaluate_obj.evaluate_bleu(references, candidates, bleu_N=4)

        print('bleu_score:{}'.format(bleu_score))


if __name__ == '__main__':
    test = Test_WMT14_Eng_Ge_Dataset()

    # TODO: 每次实验前
    #  1. 更改最终模型存放的路径
    #  2. 运行脚本  clean_training_cache_file.bat

    test.test_training(tag='TEST')

    # test.test_evaluating(tag='TEST')
