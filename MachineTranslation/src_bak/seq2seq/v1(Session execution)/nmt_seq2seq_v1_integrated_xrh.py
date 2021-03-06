#!/usr/bin/python
# -*- coding: UTF-8 -*-

#  适用于 tensorflow >= 2.0, keras 被直接集成到 tensorflow 的内部
#  ref: https://keras.io/about/

from tensorflow.keras.layers import Input, LSTM, TimeDistributed, Bidirectional, Dense, Lambda, Embedding, Dropout, \
    Concatenate, RepeatVector
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import tensorflow.keras as keras

import tensorflow.keras.backend as K

from tensorflow.keras.models import Model

from lib.build_dataset_xrh import *
from lib.evaluate_xrh import *
from lib.get_dataset_xrh import *

# from deprecated import deprecated


class MachineTranslation:
    """
    基于 LSTM + seq2seq 的神经机器翻译模型

    0. 采用静态图构建模型 (Session execution)
    1. 解码采用一体化模型的方式, 即建立推理计算图, 将每一步的解码都在计算图中完成
    2. 采用 稀疏的交叉熵损失函数, 避免了 标签 one-hot 化后的 OOM 问题
    3. 实现了基于 Sequence 的数据批量生成器

    Author: xrh
    Date: 2021-10-10

    ref:
    1. Sequence to Sequence Learning with Neural Networks
    2. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation
    3. https://keras-zh.readthedocs.io/examples/lstm_seq2seq/
    """

    def __init__(self, encoder_length, decoder_length,
                 n_h, n_embedding,
                 vocab_source, vocab_target,
                 _null_str='<NULL>',
                 _start_str='<START>',
                 _end_str='<END>',
                 _unk_str='<UNK>',
                 use_pretrain=False,
                 model_path='models/machine_translation_seq2seq.h5'):
        """
        模型初始化
        :param encoder_length: 编码器的序列长度
        :param decoder_length: 解码器的序列长度
        :param n_h: lstm 中隐藏状态的维度, n_h = 512
        :param n_embedding : 词向量维度, n_embedding= 512
        :param vocab_source: 源语言的词典对象
        :param vocab_target: 目标语言的词典对象
        :param  _null_str: 末尾填充的空
        :param  _start_str: 句子的开始
        :param  _end_str: 句子的结束
        :param  _unk_str: 未登录词
        :param use_pretrain: 使用训练好的模型
        :param model_path: 预训练模型的路径
        """
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length

        self.n_h = n_h
        self.n_embedding = n_embedding

        self.vocab_source = vocab_source
        self.vocab_target = vocab_target

        self._null = self.vocab_source.map_word_to_id(_null_str)  # 空
        self._start = self.vocab_source.map_word_to_id(_start_str)  # 句子的开始
        self._end = self.vocab_source.map_word_to_id(_end_str)  # 句子的结束
        self._unk_str = self.vocab_source.map_word_to_id(_unk_str)  # 未登录词

        self.model_path = model_path

        # 源语言的词表大小
        self.n_vocab_source = len(self.vocab_source.word_to_id)

        # 目标语言的词表大小
        self.n_vocab_target = len(self.vocab_target.word_to_id)

        # 目标语言的词表大小

        # 对组成计算图的所有网络层进行声明和初始化
        self.__init_computation_graph()

        # 用于训练的计算图
        self.model_train = self.train_model()

        # 用于推理的计算图
        self.infer_model = self.inference_model()

        if use_pretrain:  # 载入训练好的模型

            self.model_train.load_weights(self.model_path)

    def __init_computation_graph(self):
        """
        对组成计算图的所有网络层进行声明和初始化
        1.带有状态(可被反向传播调整的参数)的网络层定义为类变量后, 可以实现状态的共享
        2.需要重复使用的层可以定义为类变量
        :return:
        """

        self.source_embedding_layer = Embedding(input_dim=self.n_vocab_source, output_dim=self.n_embedding,
                                                name='source_embedding')

        self.encoder_lstm_layer = LSTM(self.n_h, return_sequences=True, return_state=True, name='encoder_lstm')

        self.target_embedding_layer = Embedding(input_dim=self.n_vocab_target, output_dim=self.n_embedding,
                                                name='target_embedding')

        self.decoder_lstm_layer = LSTM(self.n_h, return_sequences=True, return_state=True, name='decoder_lstm')

        self.dense_layer = Dense(self.n_vocab_target, activation='softmax', name='dense')

        self.output_layer = TimeDistributed(self.dense_layer, name='output')

        self.lambda_squezze = Lambda(K.squeeze, arguments={'axis': 1}, name='squeze')

        self.lambda_argmax = Lambda(K.argmax, arguments={'axis': -1}, name='argmax')

        self.lambda_expand_dims = Lambda(K.expand_dims, arguments={'axis': 1}, name='expand_dims')

    def train_model(self):
        """
        将各个 网络层(layer) 拼接为训练计算图(model)
        :return:
        """

        # 编码器 encoder

        batch_source = Input(shape=(self.encoder_length,), name='input_source')  # shape (None, encoder_length)
        source_embedding = self.source_embedding_layer(inputs=batch_source)  # shape (None, encoder_length, n_embedding)

        mask_source = None
        # mask_source = (batch_source != self._null)  # shape(N,encoder_length)
        # 因为训练时采用 mini-batch, 一个 batch 中的所有的 sentence 都是定长, 若有句子不够长度 则用 <null> 进行填充
        # 用 <null> 填充的时刻不能被计入损失中, 也不用求梯度


        out_encoder_lstm, state_h, state_c, = self.encoder_lstm_layer(
            inputs=source_embedding, mask=mask_source)  # out_lstm1 shape : (None, encoder_length, n_h)

        # ---------------------#

        # 解码器 decoder
        h1 = state_h
        c1 = state_c

        batch_target = Input(shape=(self.decoder_length,), name='input_target')  # shape (None, decoder_length)

        mask_target = None
        # mask_target = (batch_target != self._null)  # shape(N,encoder_length)

        target_embedding = self.target_embedding_layer(inputs=batch_target)  # shape (None, decoder_length, n_embedding)

        out_decoder_lstm, h1, c1 = self.decoder_lstm_layer(inputs=target_embedding, initial_state=[h1, c1], mask=mask_target)  # out_decoder_lstm shape (None, decoder_length, n_h)
        # initial_state=[previous hidden state, previous cell state]

        outputs = self.dense_layer(inputs=out_decoder_lstm)  # shape (None, decoder_length, n_vocab_target)

        model = Model(inputs=[batch_source, batch_target], outputs=outputs)

        return model

    def fit(self, train_data_generator, valid_data_generator, dataset_obj, epoch_num=5, batch_size=64):
        """
        训练模型
        :param train_data_generator: 训练数据生成器
        :param valid_data_generator: 验证数据生成器
        :param dataset_obj : 数据集实例
        :param epoch_num: 模型训练的 epoch 个数,  一般训练集所有的样本模型都见过一遍才算一个 epoch
        :param batch_size: 选择 min-Batch梯度下降时, 每一次输入模型的样本个数
        :return:
        """

        # 打印 模型(计算图) 的所有网络层
        print(self.model_train.summary())

        # 输出训练计算图的图片
        plot_model(self.model_train, to_file='docs/images/train_model.png', show_layer_names=True, show_shapes=True)

        checkpoint_models_path = 'models/cache/'

        # Callbacks
        # 在根目录下运行 tensorboard --logdir ./logs
        tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True,
                                                   write_images=True)
        model_names = checkpoint_models_path + 'model.{epoch:02d}-{val_loss:.4f}.hdf5'

        # 模型持久化: 若某次 epcho 模型在 验证集上的损失比之前的最小损失小, 则将模型作为最佳模型持久化
        model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)

        # 早停: 在验证集上, 损失经过 patience 次的迭代后, 仍然没有下降则暂停训练
        early_stop = EarlyStopping('val_loss', patience=30)

        # 根据验证数据集上的损失, 调整学习率
        # patience=5 忍受多少个 epcho 验证集上的损失没有下降, 则更新学习率
        # factor=0.1 每次更新学习率为更新前的 0.1,
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=0.001)

        # 一个好的模型训练的现象是 val_loss 也随着 train_loss 在不断下降
        opt = Adam(lr=5e-3, beta_1=0.9, beta_2=0.999, decay=0.01/epoch_num)


        # self.model_train.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        self.model_train.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


        # Final callbacks
        callbacks = [tensor_board, model_checkpoint, early_stop]

        #  N_train : 训练样本总数
        #  N_valid : 验证样本总数

        history = self.model_train.fit(
            train_data_generator,
            steps_per_epoch=dataset_obj.N_train // batch_size,
            validation_data=valid_data_generator,
            validation_steps=dataset_obj.N_valid // batch_size,
            epochs=epoch_num,
            verbose=1,
            callbacks=callbacks)

        # 将训练好的模型保存到文件
        self.model_train.save(self.model_path)

        return history

    def inference_model(self):
        """
        将各个 网络层(layer) 拼接为推理计算图
        实现了一体化模型解码
        :return:
        """
        # 编码器 encoder

        batch_source = Input(shape=(self.encoder_length,), name='input_source')  # shape (None, encoder_length)
        source_embedding = self.source_embedding_layer(inputs=batch_source)  # shape (None, encoder_length, n_embedding)

        out_encoder_lstm, state_h, state_c, = self.encoder_lstm_layer(
            inputs=source_embedding)  # out_lstm1 shape : (None, encoder_length, n_h)

        # ---------------------#
        # 解码器 decoder

        h1 = state_h
        c1 = state_c

        batch_target = Input(shape=(1,), name='input_target')  # shape: (None, 1)

        decoder_input = batch_target # shape: (None, 1)

        outs = []

        for t in range(self.decoder_length):

            target_embedding = self.target_embedding_layer(inputs=decoder_input)  # shape (None, 1, n_embedding)

            out_decoder_lstm_one_step, h1, c1 = self.decoder_lstm_layer(inputs=target_embedding, initial_state=[h1, c1]) # out_decoder_lstm_one_step shape (None, 1, n_h)
            # initial_state=[previous hidden state, previous cell state]

            out_decoder = self.lambda_squezze(out_decoder_lstm_one_step)  # shape (None, n_h)

            out_dense = self.dense_layer(inputs=out_decoder)  # shape (None, n_vocab)

            out_max_id = self.lambda_argmax(out_dense)  # shape (None, )

            decoder_input = self.lambda_expand_dims(out_max_id)  # shape (None, 1)

            outs.append(out_max_id)

        outputs = Lambda(K.permute_dimensions, arguments={'pattern': (1, 0)})(outs)
        # outs shape (decoder_length, None) -> outputs shape (None, decoder_length)

        model = Model(inputs=[batch_source, batch_target], outputs=outputs)

        return model

    def inference(self, batch_source):
        """
        使用训练好的模型进行推理
        :param batch_source:
        :return:
        """
        # 打印 模型(计算图) 的所有网络层
        print(self.infer_model.summary())

        # 输出推理计算图的图片
        plot_model(self.infer_model, to_file='docs/images/infer_model.png', show_layer_names=True, show_shapes=True)

        N = np.shape(batch_source)[0]  # 一个批次的样本的数量

        batch_target = np.ones((N, 1), dtype=np.float32) * self._start  # 全部是 <START>

        preds = self.infer_model.predict([batch_source, batch_target])

        decode_result = np.array(preds)  # shape (N, decoder_length)

        candidates = []

        # print(decode_result)

        for prediction in decode_result:
            output = ' '.join([self.vocab_target.map_id_to_word(i) for i in prediction])
            candidates.append(output)

        return candidates


class MyCbk(keras.callbacks.Callback):

    def __init__(self, model, checkpoint_models_path):
        keras.callbacks.Callback.__init__(self)
        self.model_to_save = model
        self.checkpoint_models_path = checkpoint_models_path

    def on_epoch_end(self, epoch, logs=None):
        fmt = self.checkpoint_models_path + 'model.%02d-%.4f.hdf5'
        self.model_to_save.save(fmt % (epoch, logs['val_loss']))


class TestV1:

    def test_training(self):

        # 1. 数据集的预处理, 运行 dataset_xrh.py 中的 DataPreprocess -> do_mian()

        dataset_obj = AnkiDataset(base_dir='dataset/anki', mode='train')

        # 打印预处理后的数据集的摘要信息
        print('preprocess dataset info:')
        print('N_train: {}, N_valid:{}, max_source_length:{}, max_target_length:{}'.format(dataset_obj.N_train,
                                                                                           dataset_obj.N_valid,
                                                                                           dataset_obj.max_source_length,
                                                                                           dataset_obj.max_target_length))
        print('-------------------------')

        # 2. 训练模型

        max_source_length = dataset_obj.max_source_length
        max_target_length = dataset_obj.max_target_length
        # 取决于数据集, 在 dataset_xrh.py 中的 DataPreprocess -> do_mian(max_caption_length=40, freq_threshold=0) 中进行调整

        n_h = 50
        n_embedding = 50
        encoder_length = max_source_length
        decoder_length = max_target_length - 1

        n_vocab_source = len(dataset_obj.vocab_source.word_to_id)  # 源语言的词典大小
        n_vocab_target = len(dataset_obj.vocab_target.word_to_id)  # 目标语言的词典大小

        print('model architecture param:')
        print('n_h:{}, n_embedding:{}, encoder_length:{}, decoder_length:{}'.format(n_h, n_embedding, encoder_length,
                                                                                    decoder_length))
        print('-------------------------')

        model_path = 'models/machine_translation_seq2seq_hid_50_emb_50.h5'  # 后缀只能使用 .h5

        trainer = MachineTranslation(encoder_length=encoder_length, decoder_length=decoder_length,
                                 n_h=n_h, n_embedding=n_embedding,
                                 vocab_source=dataset_obj.vocab_source,
                                 vocab_target=dataset_obj.vocab_target,
                                 model_path=model_path,
                                 use_pretrain=True
                                 )
        # use_pretrain=True: 在已有的模型参数基础上, 进行更进一步的训练

        batch_size = 128
        epoch_num = 50

        train_data_generator = BatchDataGenSequence(n_h=n_h, n_embedding=n_embedding, n_vocab_target=n_vocab_target,
                                                    batch_size=batch_size, dataset=dataset_obj.dataset['train'])
        valid_data_generator = BatchDataGenSequence(n_h=n_h, n_embedding=n_embedding, n_vocab_target=n_vocab_target,
                                                    batch_size=batch_size, dataset=dataset_obj.dataset['valid'])

        trainer.fit(train_data_generator, valid_data_generator, dataset_obj=dataset_obj, epoch_num=epoch_num,
                          batch_size=batch_size)

    def test_evaluate(self):
        # 1. 数据集的预处理, 运行 dataset_xrh.py 中的 DataPreprocess 中的 do_mian()

        dataset_obj = AnkiDataset(base_dir='dataset/anki', mode='infer')

        max_source_length = 9
        max_target_length = 16

        n_embedding = 50
        n_h = 50
        encoder_length = max_source_length
        decoder_length = max_target_length - 1

        n_vocab_source = len(dataset_obj.vocab_source.word_to_id)  # 源语言的词典大小
        n_vocab_target = len(dataset_obj.vocab_target.word_to_id)  # 目标语言的词典大小

        print('model architecture param:')
        print('n_h:{}, n_embedding:{}, encoder_length:{}, decoder_length:{}'.format(n_h, n_embedding, encoder_length,
                                                                                    decoder_length))
        print('-------------------------')

        # 2.模型推理

        # model_path = 'models/machine_translation_seq2seq_hid_50_emb_50.h5'  # 后缀只能使用 .h5

        model_path = 'models/cache/model.11-0.9418.hdf5'

        infer = MachineTranslation(encoder_length=encoder_length, decoder_length=decoder_length,
                                 n_h=n_h, n_embedding=n_embedding,
                                 vocab_source=dataset_obj.vocab_source,
                                 vocab_target=dataset_obj.vocab_target,
                                 model_path=model_path,
                                 use_pretrain=True
                                 )

        source_target_dict = dataset_obj.valid_source_target_dict
        # 训练数据集
        # source_target_dict = dataset_obj.train_source_target_dict

        source_id_list = list(source_target_dict.keys())

        m = 3151  # 测试数据集的 source_id 的总数

        # m = 100

        source_id_batch = source_id_list[:m]

        print('test source_id num:{}'.format(len(source_id_batch)))

        batch_source = np.array(
            [source_target_dict[source_id][0] for source_id in source_id_batch])

        batch_source_encoding = np.array(
            [source_target_dict[source_id][1] for source_id in source_id_batch])

        references = [source_target_dict[source_id][2] for source_id in source_id_batch]

        candidates = infer.inference(batch_source_encoding)

        print('\nbatch_sources:')
        for i in range(0, 10):
            print(batch_source[i])

        print('\ncandidates:')
        for i in range(0, 10):
            print(candidates[i])


        print('\nreferences:')
        for i in range(0, 10):
            print(references[i])

        evaluate_obj = Evaluate()

        bleu_score, _ = evaluate_obj.evaluate_bleu(references, candidates)

        print('bleu_score:{}'.format(bleu_score))


if __name__ == '__main__':
    test = TestV1()

    # TODO: 每次实验前
    #  1. 更改最终模型存放的路径
    #  2. 运行脚本  clean_training_cache_file.bat

    test.test_training()

    # test.test_evaluate()