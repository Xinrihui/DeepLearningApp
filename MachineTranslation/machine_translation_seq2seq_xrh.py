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

from tensorflow.keras.models import Model

from lib.build_dataset_xrh import *
from lib.evaluate_xrh import *
from lib.get_dataset_xrh import *
from lib.tf_data_tokenize_xrh import *


class CheckoutCallback(keras.callbacks.Callback):
    """
    回调函数, 实现在每一次 epoch 后 checkout 训练好的模型,
    并且计算在验证集上的 bleu 分数

    """

    def __init__(self, model_train, model_infer, vocab_obj, batch_size, valid_source_target_dict, checkpoint_models_path):

        keras.callbacks.Callback.__init__(self)

        self.model_train = model_train
        self.model_infer = model_infer
        self.vocab_obj = vocab_obj

        self.batch_source_dataset, self.references = self.prepare_valid_data(batch_size, valid_source_target_dict)

        self.evaluate_obj = Evaluate(
                                    with_unk=False,
                                    _null_str='',
                                    _start_str='[START]',
                                    _end_str='[END]',
                                    _unk_str='[UNK]')

        self.checkpoint_models_path = checkpoint_models_path

    def prepare_valid_data(self, batch_size, valid_source_target_dict):
        """
        返回 图片的 embedding 向量 和 图片对应的 caption

        :param batch_size:
        :param valid_source_target_dict:
        :return:
        """

        source_list = list(valid_source_target_dict.keys())

        print('valid source num:{}'.format(len(source_list)))

        source_vector = np.array(
            [list(valid_source_target_dict[source]['vector']) for source in source_list])

        references = [valid_source_target_dict[source]['target'] for source in source_list]

        source_dataset = tf.data.Dataset.from_tensor_slices(source_vector)

        batch_source_dataset = source_dataset.batch(batch_size)

        return batch_source_dataset, references

    def inference_bleu(self):
        """
        使用验证数据集进行推理, 并计算 bleu

        :return:
        """

        # batch_source_dataset shape (N_batch, encoder_length)
        preds = self.model_infer.predict(self.batch_source_dataset)

        # decode_result = np.array(preds)  # shape (N_batch, encoder_length)
        # candidates = []
        # for prediction in decode_result:
        #     output = ' '.join(self.vocab_obj.map_id_to_word(prediction))
        #     candidates.append(output)

        decode_result = self.vocab_obj.map_id_to_word(preds)

        decode_result = tf.strings.reduce_join(decode_result, axis=1,
                                             separator=' ')

        candidates = [sentence.numpy().decode('utf-8') for sentence in decode_result]


        bleu_score, _ = self.evaluate_obj.evaluate_bleu(self.references, candidates)

        print()
        print('bleu_score:{}'.format(bleu_score))


    def on_epoch_end(self, epoch, logs=None):

        # checkout 模型
        fmt = self.checkpoint_models_path + 'model.%02d-%.4f.h5'
        self.model_train.save(fmt % (epoch, logs['val_loss']))

        # 计算 bleu 分数
        self.inference_bleu()



class MachineTranslation:
    """

    基于 seq2seq 的神经机器翻译模型

    1. 解码采用一体化模型的方式, 即建立推理计算图, 将每一步的解码都在计算图中完成
    2. 实现了基于 tf.data 的数据生成器

    Author: xrh
    Date: 2021-11-10

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

        self._null = int(self.vocab_source.map_word_to_id(_null_str))  # 空
        self._start = int(vocab_source.map_word_to_id(_start_str))  # 句子的开始
        self._end = int(self.vocab_source.map_word_to_id(_end_str))  # 句子的结束
        self._unk = int(self.vocab_source.map_word_to_id(_unk_str))  # 未登录词

        # vocab_source 和 vocab_target 的标号不同
        self._null_target = int(self.vocab_target.map_word_to_id(_null_str))  # 空
        self._start_target = int(self.vocab_target.map_word_to_id(_start_str))  # 句子的开始
        self._end_target = int(self.vocab_target.map_word_to_id(_end_str)) # 句子的结束
        self._unk_target = int(self.vocab_target.map_word_to_id(_unk_str))  # 未登录词

        self.model_path = model_path

        # 源语言的词表大小
        self.n_vocab_source = self.vocab_source.n_vocab

        # 目标语言的词表大小
        self.n_vocab_target = self.vocab_target.n_vocab

        # 目标语言的词表大小

        # 对组成计算图的所有网络层进行声明和初始化
        self.__init_computation_graph()

        # 用于训练的计算图
        self.model_train = self.train_model()

        # 用于推理的计算图
        self.model_infer = self.infer_model()

        if use_pretrain:  # 载入训练好的模型

            self.model_train.load_weights(self.model_path)

    def __init_computation_graph(self):
        """
        对组成计算图的所有网络层进行声明和初始化

        1.带有状态(可被反向传播调整的参数)的网络层定义为类变量后, 可以实现状态的共享

        2.需要重复使用的层可以定义为类变量

        :return:
        """

        self.image_embedding_dense = Dense(self.n_h, activation='relu', name='pict_embedding')

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
        将各个 网络层(layer) 拼接为训练计算图

        :return:
        """

        # 编码器 encoder

        batch_source = Input(shape=(self.encoder_length,), name='input_source')  # shape (None, encoder_length)

        batch_target = Input(shape=(self.decoder_length,), name='input_target')  # shape (None, decoder_length)


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


        mask_target = None
        # mask_target = (batch_target != self._null)  # shape(N,encoder_length)

        target_embedding = self.target_embedding_layer(inputs=batch_target)  # shape (None, decoder_length, n_embedding)

        out_decoder_lstm, h1, c1 = self.decoder_lstm_layer(inputs=target_embedding, initial_state=[h1, c1], mask=mask_target)  # out_decoder_lstm shape (None, decoder_length, n_h)
        # initial_state=[previous hidden state, previous cell state]

        outputs = self.output_layer(inputs=out_decoder_lstm)  # shape (None, decoder_length, n_vocab_target)

        model = Model(inputs=[batch_source, batch_target], outputs=outputs)

        return model


    def __loss_function(self, real, pred):
        """
        自定义的损失函数

        :param real: 标签值
        :param pred: 预测值
        :return:
        """
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')

        mask = tf.math.logical_not(tf.math.equal(real, self._null_target))  # 输出序列中为空的不计入损失函数
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def fit_tf_data(self, train_dataset, valid_dataset, valid_source_target_dict, epoch_num=20,
            batch_size=256, buffer_size=10000,
            ):
        """
        使用内置方法训练模型

        :param train_dataset: 训练数据生成器
        :param valid_dataset: 验证数据生成器
        :param valid_source_target_dict: 验证数据的字典, 用于计算 bleu

        :param epoch_num: 模型训练的 epoch 个数,  一般训练集所有的样本模型都见过一遍才算一个 epoch
        :param batch_size: 选择 min-Batch梯度下降时, 每一次输入模型的样本个数 (默认 = 128)
        :param buffer_size:

        :return:
        """

        # 打印 模型(计算图) 的所有网络层
        print(self.model_train.summary())

        # tf.data 的数据混洗,分批和预取
        # Shuffle and batch
        train_dataset_batch = train_dataset.shuffle(buffer_size).batch(batch_size)
        train_dataset_prefetch = train_dataset_batch.prefetch(
            buffer_size=tf.data.AUTOTUNE)  # 要预取的元素数量应等于（或大于）单个训练步骤 epoch 消耗的批次数量

        valid_dataset_batch = valid_dataset.shuffle(buffer_size).batch(batch_size)
        valid_dataset_prefetch = valid_dataset_batch.prefetch(buffer_size=tf.data.AUTOTUNE)

        checkpoint_models_path = 'models/cache/'

        # Callbacks
        # 在根目录下运行 tensorboard --logdir ./logs
        tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True,
                                                   write_images=True)

        model_names = checkpoint_models_path + 'model.{epoch:02d}-{val_loss:.4f}.h5'

        # 模型持久化: 若某次 epcho 模型在 验证集上的损失比之前的最小损失小, 则将模型作为最佳模型持久化
        # model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=False)

        model_checkpoint_with_eval = CheckoutCallback(model_train=self.model_train, model_infer=self.model_infer,
                                                      vocab_obj=self.vocab_target, batch_size=batch_size,
                                                      valid_source_target_dict=valid_source_target_dict,
                                                      checkpoint_models_path=checkpoint_models_path)

        # 早停: 在验证集上, 损失经过 patience 次的迭代后, 仍然没有下降则暂停训练
        early_stop = EarlyStopping('val_loss', patience=5)

        # optimizer = tf.keras.optimizers.Adam(learning_rate=4e-4)

        optimizer = 'rmsprop'
        loss_function = 'sparse_categorical_crossentropy'

        self.model_train.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

        # Final callbacks
        callbacks = [model_checkpoint_with_eval, early_stop, tensor_board]


        # 查看会话管理可知需要8GB显存
        history = self.model_train.fit(
            x=train_dataset_prefetch,
            epochs=epoch_num,
            validation_data=valid_dataset_prefetch,
            verbose=1,
            callbacks=callbacks)


        # 将训练好的模型保存到文件
        self.model_train.save(self.model_path)

        return history

    def infer_model(self):
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

        # batch_target = Input(shape=(1,), name='input_target')  # shape: (None, 1)

        N_batch = tf.shape(batch_source)[0]
        batch_target = tf.ones((N_batch, 1)) * self._start_target  # (N_batch, 1)

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

        model = Model(inputs=[batch_source], outputs=outputs)

        return model

    def inference(self, batch_source):
        """
        使用训练好的模型进行推理

        :param batch_source:

        :return:
        """
        # 打印 模型(计算图) 的所有网络层
        print(self.model_infer.summary())

        # 输出推理计算图的图片
        # plot_model(self.model_infer, to_file='docs/images/model_infer.png', show_layer_names=True, show_shapes=True)

        preds = self.model_infer.predict(batch_source)

        decode_result = self.vocab_target.map_id_to_word(preds)

        decode_result = tf.strings.reduce_join(decode_result, axis=1,
                                             separator=' ')

        candidates = [sentence.numpy().decode('utf-8') for sentence in decode_result]

        return candidates



class Test_WMT14_Eng_Ge_Dataset:

    def test_training(self):

        # 1. 数据集的预处理, 运行 tf_data_utils_xrh.py 中的 DataPreprocess -> do_mian()

        # dataset_obj = WMT14_Eng_Ge_Dataset(base_dir='dataset/WMT-14-English-Germa', cache_data_folder='cache_small_data', reverse_source=False, mode='train')

        dataset_obj = WMT14_Eng_Ge_Dataset(base_dir='dataset/WMT-14-English-Germa', cache_data_folder='cache_data', reverse_source=True, mode='train')


        # 2. 训练模型

        max_seq_length = 50

        # 编码器的长度
        encoder_length = max_seq_length

        # 解码器的长度
        decoder_length = max_seq_length-1

        # 词嵌入的维度
        n_embedding = 1000

        # 编码器 和 解码器的隐状态维度
        n_h = 1000

        # 词表大小
        n_vocab_source = dataset_obj.vocab_source.n_vocab  # 源语言的词典大小
        n_vocab_target = dataset_obj.vocab_target.n_vocab  # 目标语言的词典大小

        dropout_rates = (0.8,)

        print('model architecture param:')
        print('n_h:{}, n_embedding:{}, encoder_length:{}, decoder_length:{}'.format(n_h, n_embedding, encoder_length, decoder_length))
        print('n_vocab_source:{}, n_vocab_target:{}'.format(n_vocab_source, n_vocab_target))
        print('-------------------------')

        model_path = 'models/machine_translation_seq2seq_hid_1000_emb_1000.h5'  # 后缀只能使用 .h5

        trainer = MachineTranslation(encoder_length=encoder_length, decoder_length=decoder_length,
                                 n_h=n_h, n_embedding=n_embedding,
                                 vocab_source=dataset_obj.vocab_source,
                                 vocab_target=dataset_obj.vocab_target,
                                 _null_str='',
                                 _start_str='[START]',
                                 _end_str='[END]',
                                 _unk_str='[UNK]',
                                 model_path=model_path,
                                 use_pretrain=False
                                 )
        # use_pretrain=True: 在已有的模型参数基础上, 进行更进一步的训练

        batch_size = 256
        buffer_size = 10000

        epoch_num = 10

        trainer.fit_tf_data(train_dataset=dataset_obj.train_dataset, valid_dataset=dataset_obj.valid_dataset,
                          valid_source_target_dict=dataset_obj.valid_source_target_dict,
                          epoch_num=epoch_num, batch_size=batch_size, buffer_size=buffer_size)



    def test_evaluating(self):

        # 1. 数据集的预处理, 运行 tf_data_utils_xrh.py 中的 DataPreprocess -> do_mian()

        # dataset_obj = WMT14_Eng_Ge_Dataset(base_dir='dataset/WMT-14-English-Germa', cache_data_folder='cache_small_data', reverse_source=True, mode='infer')

        dataset_obj = WMT14_Eng_Ge_Dataset(base_dir='dataset/WMT-14-English-Germa', cache_data_folder='cache_data', reverse_source=True, mode='infer')

        # 2. 训练模型

        max_seq_length = 50

        # 编码器的长度
        encoder_length = max_seq_length

        # 解码器的长度
        decoder_length = max_seq_length-1

        # 词嵌入的维度
        n_embedding = 1000

        # 编码器 和 解码器的隐状态维度
        n_h = 1000

        # 词表大小
        n_vocab_source = dataset_obj.vocab_source.n_vocab # 源语言的词典大小
        n_vocab_target = dataset_obj.vocab_target.n_vocab # 目标语言的词典大小

        dropout_rates = (0.8,)

        print('model architecture param:')
        print('n_h:{}, n_embedding:{}, encoder_length:{}, decoder_length:{}'.format(n_h, n_embedding, encoder_length, decoder_length))
        print('n_vocab_source:{}, n_vocab_target:{}'.format(n_vocab_source, n_vocab_target))
        print('-------------------------')


        # 2.模型推理

        model_path = 'models/machine_translation_seq2seq_hid_1000_emb_1000.h5'

        # model_path = 'models/cache/model.13-1.2755.h5'

        infer = MachineTranslation(encoder_length=encoder_length, decoder_length=decoder_length,
                                 n_h=n_h, n_embedding=n_embedding,
                                 vocab_source=dataset_obj.vocab_source,
                                 vocab_target=dataset_obj.vocab_target,
                                 _null_str='',
                                 _start_str='[START]',
                                 _end_str='[END]',
                                 _unk_str='[UNK]',
                                 model_path=model_path,
                                 use_pretrain=True
                                 )

        batch_size = 256

        test_source_target_dict = dataset_obj.test_source_target_dict

        source_list = list(test_source_target_dict.keys())

        print('valid source num:{}'.format(len(source_list)))

        source_vector = np.array(
            [list(test_source_target_dict[source]['vector']) for source in source_list])

        references = [test_source_target_dict[source]['target'] for source in source_list]

        source_dataset = tf.data.Dataset.from_tensor_slices(source_vector)

        batch_source_dataset = source_dataset.batch(batch_size)

        candidates = infer.inference(batch_source_dataset)

        print('\ncandidates:')
        for i in range(0, 10):
            print(candidates[i])

        print('\nreferences:')
        for i in range(0, 10):
            print(references[i])

        evaluate_obj = Evaluate(
                                 with_unk=False,
                                _null_str='',
                                _start_str='[START]',
                                _end_str='[END]',
                                _unk_str='[UNK]')

        bleu_score, _ = evaluate_obj.evaluate_bleu(references, candidates, bleu_N=4)

        print('bleu_score:{}'.format(bleu_score))


if __name__ == '__main__':


    test = Test_WMT14_Eng_Ge_Dataset()

    # TODO: 每次实验前
    #  1. 更改最终模型存放的路径
    #  2. 运行脚本  clean_training_cache_file.bat

    test.test_training()

    # test.test_evaluating()
