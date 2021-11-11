#!/usr/bin/python
# -*- coding: UTF-8 -*-

#  适用于 tensorflow >= 2.0, keras 被直接集成到 tensorflow 的内部
#  ref: https://keras.io/about/

from tensorflow.keras.layers import Input, LSTM, TimeDistributed, Bidirectional,Dense, Lambda, Embedding, Dropout, Concatenate, RepeatVector
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import tensorflow.keras as keras

from tensorflow.keras.models import Model

from lib.build_dataset_xrh import *
from lib.evaluate_xrh import *
from lib.get_dataset_xrh import *

# from deprecated import deprecated


import keras_tuner as kt

class ImageCaptionV3:
    """

    基于 LSTM 的图片描述生成器 V3

    1. 实现了 基于 Sequence 的数据批量生成器
    2. 采用 稀疏的交叉熵损失函数, 避免了 标签 one-hot 化后的 OOM 问题
    3. 集成模型, 即将多层 LSTM 进行堆叠, 中间使用 dropout 连接
    4. 使用 Keras Tuner 进行模型超参数的自动搜索

    Author: xrh
    Date: 2021-10-05

    ref:
    1. Show and Tell: A Neural Image Caption Generator
    2. Deep Visual-Semantic Alignments for Generating Image Descriptions
    3. https://github.com/foamliu/Image-Captioning
    """

    def __init__(self, train_seq_length, infer_seq_length,
                 n_h, n_image_feature, n_embedding, n_vocab,
                 vocab_obj,
                 dropout_rates=(0.1, 0.2, 0.4),
                 _null_str='<NULL>',
                 _start_str='<START>',
                 _end_str='<END>',
                 _unk_str='<UNK>',
                 use_pretrain=False,
                 model_path='models/image_caption_model.h5'):
        """
        模型初始化

        :param train_seq_length: 训练时的序列长度
        :param infer_seq_length: 推理时的序列长度
        :param n_h: lstm 中隐藏状态的维度, n_h = 512
        :param n_image_feature: 图片经过 CNN 映射后的维度, n_image_feature = 2048
        :param n_embedding : 词向量维度, n_embedding= 512

        :param n_vocab: 词典的大小
        :param vocab_obj: 词典对象

        :param dropout_rates: 弃置的权重列表, 从底层到顶层

        :param  _null_str: 末尾填充的空
        :param  _start_str: 句子的开始
        :param  _end_str: 句子的结束
        :param  _unk_str: 未登录词

        :param use_pretrain: 使用训练好的模型
        :param model_path: 预训练模型的路径

        """
        self.train_seq_length = train_seq_length
        self.infer_seq_length = infer_seq_length

        self.dropout_rates = dropout_rates

        self.n_h = n_h
        self.n_image_feature = n_image_feature
        self.n_embedding = n_embedding
        self.n_vocab = n_vocab

        self.vocab_obj = vocab_obj

        self._null = self.vocab_obj.map_word_to_id(_null_str)  # 空
        self._start = self.vocab_obj.map_word_to_id(_start_str)  # 句子的开始
        self._end = self.vocab_obj.map_word_to_id(_end_str)  # 句子的结束
        self._unk_str = self.vocab_obj.map_word_to_id(_unk_str) # 未登录词

        self.model_path = model_path

        # 词表大小
        self.n_vocab = len(self.vocab_obj.word_to_id)  # 词表大小 9633

        # 对组成计算图的所有网络层进行声明和初始化
        self.__init_computation_graph()

        # 用于训练的计算图
        self.model_train = self.train_model(self.train_seq_length)

        # 用于推理的计算图
        self.infer_model = self.inference_model(self.infer_seq_length)

        if use_pretrain:  # 载入训练好的模型

            self.model_train.load_weights(self.model_path)


    def __init_computation_graph(self):
        """
        对组成计算图的所有网络层进行声明和初始化

        1.带有状态(可被反向传播调整的参数)的网络层定义为类变量后, 可以实现状态的共享

        2.需要重复使用的层可以定义为类变量

        :return:
        """

        self.pict_embedding_layer = Dense(self.n_h, activation='relu', name='pict_embedding')  # 图片映射的维度必须 与 LSTM隐藏状态一致

        self.word_embedding_layer = Embedding(self.n_vocab, self.n_embedding, name='word_embedding')

        self.lstm_layer1 = LSTM(self.n_h, return_sequences=True, return_state=True, name='lstm1')
        self.dropout_layer1 = Dropout(self.dropout_rates[0], name='dropout1')  # 神经元有 0.1 的概率被弃置

        self.lstm_layer2 = LSTM(self.n_h, return_sequences=True, return_state=True, name='lstm2')
        self.dropout_layer2 = Dropout(self.dropout_rates[1], name='dropout2')  # 神经元有 0.2 的概率被弃置

        self.lstm_layer3 = LSTM(self.n_h, return_sequences=True, return_state=True, name='lstm3')
        self.dropout_layer3 = Dropout(self.dropout_rates[2], name='dropout3')  # 神经元有 0.4 的概率被弃置

        self.dense_layer = Dense(self.n_vocab, activation='softmax', name='dense')

        self.output_layer = TimeDistributed(self.dense_layer, name='output')

        self.lambda_argmax = Lambda(K.argmax, arguments={'axis': -1}, name='argmax')

        self.lambda_squezze = Lambda(K.squeeze, arguments={'axis': 1}, name='squezze')

        self.lambda_expand_dims = Lambda(K.expand_dims, arguments={'axis': 1}, name='expand_dims')

        self.lambda_permute_dimensions = Lambda(K.permute_dimensions, arguments={'pattern': (1, 0)})


    def train_model(self, train_seq_length):
        """
        将各个 网络层(layer) 拼接为训练计算图

        :param train_seq_length:
        :return:
        """

        batch_caption = Input(shape=(train_seq_length), name='input_caption')

        batch_image_feature = Input(shape=(self.n_image_feature), name='input_image_feature')

        pict_embedding = self.pict_embedding_layer(inputs=batch_image_feature)

        word_embedding = self.word_embedding_layer(inputs=batch_caption)

        h_init = pict_embedding  # hidden state
        c_init = Input(shape=(self.n_h), name='c_init')  # cell state

        h1 = h_init
        c1 = c_init

        h2 = h_init
        c2 = c_init

        h3 = h_init
        c3 = c_init

        mask = None

        out_lstm1, _, _ = self.lstm_layer1(inputs=word_embedding, initial_state=[h1, c1], mask=mask)
        #  initial_state=[previous hidden state, previous cell state]
        out_dropout1 = self.dropout_layer1(out_lstm1)

        out_lstm2, _, _ = self.lstm_layer2(inputs=out_dropout1, initial_state=[h2, c2], mask=mask)
        out_dropout2 = self.dropout_layer2(out_lstm2)

        out_lstm3, _, _ = self.lstm_layer3(inputs=out_dropout2, initial_state=[h3, c3], mask=mask)
        out_dropout3 = self.dropout_layer3(out_lstm3)

        outputs = self.output_layer(inputs=out_dropout3)

        model = Model(inputs=[batch_caption, batch_image_feature, c_init], outputs=outputs)

        return model


    def fit(self, train_data_generator, valid_data_generator, dataset_obj, epoch_num=5, batch_size=64):
        """
        训练模型

        :param train_data_generator: 训练数据生成器
        :param valid_data_generator: 验证数据生成器

        :param dataset_obj : 数据集实例
        :param epoch_num: 模型训练的 epoch 个数,  一般训练集所有的样本模型都见过一遍才算一个 epoch
        :param batch_size: 选择 min-Batch梯度下降时, 每一次输入模型的样本个数 (默认 = 2048)

        :return:
        """

        # 打印 模型(计算图) 的所有网络层
        print(self.model_train.summary())

        # 输出训练计算图的图片
        # plot_model(self.model_train, to_file='docs/images/train_model.png')

        checkpoint_models_path = 'models/cache/'

        # Callbacks
        # 在根目录下运行 tensorboard --logdir ./logs
        tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True,
                                                              write_images=True)
        model_names = checkpoint_models_path + 'model.{epoch:02d}-{val_loss:.4f}.h5'

        # 模型持久化: 若某次 epcho 模型在 验证集上的损失比之前的最小损失小, 则将模型作为最佳模型持久化
        model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)

        # 早停: 在验证集上, 损失经过 patience 次的迭代后, 仍然没有下降则暂停训练
        early_stop = EarlyStopping('val_loss', patience=20)

        # 根据验证数据集上的损失, 调整学习率
        # patience=10 忍受多少个 epcho 验证集上的损失没有下降, 则更新学习率
        # factor=0.5 每次更新学习率为更新前的 0.5,
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=0.001)
        # opt = Adam(lr=5e-3, beta_1=0.9, beta_2=0.999)

        #TODO: 学习率的衰减率设置不好也会导致模型不收敛, 无语子
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

    def inference_model(self, infer_seq_length):
        """
        将各个 网络层(layer) 拼接为推理计算图

        :param infer_seq_length:

        :return:
        """
        # 推理时 第一个时间步的输入为 '<start>'的标号
        batch_caption = Input(shape=(1), name='input_caption')  # shape: (m, 1)  m - 当前批次的样本数

        batch_image_feature = Input(shape=(self.n_image_feature), name='input_image_feature')  # shape: (m, n_image_feature)

        pict_embedding = self.pict_embedding_layer(inputs=batch_image_feature)  # shape: (m, n_h)

        c_init = Input(shape=(self.n_h), name='c0')  # cell state, shape: (m, n_h)

        h_init = pict_embedding  # hidden state

        # 第一个时间步的输入
        # 第一层的 LSTM
        h1 = h_init
        c1 = c_init

        # 第二层的 LSTM
        h2 = h_init
        c2 = c_init

        # 第三层的 LSTM
        h3 = h_init
        c3 = c_init

        inputs = batch_caption

        outs = []

        for t in range(infer_seq_length):  # max_length 遍历所有的时间步

            word_embedding = self.word_embedding_layer(inputs=inputs)  # shape: (m, 1, n_embedding)

            out_lstm1_one_step, h1, c1 = self.lstm_layer1(inputs=word_embedding, initial_state=[h1, c1])
            out_dropout1 = self.dropout_layer1(out_lstm1_one_step)

            out_lstm2_one_step, h2, c2 = self.lstm_layer2(inputs=out_dropout1, initial_state=[h2, c2])
            out_dropout2 = self.dropout_layer2(out_lstm2_one_step)

            out_lstm3_one_step, h3, c3 = self.lstm_layer3(inputs=out_dropout2, initial_state=[h3, c3])
            out_dropout3 = self.dropout_layer2(out_lstm3_one_step)

            out_one_step = self.lambda_squezze(inputs=out_dropout3)
            # out_one_step shape=(m, 300)

            out_dense = self.dense_layer(inputs=out_one_step)  # shape (m, n_vocab)

            max_idx = self.lambda_argmax(inputs=out_dense)  # shape (m, )

            inputs = self.lambda_expand_dims(inputs=max_idx)  # shape (m, 1)

            outs.append(max_idx)

        outputs = self.lambda_permute_dimensions(outs)

        # outputs shape (m, infer_seq_length)
        model = Model(inputs=[batch_caption, batch_image_feature, c_init], outputs=outputs)

        return model

    def inference(self, batch_image_feature):
        """
        使用训练好的模型进行推理

        :param batch_image_feature: 图片向量 shape (N_batch,n_image_feature)
        :return:
        """
        # 打印 模型(计算图) 的所有网络层
        # print(self.infer_model.summary())

        # 输出训练计算图的图片
        # plot_model(self.infer_model, to_file='docs/images/infer_model.png')

        N_batch = np.shape(batch_image_feature)[0]  # 一个批次的样本的数量

        batch_caption = np.ones((N_batch, 1), dtype=np.float32) * self._start  # 全部是 <START>

        zero_init = np.zeros((N_batch, self.n_h))

        preds = self.infer_model.predict([batch_caption, batch_image_feature, zero_init])  # preds shape (max_length+1, m)

        decode_result = np.array(preds)  # shape (m, infer_seq_length)

        candidates = []

        # print(decode_result)

        for prediction in decode_result:
            output = ' '.join([self.vocab_obj.map_id_to_word(i) for i in prediction])
            candidates.append(output)

        return candidates



class MyCbk(keras.callbacks.Callback):
    
    def __init__(self, model, checkpoint_models_path):
        keras.callbacks.Callback.__init__(self)
        self.model_to_save = model
        self.checkpoint_models_path = checkpoint_models_path

    def on_epoch_end(self, epoch, logs=None):
        fmt = self.checkpoint_models_path + 'model.%02d-%.4f.h5'
        self.model_to_save.save(fmt % (epoch, logs['val_loss']))



class SearchHyperParameter:
    """
    搜索模型的最优超参数

    Author: xrh
    Date: 2021-10-15

    """
    def __init__(self, batch_size=128, max_epochs=15, log_dir='E:\python package\python-project\DeepLearningApp\Image Caption\logs', project_name='turning',
                 best_model_folder='models/sota/'
                 ):

        self.dataset_obj = FlickerDataset(base_dir='dataset/', mode='train')

        # 打印预处理后的数据集的摘要信息
        print('preprocess dataset info:')
        print('N_train: {}, N_valid:{}, n_image_feature:{}, max_caption_length:{}'.format(self.dataset_obj.N_train, self.dataset_obj.N_valid, self.dataset_obj.feature_dim, self.dataset_obj.caption_length))
        print('-------------------------')

        # current_dir = os.getcwd()  # 当前路径

        self.batch_size = batch_size
        self.max_epochs = max_epochs

        self.best_model_folder = best_model_folder

        self.tuner = kt.Hyperband(
            self.build_model,
            objective='val_loss',  # 优化目标
            max_epochs=self.max_epochs,
            directory=log_dir,  # TODO: Bug: 必须使用绝对的路径, 并且不能太长, 否则报错 UnicodeDecodeError
            project_name=project_name)


    def search_best_hps(self):

        train_data_generator = BatchDataGenSequence(n_h=self.n_h, n_embedding=self.n_embedding, n_vocab=self.n_vocab, batch_size=self.batch_size, dataset=self.dataset_obj.dataset['train'])
        valid_data_generator = BatchDataGenSequence(n_h=self.n_h, n_embedding=self.n_embedding, n_vocab=self.n_vocab, batch_size=self.batch_size, dataset=self.dataset_obj.dataset['valid'])

        # 提前停止: 在验证集上, 损失经过 patience 次的迭代后, 仍然没有下降则暂停训练
        early_stop = EarlyStopping('val_loss', patience=3)

        callbacks = [early_stop]

        #  N_train : 训练样本总数
        #  N_valid : 验证样本总数

        self.tuner.search(
                           train_data_generator,
                           steps_per_epoch=self.dataset_obj.N_train // self.batch_size,
                           validation_data=valid_data_generator,
                           validation_steps=self.dataset_obj.N_valid // self.batch_size,
                           epochs=self.max_epochs,
                           verbose=1,
                           callbacks=callbacks)

        print('search best model complete ')

        print(self.tuner.results_summary(num_trials=3))

    def find_best_epoch_num(self, top_k=0):
        """
        加载最佳的超参数对应的模型, 并找到训练该模型的最佳周期数

        :param top_k: 模型的排名, 排名从 0 开始, 0 代表最好
        :return:
        """

        # Get the optimal hyperparameters
        best_hps = self.tuner.get_best_hyperparameters(num_trials=top_k+1)[top_k]

        print('best hyperparameters: ')
        print('dropout_rate1:', best_hps.get('dropout_rate1'))
        print('dropout_rate2:', best_hps.get('dropout_rate2'))
        print('dropout_rate2:', best_hps.get('dropout_rate3'))

        # best_model = self.tuner.get_best_models(num_models=top_k+1)[top_k]

        best_model = self.tuner.hypermodel.build(best_hps)

        train_data_generator = BatchDataGenSequence(n_h=self.n_h, n_embedding=self.n_embedding, n_vocab=self.n_vocab, batch_size=self.batch_size, dataset=self.dataset_obj.dataset['train'])
        valid_data_generator = BatchDataGenSequence(n_h=self.n_h, n_embedding=self.n_embedding, n_vocab=self.n_vocab, batch_size=self.batch_size, dataset=self.dataset_obj.dataset['valid'])

        # Callbacks

        checkpoint_models_path = 'models/cache/'

        # 在根目录下运行 tensorboard --logdir ./logs
        tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True,
                                                              write_images=True)
        model_names = checkpoint_models_path + 'model.{epoch:02d}-{val_loss:.4f}.h5'

        # 模型持久化: 若某次 epcho 模型在 验证集上的损失比之前的最小损失小, 则将模型作为最佳模型持久化
        model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)

        # 早停: 在验证集上, 损失经过 patience 次的迭代后, 仍然没有下降则暂停训练
        early_stop = EarlyStopping('val_loss', patience=5)

        callbacks = [tensor_board, model_checkpoint, early_stop]

        #  N_train : 训练样本总数
        #  N_valid : 验证样本总数

        history = best_model.fit(
                       train_data_generator,
                       steps_per_epoch=self.dataset_obj.N_train // self.batch_size,
                       validation_data=valid_data_generator,
                       validation_steps=self.dataset_obj.N_valid // self.batch_size,
                       epochs=self.max_epochs,
                       verbose=1,
                       callbacks=callbacks)

        val_loss_per_epoch = history.history['val_loss']
        best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))



    def build_model(self, hp):

        n_image_feature = self.dataset_obj.feature_dim
        max_caption_length = self.dataset_obj.caption_length
        # 取决于数据集, 在 dataset_xrh.py 中的 DataPreprocess -> do_mian(max_caption_length=40, freq_threshold=0) 中进行调整

        self.n_h = 512
        self.n_embedding = 512
        self.max_length = max_caption_length - 1
        self.n_vocab = len(self.dataset_obj.vocab_obj.word_to_id)  # 词表大小

        print('model architecture param:')
        print('n_h:{}, n_embedding:{}, max_length:{}, n_vocab:{}'.format(self.n_h, self.n_embedding, self.max_length, self.n_vocab))
        print('-------------------------')

        dropout_rate1 = hp.Choice('dropout_rate1',
                      values=[0.1, 0.2, 0.4])

        dropout_rate2 = hp.Choice('dropout_rate2',
                      values=[0.1, 0.2, 0.4])

        dropout_rate3 = hp.Choice('dropout_rate3',
                      values=[0.1, 0.2, 0.4])

        dropout_rates = (dropout_rate1, dropout_rate2, dropout_rate3)

        image_caption = ImageCaptionV3(train_seq_length=self.max_length,
                                       infer_seq_length=self.max_length,
                                       n_h=self.n_h,
                                       n_image_feature=n_image_feature,
                                       n_embedding=self.n_embedding,
                                       n_vocab=self.n_vocab,
                                       dropout_rates=dropout_rates,
                                       vocab_obj=self.dataset_obj.vocab_obj,
                                       use_pretrain=False
                                       )

        model = image_caption.model_train

        model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return model




class TestV3:

    def test_training(self):

        # 1. 数据集的预处理, 运行 dataset_xrh.py 中的 DataPreprocess -> do_mian()

        dataset_obj = FlickerDataset(base_dir='dataset/', mode='train')

        # 打印预处理后的数据集的摘要信息
        print('preprocess dataset info:')
        print('N_train: {}, N_valid:{}, n_image_feature:{}, max_caption_length:{}'.format(dataset_obj.N_train, dataset_obj.N_valid, dataset_obj.feature_dim, dataset_obj.caption_length))
        print('-------------------------')


        # 2. 训练模型

        n_image_feature = dataset_obj.feature_dim
        max_caption_length = dataset_obj.caption_length
        # 取决于数据集, 在 dataset_xrh.py 中的 DataPreprocess -> do_mian(max_caption_length=40, freq_threshold=0) 中进行调整

        n_h = 512
        n_embedding = 512
        max_length = max_caption_length - 1

        # N_train = 32360  # 训练集样本个数
        n_vocab = len(dataset_obj.vocab_obj.word_to_id)  # 词表大小

        print('model architecture param:')
        print('n_h:{}, n_embedding:{}, max_length:{}, n_vocab:{}'.format(n_h, n_embedding, max_length, n_vocab))
        print('-------------------------')

        model_path = 'models/image_caption_ensemble_3lstm_hid_512_emb_512_thres_0_len_37.h5'  # 后缀只能使用 .h5

        image_caption = ImageCaptionV3(train_seq_length=max_length,
                                       infer_seq_length=max_length,
                                       n_h=n_h,
                                       n_image_feature=n_image_feature,
                                       n_embedding=n_embedding,
                                       n_vocab=n_vocab,
                                       vocab_obj=dataset_obj.vocab_obj,
                                       model_path=model_path,
                                       use_pretrain=False
                                       )
        # use_pretrain=True: 在已有的模型参数基础上, 进行更进一步的训练

        batch_size = 128
        epoch_num = 20

        train_data_generator = BatchDataGenSequence(n_h=n_h, n_embedding=n_embedding, n_vocab=n_vocab, batch_size=batch_size, dataset=dataset_obj.dataset['train'])
        valid_data_generator = BatchDataGenSequence(n_h=n_h, n_embedding=n_embedding, n_vocab=n_vocab, batch_size=batch_size, dataset=dataset_obj.dataset['valid'])


        image_caption.fit(train_data_generator, valid_data_generator, dataset_obj=dataset_obj, epoch_num=epoch_num, batch_size=batch_size)



    def test_evaluating(self):

        # 1. 数据集的预处理, 运行 dataset_xrh.py 中的 DataPreprocess 中的 do_mian()

        dataset_obj = FlickerDataset(base_dir='dataset/', mode='infer')

        n_image_feature = 2048
        max_caption_length = 37

        max_length = max_caption_length - 1  # 37 - 1 = 36
        n_embedding = 512
        n_h = 512
        n_vocab = len(dataset_obj.vocab_obj.word_to_id)  # 词表大小
        dropout_rates = (0.2, 0.1, 0.2)


        print('n_h:{}, n_embedding:{}, max_length:{}, n_vocab:{}'.format(n_h, n_embedding, max_length, n_vocab))

        # 2.模型推理

        # model_path = 'models/image_caption_ensemble_3lstm_hid_512_emb_512_thres_0_len_37.h5'

        model_path = 'models/cache/model.11-1.0095.h5'

        image_caption_infer = ImageCaptionV3(train_seq_length=max_length,
                                       infer_seq_length=max_length,
                                       n_h=n_h,
                                       n_image_feature=n_image_feature,
                                       n_embedding=n_embedding,
                                       n_vocab=n_vocab,
                                       vocab_obj=dataset_obj.vocab_obj,
                                       dropout_rates=dropout_rates,
                                       model_path=model_path,
                                       use_pretrain=True
                                       )

        image_caption_dict = dataset_obj.image_caption_dict

        image_dir_list = list(image_caption_dict.keys())

        m = 1619  # 测试数据集的图片个数

        image_dir_batch = image_dir_list[:m]

        print('test image num:{}'.format(len(image_dir_batch)))

        batch_image_feature = np.array(
            [list(image_caption_dict[image_dir]['feature']) for image_dir in image_dir_batch])

        references = [image_caption_dict[image_dir]['caption'] for image_dir in image_dir_batch]

        candidates = image_caption_infer.inference(batch_image_feature)

        # print('candidates: ', candidates[:10])
        # print('references: ', references[:10])

        print('\ncandidates:')
        for i in range(0,10):
            print(candidates[i])

        print('\nreferences:')
        for i in range(0,10):
            print(references[i])

        evaluate_obj = Evaluate()

        bleu_score, _ = evaluate_obj.evaluate_bleu(references, candidates)

        print('bleu_score:{}'.format(bleu_score))


    def test_turning(self):

        search_obj = SearchHyperParameter()

        search_obj.search_best_hps()

        search_obj.find_best_epoch_num(top_k=0)

if __name__ == '__main__':



    test = TestV3()

    # TODO: 每次实验前
    #  1. 更改最终模型存放的路径
    #  2. 运行脚本  clean_training_cache_file.bat

    test.test_training()

    # test.test_evaluating()

    # test.test_turning()

    # test.test_evaluating()