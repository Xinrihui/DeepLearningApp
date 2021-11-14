#!/usr/bin/python
# -*- coding: UTF-8 -*-

#  适用于 tensorflow >= 2.0, keras 被直接集成到 tensorflow 的内部
#  ref: https://keras.io/about/


from tensorflow.keras.layers import Layer, Input, LSTM, TimeDistributed, Bidirectional,Dense, Lambda, Embedding, Dropout, Concatenate, RepeatVector
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import tensorflow.keras as keras

from tensorflow.keras.models import Model

from lib.evaluate_xrh import *
from lib.tf_data_utils_xrh import *

# from deprecated import deprecated


# import keras_tuner as kt


class CheckoutCallback(keras.callbacks.Callback):
    """
    回调函数, 实现在每一次 epoch 后 checkout 训练好的模型,
    并且计算在验证集上的 bleu 分数

    """

    def __init__(self, model_train, model_infer, vocab_obj, batch_size, valid_image_caption_dict, checkpoint_models_path):

        keras.callbacks.Callback.__init__(self)

        self.model_train = model_train
        self.model_infer = model_infer
        self.vocab_obj = vocab_obj

        self.batch_image_feature, self.references = self.prepare_valid_data(batch_size, valid_image_caption_dict)

        self.evaluate_obj = Evaluate()

        self.checkpoint_models_path = checkpoint_models_path

    def prepare_valid_data(self, batch_size, valid_image_caption_dict):
        """
        返回 图片的 embedding 向量 和 图片对应的 caption

        :param batch_size:
        :param valid_image_caption_dict:
        :return:
        """

        image_dir_list = list(valid_image_caption_dict.keys())

        image_dirs = image_dir_list[:]

        print('valid image num:{}'.format(len(image_dirs)))

        image_feature = np.array(
            [list(valid_image_caption_dict[image_dir]['feature']) for image_dir in image_dirs])

        references = [valid_image_caption_dict[image_dir]['caption'] for image_dir in image_dirs]

        image_feature_datset = tf.data.Dataset.from_tensor_slices(image_feature)

        batch_image_feature = image_feature_datset.batch(batch_size)

        return batch_image_feature, references

    def inference_bleu(self):
        """
        使用验证数据集进行推理, 并计算 bleu

        :return:
        """

        # batch_image_feature: 图片向量 shape (N_batch,n_image_feature)
        preds = self.model_infer.predict(self.batch_image_feature)

        decode_result = np.array(preds)  # shape (N_batch, infer_seq_length)

        candidates = []
        for prediction in decode_result:
            output = ' '.join([self.vocab_obj.map_id_to_word(i) for i in prediction])
            candidates.append(output)

        bleu_score, _ = self.evaluate_obj.evaluate_bleu(self.references, candidates)

        print()
        print('bleu_score:{}'.format(bleu_score))


    def on_epoch_end(self, epoch, logs=None):

        # checkout 模型
        fmt = self.checkpoint_models_path + 'model.%02d-%.4f.h5'
        self.model_train.save(fmt % (epoch, logs['val_loss']))

        # 计算 bleu 分数
        self.inference_bleu()



class ImageCaptionV4:
    """

    基于 LSTM + attention 的图片描述生成器

    1. 实现了 soft-attention

    2. 使用 inceptionV3 的倒数第 2层的特征图 shape(8,8,2048)  作为图片的 embedding
       使用 VGG-16 的倒数第 6层的特征图 shape(14,14,512)  作为图片的 embedding

    4. 实现了双重注意力机制和正则化

    5. 实现了基于 bleu score 的训练早停


    Author: xrh
    Date: 2021-10-15

    ref:
    1. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
    2. https://tensorflow.google.cn/tutorials/text/image_captioning
    3. https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning(训练细节)

    """

    def __init__(self, train_seq_length, infer_seq_length,
                 n_h, n_image_feature, n_embedding, n_vocab,
                 vocab_obj,
                 dropout_rates=(0.5),
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
        self._unk_str = self.vocab_obj.map_word_to_id(_unk_str)  # 未登录词

        self.model_path = model_path

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

        self.encoder = CNN_Encoder(self.n_embedding)

        self.trian_decoder = trian_LSTM_Decoder(self.infer_seq_length, self.n_embedding, self.n_h, self.n_vocab)

        self.infer_decoder = infer_LSTM_Decoder(self.trian_decoder, self._start)


    def train_model(self):
        """
        将各个 网络层(layer) 拼接为训练计算图

        :return:
        """

        batch_caption_in = Input(shape=(self.infer_seq_length), name='batch_caption_in')

        batch_image_feature = Input(shape=(self.train_seq_length, self.n_image_feature),
                                    name='batch_image_feature')  # shape (N_batch, train_seq_length, n_image_feature)

        batch_image_embbeding = self.encoder(batch_image_feature)  # shape (N_batch, train_seq_length, n_embedding)

        outputs = self.trian_decoder(batch_caption_in, batch_image_embbeding)

        model = Model(inputs=[batch_image_feature, batch_caption_in], outputs=outputs)

        return model

    def infer_model(self):
        """
        将各个 网络层(layer) 拼接为推理计算图

        :return:
        """

        batch_image_feature = Input(shape=(self.train_seq_length, self.n_image_feature),
                                    name='batch_image_feature')  # shape (N_batch, train_seq_length, n_image_feature)

        batch_image_embbeding = self.encoder(batch_image_feature)  # shape (N_batch, train_seq_length, n_embedding)

        outputs = self.infer_decoder(batch_image_embbeding)

        model = Model(inputs=[batch_image_feature], outputs=outputs)

        return model

    def fit(self, train_dataset, valid_dataset, valid_image_caption_dict, epoch_num=20,
            N_trian=32360, N_valid = 8095,
            batch_size=128, buffer_size=2000,
            ):
        """
        使用内置方法训练模型

        :param train_dataset: 训练数据生成器
        :param valid_dataset: 验证数据生成器
        :param valid_image_caption_dict: 验证数据的字典, 用于计算 bleu

        :param epoch_num: 模型训练的 epoch 个数,  一般训练集所有的样本模型都见过一遍才算一个 epoch
        :param batch_size: 选择 min-Batch梯度下降时, 每一次输入模型的样本个数 (默认 = 128)
        :param buffer_size:

        :return:
        """

        # 打印 模型(计算图) 的所有网络层
        print(self.model_train.summary())

        # 输出训练计算图的图片
        # plot_model(self.model_train, to_file='docs/images/train_model.png')

        # tf.data 的数据混洗,分批和预取

        # Shuffle and batch
        train_dataset_batch = train_dataset.shuffle(buffer_size).batch(batch_size)
        train_dataset_prefetch = train_dataset_batch.prefetch(
            buffer_size=tf.data.AUTOTUNE)  # 要预取的元素数量应等于（或大于）单个训练步骤 epoch 消耗的批次数量

        valid_dataset_batch = valid_dataset.shuffle(buffer_size).batch(batch_size)
        valid_dataset_prefetch = valid_dataset_batch.prefetch(buffer_size=tf.data.AUTOTUNE)

        # 自定义的损失函数
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')

        def loss_function(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, self._null))
            loss_ = loss_object(real, pred)
            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask

            return tf.reduce_mean(loss_)

        checkpoint_models_path = 'models/cache/'

        # Callbacks
        # 在根目录下运行 tensorboard --logdir ./logs
        tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True,
                                                   write_images=True)

        model_names = checkpoint_models_path + 'model.{epoch:02d}-{val_loss:.4f}.h5'

        # 模型持久化: 若某次 epcho 模型在 验证集上的损失比之前的最小损失小, 则将模型作为最佳模型持久化
        # model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=False)

        model_checkpoint_with_eval = CheckoutCallback(model_train=self.model_train, model_infer=self.model_infer,
                                                      vocab_obj=self.vocab_obj, batch_size=batch_size,
                                                      valid_image_caption_dict=valid_image_caption_dict,
                                                      checkpoint_models_path=checkpoint_models_path)

        # 早停: 在验证集上, 损失经过 patience 次的迭代后, 仍然没有下降则暂停训练
        early_stop = EarlyStopping('val_loss', patience=10)

        # optimizer = tf.keras.optimizers.Adam(learning_rate=4e-4)

        optimizer = 'rmsprop'

        self.model_train.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

        # Final callbacks
        callbacks = [model_checkpoint_with_eval, early_stop, tensor_board]

        #  N_train : 训练样本总数
        #  N_valid : 验证样本总数

        # 查看会话管理可知需要8GB显存
        history = self.model_train.fit(
            x=train_dataset_prefetch,
            epochs=epoch_num,
            validation_data=valid_dataset_prefetch,
            verbose=1,
            callbacks=callbacks)

 # 数据集会在每个周期结束时重置，因此可以在下一个周期重复使用。
 # 如果只想在来自此数据集的特定数量批次上进行训练，则可以传递 steps_per_epoch 参数，此参数可以指定在继续下一个周期之前，模型应使用此数据集运行多少训练步骤。
 # 如果执行此操作，则不会在每个周期结束时重置数据集，而是会继续绘制接下来的批次。数据集最终将用尽数据（除非它是无限循环的数据集）。

        # history = self.model_train.fit(
        #     x=train_dataset_prefetch.repeat(),
        #     steps_per_epoch=N_trian // batch_size,
        #     epochs=epoch_num,
        #     validation_data=valid_dataset_prefetch.repeat(),
        #     validation_steps=N_valid // batch_size,
        #     verbose=1,
        #     callbacks=callbacks)

        # 将训练好的模型保存到文件
        self.model_train.save(self.model_path)

        return history

    def inference(self, batch_image_feature):
        """
        使用训练好的模型进行推理

        :param batch_image_feature: 图片向量 shape (N_batch,n_image_feature)
        :return:
        """

        preds = self.model_infer.predict(batch_image_feature)

        decode_result = np.array(preds)  # shape (N_batch, infer_seq_length)

        candidates = []

        # print(decode_result)

        for prediction in decode_result:
            output = ' '.join([self.vocab_obj.map_id_to_word(i) for i in prediction])
            candidates.append(output)

        return candidates

class OutputLayer(Layer):
    """
    输出层

    参考论文 Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
    3.1.2. DECODER: LONG SHORT-TERM MEMORY NETWORK
    公式(2)

    """

    def __init__(self, n_embedding, n_vocab):

        super(OutputLayer, self).__init__()

        self.dense_L_h_layer = Dense(n_embedding)
        self.dense_L_z_layer = Dense(n_embedding)
        self.dense_L_o_layer = Dense(n_vocab, activation='softmax')

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dense_L_h_layer': self.dense_L_h_layer,
            'dense_L_z_layer': self.dense_L_z_layer,
            'dense_L_o_layer': self.dense_L_o_layer
        })
        return config

    def call(self, h, z, y_emb):

        out = self.dense_L_o_layer(
            y_emb + self.dense_L_h_layer(h) + self.dense_L_z_layer(z))  # shape (N_batch, n_vocab)

        # out = tf.nn.softmax(out, axis=-1)

        return out


class OneStepAttention(Layer):
    """
    单步的注意力机制

    """

    def __init__(self, n_h):
        super(OneStepAttention, self).__init__()

        self.dense_W_a_layer = Dense(n_h)
        self.dense_U_a_layer = Dense(n_h)
        self.dense_V_a_layer = Dense(1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dense_W_a_layer': self.dense_W_a_layer,
            'dense_U_a_layer': self.dense_U_a_layer,
            'dense_V_a_layer': self.dense_V_a_layer
        })
        return config

    def call(self, h_list, s_prev):
        # h_list CNN 编码器的输出 shape ( N_batch, train_seq_length, n_embedding)
        # s_prev 解码器上一步的隐状态 shape (N_batch, n_h)

        s_prev_time = tf.expand_dims(s_prev, 1)  # shape (N_batch, 1, n_h)

        # 参考论文 《NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE》
        # 附录 A.2.2 DECODER

        # 对齐模型
        e = self.dense_V_a_layer(tf.nn.relu(
            self.dense_W_a_layer(s_prev_time) + self.dense_U_a_layer(h_list)))  # shape (N_batch, train_seq_length, 1)

        # 对 h_list 的注意力权重
        alpha = tf.nn.softmax(e, axis=1)  # shape (N_batch, train_seq_length, 1)

        # context_vector
        c = alpha * h_list  # shape (N_batch, train_seq_length, n_embedding)

        c = tf.reduce_sum(c, axis=1)  # shape (N_batch, n_embedding)

        alpha = tf.squeeze(alpha, axis=2)  # shape (N_batch, train_seq_length)
        #         print('alpha', alpha)

        return c, alpha


class CNN_Encoder(Layer):
    """
    基于 CNN 的编码器

    CNN 对图像进行抽取后输出的 Feature Map 的 shape 为 (N_batch, train_seq_length, n_image_feature) ,
    我们要将其投影为 shape  (N_batch, train_seq_length, n_embedding)

    """

    def __init__(self, n_embedding):
        super(CNN_Encoder, self).__init__()

        # 全连接层 fc
        self.fc_layer = Dense(n_embedding, activation='relu')  # shape (N_batch, train_seq_length, n_embedding)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'fc_layer': self.fc_layer
        })
        return config

    def call(self, x):
        x = self.fc_layer(x)
        return x


class trian_LSTM_Decoder(Layer):
    """
    训练模式下的基于 LSTM 的解码器

    """


    def __init__(self, infer_seq_length, n_embedding, n_h, n_vocab, lambda_alpha=1.0):

        super(trian_LSTM_Decoder, self).__init__()

        self.infer_seq_length = infer_seq_length  # 解码器的长度
        self.n_embedding = n_embedding
        self.n_h = n_h
        self.n_vocab = n_vocab
        self.lambda_alpha = lambda_alpha

        self.embedding_layer = Embedding(n_vocab, self.n_embedding)
        self.fc_init_c_layer = Dense(self.n_h)  # fc 全连接层(full connected)
        self.fc_init_h_layer = Dense(self.n_h)

        self.one_step_attention_layer = OneStepAttention(self.n_h)

        self.beta_gate = Dense(self.n_embedding, activation='sigmoid')

        self.lstm_layer = LSTM(self.n_h, return_state=True)

        self.dropout_layer = Dropout(0.5)

        self.fc_out_layer = Dense(self.n_vocab, activation='softmax')

        # self.fc_out_layer = OutputLayer(n_embedding, n_vocab)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'infer_seq_length': self.infer_seq_length,
            'n_embedding': self.n_embedding,
            'n_h': self.n_h,
            'n_vocab': self.n_vocab,
            'lambda_alpha': self.lambda_alpha,

            'embedding_layer': self.embedding_layer,
            'fc_init_c_layer': self.fc_init_c_layer,
            'fc_init_h_layer': self.fc_init_h_layer,
            'one_step_attention_layer': self.one_step_attention_layer,
            'beta_gate': self.beta_gate,
            'lstm_layer': self.lstm_layer,
            'dropout_layer': self.dropout_layer,
            'fc_out_layer': self.fc_out_layer,

        })
        return config

    def call(self, batch_caption_in, batch_image_embbeding):
        # batch_caption_in shape (N_batch, infer_seq_length)
        # batch_image_embbeding shape (N_batch, train_seq_length, n_embedding)

        batch_caption_embbeding = self.embedding_layer(
            batch_caption_in)  # shape (N_batch, infer_seq_length, n_embedding)

        N_batch = tf.shape(batch_caption_in)[0]

        mean_a = tf.math.reduce_mean(batch_image_embbeding, axis=1)  # shape (N_batch, n_embedding)

        h = self.fc_init_h_layer(mean_a)  # shape: (N_batch, n_h)
        c = self.fc_init_c_layer(mean_a)  # shape: (N_batch, n_h)

        alpha_list = []
        outs = []

        for t in range(self.infer_seq_length):
            batch_token_embbeding = tf.expand_dims(batch_caption_embbeding[:, t, :], axis=1)
            # Teacher Forcing: 每一个时间步的输入为真实的标签值而不是上一步预测的结果
            # batch_token_embbeding shape (N_batch, 1, n_embedding)

            z, alpha = self.one_step_attention_layer(h_list=batch_image_embbeding, s_prev=h)
            # h_list CNN 编码器的输出 shape ( N_batch, train_seq_length, n_embedding)
            # s_prev 解码器上一步的隐状态 shape (N_batch, n_h)
            # z shape (N_batch, n_embedding)
            # alpha shape (N_batch, train_seq_length)

            alpha_list.append(alpha)

            # 双重注意力机制的门
            z = z * self.beta_gate(h)

            context = tf.concat([tf.expand_dims(z, axis=1), batch_token_embbeding], axis=-1)
            #  shape (N_batch, 1, n_embedding + n_embedding)

            out_lstm, h, c = self.lstm_layer(inputs=context, initial_state=[h, c])  # 输入 context 只有1个时间步

            h_dropout = self.dropout_layer(h)

            out = self.fc_out_layer(h_dropout)  # shape (N_batch, n_vocab)

            # out = self.fc_out_layer(h=h_dropout, z=z, y_emb=tf.squeeze(batch_token_embbeding, axis=1))
            # h shape (N_batch, n_h), z shape (N_batch, n_h)
            # batch_token_embbeding shape (N_batch, 1, n_embedding) -> shape (N_batch, n_embedding)

            outs.append(out)  # shape (infer_seq_length, N_batch, n_vocab)

        outputs = tf.transpose(outs, perm=[1, 0, 2])  # shape (N_batch, infer_seq_length, n_vocab)

        alpha_list = tf.transpose(alpha_list, perm=[1, 0, 2])  # shape (N_batch, infer_seq_length, train_seq_length)

        # 双重注意力机制中的正则化项
        loss_reg = self.lambda_alpha * tf.math.reduce_mean(((1. - tf.math.reduce_sum(alpha_list, axis=1)) ** 2))

        self.add_loss(loss_reg)

        return outputs


class infer_LSTM_Decoder(Layer):
    """
    推理模式下的基于 LSTM 的解码器

    """

    def __init__(self, train_decoder_obj, _start):

        super(infer_LSTM_Decoder, self).__init__()

        self.train_decoder_obj = train_decoder_obj
        self._start = _start

        self.infer_seq_length = self.train_decoder_obj.infer_seq_length
        self.n_embedding = self.train_decoder_obj.n_embedding
        self.n_h = self.train_decoder_obj.n_h
        self.n_vocab = self.train_decoder_obj.n_vocab
        self.lambda_alpha = self.train_decoder_obj.lambda_alpha

        self.embedding_layer = self.train_decoder_obj.embedding_layer
        self.fc_init_c_layer = self.train_decoder_obj.fc_init_c_layer  # fc 全连接层(full connected)
        self.fc_init_h_layer = self.train_decoder_obj.fc_init_h_layer
        self.one_step_attention_layer = self.train_decoder_obj.one_step_attention_layer

        self.beta_gate = self.train_decoder_obj.beta_gate
        self.lstm_layer = self.train_decoder_obj.lstm_layer

        self.dropout_layer = self.train_decoder_obj.dropout_layer
        self.fc_out_layer = self.train_decoder_obj.fc_out_layer

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'infer_seq_length': self.infer_seq_length,
            'n_embedding': self.n_embedding,
            'n_h': self.n_h,
            'n_vocab': self.n_vocab,
            'lambda_alpha': self.lambda_alpha,

            'embedding_layer': self.embedding_layer,
            'fc_init_c_layer': self.fc_init_c_layer,
            'fc_init_h_layer': self.fc_init_h_layer,
            'one_step_attention_layer': self.one_step_attention_layer,
            'beta_gate': self.beta_gate,
            'lstm_layer': self.lstm_layer,
            'dropout_layer': self.dropout_layer,
            'fc_out_layer': self.fc_out_layer,

        })
        return config

    def call(self, batch_image_embbeding):
        # batch_image_embbeding shape (N_batch, train_seq_length, n_embedding)

        N_batch = tf.shape(batch_image_embbeding)[0]

        batch_token = tf.ones((N_batch, 1)) * self._start  # (N_batch, 1)

        #         print('batch_token:', batch_token)

        mean_a = tf.math.reduce_mean(batch_image_embbeding, axis=1)  # shape (N_batch, n_embedding)

        h = self.fc_init_h_layer(mean_a)  # shape: (N_batch, n_h)
        c = self.fc_init_c_layer(mean_a)  # shape: (N_batch, n_h)

        outs = []

        for t in range(self.infer_seq_length):

            batch_token_embbeding = self.embedding_layer(batch_token)  # shape (N_batch, 1, n_embedding)

            #             print('batch_token_embbeding:', batch_token_embbeding)

            z, alpha = self.one_step_attention_layer(h_list=batch_image_embbeding, s_prev=h)
            # h_list CNN 编码器的输出 shape ( N_batch, train_seq_length, n_embedding)
            # s_prev 解码器上一步的隐状态 shape (N_batch, n_h)
            # z shape (N_batch, n_embedding)
            #

            #             print('z:', z)

            z = z * self.beta_gate(h)

            context = tf.concat([tf.expand_dims(z, axis=1), batch_token_embbeding], axis=-1)
            #  shape (N_batch, 1, n_embedding + n_embedding)

            #             print('context:', context)

            out_lstm, h, c = self.lstm_layer(inputs=context, initial_state=[h, c])  # 输入 context 只有1个时间步

            h_dropout = self.dropout_layer(h)

            out_fc = self.fc_out_layer(h_dropout)

            # out_fc = self.fc_out_layer(h=h_dropout, z=z, y_emb=tf.squeeze(batch_token_embbeding, axis=1))
            # h shape (N_batch, n_h), z shape (N_batch, n_h)
            # batch_token_embbeding shape (N_batch, 1, n_embedding) -> shape (N_batch, n_embedding)


            #print('out_fc:', out_fc)

            max_idx = tf.math.argmax(out_fc, axis=1)  # shape (N_batch, )

            #             print('max_idx', max_idx)

            batch_token = tf.expand_dims(max_idx, axis=1)  # shape (N_batch, 1)

            outs.append(max_idx)  # shape (infer_seq_length, N_batch)

        outputs = tf.transpose(outs, perm=[1, 0])  # shape (N_batch, infer_seq_length)

        return outputs






class TestV3:

    def test_training(self):

        # 1. 数据集的预处理, 运行 tf_data_utils_xrh.py 中的 DataPreprocess -> do_mian()

        dataset_obj = FlickerDataset(base_dir='dataset/', mode='train')

        # infer_dataset_obj = FlickerDataset(base_dir='dataset/', mode='infer')

        # 2. 训练模型

        # Feature Map 的维度
        n_image_feature = 2048
        # n_image_feature = 512

        # Feature Map 的像素点的个数(论文中的 L), 即编码器的长度
        train_seq_length = 64
        # train_seq_length = 196

        # caption 的长度 -1 , 即解码器的长度
        infer_seq_length = 36

        # 编码器图片嵌入的维度, 词嵌入的维度
        n_embedding = 512

        # 解码器的隐状态维度
        n_h = 512

        # 词表大小
        n_vocab = 8868

        # N_train = 32360  # 训练集样本个数
        # N_valid = 8095  # 验证集样本个数

        N_train = 30000  # 训练集样本个数
        N_valid = 5000  # 验证集样本个数

        print('model architecture param:')
        print('n_h:{}, n_embedding:{}, n_vocab:{}, train_seq_length:{}, infer_seq_length:{}'.format(n_h, n_embedding, n_vocab, train_seq_length, infer_seq_length))
        print('-------------------------')

        model_path = 'models/image_caption_attention_model.h5'  # 后缀只能使用 .h5

        image_caption = ImageCaptionV4(train_seq_length=train_seq_length,
                                       infer_seq_length=infer_seq_length,
                                       n_h=n_h,
                                       n_image_feature=n_image_feature,
                                       n_embedding=n_embedding,
                                       n_vocab=n_vocab,
                                       vocab_obj=dataset_obj.vocab,
                                       model_path=model_path,
                                       use_pretrain=False
                                       )
        # use_pretrain=True: 在已有的模型参数基础上, 进行更进一步的训练

        batch_size = 256
        epoch_num = 15

        image_caption.fit(train_dataset=dataset_obj.train_dataset, valid_dataset=dataset_obj.valid_dataset,
                          valid_image_caption_dict=dataset_obj.valid_image_caption_dict,
                          N_trian=N_train, N_valid=N_valid,
                          epoch_num=epoch_num, batch_size=batch_size)



    def test_evaluating(self):

        # 1. 数据集的预处理, 运行 tf_data_utils_xrh.py 中的 DataPreprocess -> do_mian()

        dataset_obj = FlickerDataset(base_dir='dataset/', mode='infer')

        # Feature Map 的维度
        n_image_feature = 2048
        # n_image_feature = 512

        # Feature Map 的像素点的个数(论文中的 L), 即编码器的长度
        train_seq_length = 64
        # train_seq_length = 196

        # caption 的长度 -1 , 即解码器的长度
        infer_seq_length = 36

        # 编码器图片嵌入的维度, 词嵌入的维度
        n_embedding = 512

        # 解码器的隐状态维度
        n_h = 512

        # 词表大小
        n_vocab = 8868

        print('model architecture param:')
        print('n_h:{}, n_embedding:{}, n_vocab:{}, train_seq_length:{}, infer_seq_length:{}'.format(n_h, n_embedding, n_vocab, train_seq_length, infer_seq_length))
        print('-------------------------')

        # 2.模型推理

        # model_path = 'models/image_caption_attention_model.h5'

        model_path = 'models/cache/model.08-1.2417.h5'

        image_caption_infer = ImageCaptionV4(train_seq_length=train_seq_length,
                                       infer_seq_length=infer_seq_length,
                                       n_h=n_h,
                                       n_image_feature=n_image_feature,
                                       n_embedding=n_embedding,
                                       n_vocab=n_vocab,
                                       vocab_obj=dataset_obj.vocab,
                                       model_path=model_path,
                                       use_pretrain=True
                                       )

        test_image_caption_dict = dataset_obj.test_image_caption_dict

        image_dir_list = list(test_image_caption_dict.keys())

        # m = 1619  # 测试数据集的图片个数

        image_dirs = image_dir_list[:]

        print('test image num:{}'.format(len(image_dirs)))

        image_feature = np.array(
            [list(test_image_caption_dict[image_dir]['feature']) for image_dir in image_dirs])

        print('image_feature shape: ', np.shape(image_feature))

        image_feature_datset = tf.data.Dataset.from_tensor_slices(image_feature)

        image_feature_datset_batch = image_feature_datset.batch(64)

        references = [test_image_caption_dict[image_dir]['caption'] for image_dir in image_dirs]

        candidates = image_caption_infer.inference(image_feature_datset_batch)

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


    test = TestV3()

    # TODO: 每次实验前
    #  1. 更改最终模型存放的路径
    #  2. 运行脚本  clean_training_cache_file.bat

    test.test_training()

    # test.test_evaluating()

