#!/usr/bin/python
# -*- coding: UTF-8 -*-

#  适用于 tensorflow >= 2.0, keras 被直接集成到 tensorflow 的内部
#  ref: https://keras.io/about/

from tensorflow.keras.layers import Input, LSTM, TimeDistributed
from tensorflow.keras.layers import Dense, Lambda, Embedding
from tensorflow.keras.optimizers import Adam


from build_dataset_xrh import *

from lib.evaluate_xrh import *

class ImageCaption:
    """

    基于 inceptionV3 + LSTM 的图片描述器 (image caption)

    1.使用预训练的 inceptionV3 将图片映射为 2048 维的向量, 在映射之前要对图片进行标准化,
    否则会导致生成出来的 图片表示不正确, 进而使得下游的 LSTM 训练不收敛

    2.将 embedding 后的图片向量 作为 LSTM 第一个时间步的隐状态 (hidden state)

    3. 在 LSTM 层中加入 mask, 即在输出序列中, 标为 <NULL> 的时间步, 我们不计入损失和梯度计算

    Author: xrh
    Date: 2021-9-25

    ref:
    1. Show and Tell: A Neural Image Caption Generator
    2. cs231n: http://cs231n.stanford.edu/slides/2021/lecture_10.pdf (第74页 - 第87页)
    3. https://github.com/yashk2810/Image-Captioning/

    """

    def __init__(self, max_length,
                 n_h, n_image_feature, n_embedding,
                 vocab_obj,
                 _null_str='<NULL>',
                 _start_str='<START>',
                 _end_str='<END>',
                 _unk_str='<UNK>',
                 use_pretrain=False,
                 model_path='models/image_caption_naive_lstm.h5'):
        """
        模型初始化

        :param max_length: lstm 编码器的长度, 该长度为 max_caption_length-1
        :param n_h: lstm 中隐藏状态的维度, n_h = 512
        :param n_image_feature: 图片经过 CNN 映射后的维度, n_image_feature = 2048
        :param n_embedding : 词向量维度, n_embedding= 256
        :param vocab_obj: 词典对象

        :param  _null_str: 末尾填充的空
        :param  _start_str: 句子的开始
        :param  _end_str: 句子的结束
        :param  _unk_str: 未登录词

        :param use_pretrain: 使用训练好的模型
        :param model_path: 预训练模型的路径

        """
        self.max_length = max_length
        self.n_h = n_h

        self.n_image_feature = n_image_feature
        self.n_embedding = n_embedding

        self.vocab_obj = vocab_obj

        self._null = self.vocab_obj.map_word_to_id(_null_str)  # 空
        self._start = self.vocab_obj.map_word_to_id(_start_str)  # 句子的开始
        self._end = self.vocab_obj.map_word_to_id(_end_str)  # 句子的结束
        self._unk_str = self.vocab_obj.map_word_to_id(_unk_str)

        self.model_path = model_path

        # 词表大小
        self.n_vocab = len(self.vocab_obj.word_to_id)  # 词表大小 9633

        # 对组成计算图的所有网络层进行声明和初始化
        self.__init_computation_graph()

        # 用于训练的计算图
        self.model_train = self.train_model(self.n_image_feature, self.max_length, self.n_h)

        # 用于推理的计算图
        self.infer_model = self.inference_model(self.n_image_feature, self.max_length, self.n_h)

        if use_pretrain:  # 载入训练好的模型

            self.model_train.load_weights(self.model_path)


    def __init_computation_graph(self):
        """
        对组成计算图的所有网络层进行声明和初始化

        :return:
        """

        # self.pict_embedding_layer = Dense(self.n_h, activation='relu', name='pict_embedding')  # 图片映射的维度必须 与 LSTM隐藏状态一致

        # self.pict_embedding_layer = Dense(self.n_h, kernel_initializer="glorot_normal", name='pict_embedding')  # 图片映射的维度必须 与 LSTM隐藏状态一致

        self.pict_embedding_layer = Dense(self.n_h, activation='relu',name='pict_embedding')  # 图片映射的维度必须 与 LSTM隐藏状态一致


        self.word_embedding_layer = Embedding(self.n_vocab, self.n_embedding, input_length=self.max_length,
                                              name='word_embedding')

        # self.lstm_layer = LSTM(self.n_h, return_sequences=True, return_state=True, kernel_initializer='glorot_normal',recurrent_initializer="orthogonal" ,name='lstm')

        self.lstm_layer = LSTM(self.n_h, return_sequences=True, return_state=True, name='lstm')

        # self.dense_layer = Dense(self.n_vocab, activation='softmax', kernel_initializer="glorot_normal", name='dense')

        self.dense_layer = Dense(self.n_vocab, activation='softmax', name='dense')


        self.output_layer = TimeDistributed(self.dense_layer, name='output')

        self.lambda_argmax = Lambda(TensorUtils.argmax_tensor, arguments={'axis': -1}, name='argmax_tensor')

        self.lambda_squezze = Lambda(K.squeeze, arguments={'axis': 1}, name='squezze_tensor')

        self.lambda_expand_dims = Lambda(K.expand_dims, arguments={'axis': 1}, name='expand_dims')

    def train_model(self, n_image_feature, max_length, n_h):
        """
        将各个 网络层(layer) 拼接为训练计算图

        :param n_image_feature:
        :param max_length:
        :param n_h:
        :return:
        """
        batch_caption = Input(shape=(max_length), name='input_caption')

        batch_image_feature = Input(shape=(n_image_feature), name='input_image_feature')

        pict_embedding = self.pict_embedding_layer(inputs=batch_image_feature)

        word_embedding = self.word_embedding_layer(inputs=batch_caption)

        h0 = pict_embedding  # hidden state

        c0 = Input(shape=(n_h), name='c0')  # cell state

        mask = (batch_caption != self._null)  # shape(N,max_length)
        # 因为训练时采用 mini-batch, 一个 batch 中的所有的 sentence 都是定长, 若有句子不够长度 则用 <null> 进行填充
        # 用 <null> 填充的时刻不能被计入损失中, 也不用求梯度

        out_lstm, state_h, state_c = self.lstm_layer(inputs=word_embedding, initial_state=[h0, c0], mask=mask)
        #  initial_state=[previous hidden state, previous cell state]

        outputs = self.output_layer(inputs=out_lstm)

        model = Model(inputs=[batch_caption, batch_image_feature, c0], outputs=outputs)

        return model

    def fit(self, data_generator, m, epoch_num=5, batch_size=64):
        """
        训练模型

        :param data_generator: 训练数据生成器
        :param m : 训练样本总数
        :param epoch_num: 模型训练的 epoch 个数,  一般训练集所有的样本模型都见过一遍才算一个 epoch
        :param batch_size: 选择 min-Batch梯度下降时, 每一次输入模型的样本个数 (默认 = 2048)

        :return:
        """

        # 打印 模型(计算图) 的所有网络层
        # print(model_train.summary())

        opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
        self.model_train.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        steps_per_epoch = m // batch_size

        history = self.model_train.fit_generator(data_generator(self.n_h, self.n_vocab, m, batch_size=batch_size),
                                                 steps_per_epoch=steps_per_epoch, epochs=epoch_num,
                                                 verbose=1)

        # 将训练好的模型保存到文件
        self.model_train.save(self.model_path)

        return history

    def inference_model(self, n_image_feature, max_length, n_h):
        """
        将各个 网络层(layer) 拼接为推理计算图

        :param n_image_feature:
        :param max_length:
        :param n_h:
        :return:
        """
        # 推理时 第一个时间步的输入为 '<start>'的标号
        batch_caption = Input(shape=(1), name='input_caption')  # shape: (m, 1) m - 当前批次的样本数

        batch_image_feature = Input(shape=(n_image_feature), name='input_image_feature')  # shape: (m, n_image_feature)

        pict_embedding = self.pict_embedding_layer(inputs=batch_image_feature)  # shape: (m, n_h)

        word_embedding = self.word_embedding_layer(inputs=batch_caption)  # shape: (m, 1, n_embedding)

        #     print(word_embedding)
        #    Tensor("word_embedding_2/Identity:0", shape=(None, 1, 300), dtype=float32)

        c0 = Input(shape=(n_h), name='c0')  # cell state, shape: (m, n_h)

        h0 = pict_embedding  # hidden state

        # 第一个时间步的输入
        h = h0
        c = c0

        inputs = word_embedding

        outputs = []

        for t in range(max_length):  # max_length 遍历所有的时间步

            out_lstm_one_step, h, c = self.lstm_layer(inputs=inputs, initial_state=[h, c])
            # out_lstm_one_step shape=(m, 1, 300)

            out_one_step = self.lambda_squezze(inputs=out_lstm_one_step)
            # out_one_step shape=(m, 300)

            out_dense = self.dense_layer(inputs=out_one_step)  # shape (m, n_vocab)

            max_idx = self.lambda_argmax(inputs=out_dense)  # shape (m, )

            out_idx = self.lambda_expand_dims(inputs=max_idx)  # shape (m, 1)

            inputs = self.word_embedding_layer(inputs=out_idx)  # shape=(m, 1, n_embedding)

            outputs.append(max_idx)

        # outputs shape ( max_length+1, m)
        model = Model(inputs=[batch_caption, batch_image_feature, c0], outputs=outputs)

        return model

    def inference(self, batch_image_feature):
        """
        使用训练好的模型进行推理

        :param batch_image_feature: 图片向量 shape (m,n_image_feature)
        :return:
        """

        m = np.shape(batch_image_feature)[0]  # 一个批次的样本的数量

        c0 = np.zeros((m, self.n_h))

        batch_caption = np.ones((m, 1)) * self._start  # 全部是 <START>

        preds = self.infer_model.predict([batch_caption, batch_image_feature, c0])  # preds shape (max_length+1, m)

        preds = np.array(preds)

        decode_result = preds.T  # shape (m, max_length+1)

        candidates = []

        # print(decode_result)

        for prediction in decode_result:
            output = ' '.join([self.vocab_obj.map_id_to_word(i) for i in prediction])
            candidates.append(output)

        return candidates




class Test:

    def test_training(self):
        # 1. 数据集的预处理, 运行 dataset_xrh.py 中的 DataPreprocess 中的 do_mian()

        vocab_obj = BuildVocab(load_vocab_dict=True)

        batch_data_generator = BatchDataGenerator()

        # 2. 训练模型

        n_image_feature = 2048

        max_caption_length = 30
        n_embedding = 512
        max_length = max_caption_length - 1  #
        n_h = 512

        m = 32360  # 训练集样本个数
        n_vocab = len(vocab_obj.word_to_id)  # 词表大小
        print('n_h:{}, n_embedding:{}, max_length:{}, n_vocab:{}'.format(n_h, n_embedding, max_length, n_vocab))

        model_path = 'models/image_caption_naive_lstm_hid_512_emb_512_thres_0_len_30.h5'  # 后缀只能使用 .h5

        image_caption = ImageCaption(max_length=max_length,
                                     n_h=n_h,
                                     n_image_feature=n_image_feature,
                                     n_embedding=n_embedding,
                                     vocab_obj=vocab_obj,
                                     model_path=model_path,
                                     use_pretrain=True
                                     )
        # use_pretrain=True: 在已有的模型参数基础上, 进行更进一步的训练

        image_caption.fit(batch_data_generator.read_all, m, epoch_num=50, batch_size=64)

    def test_inference(self):
        # 1. 数据集的预处理, 运行 dataset_xrh.py 中的 DataPreprocess 中的 do_mian()

        data_process_obj = DataPreprocess()

        vocab_obj = BuildVocab(load_vocab_dict=True)

        max_caption_length = 40
        n_image_feature = 2048
        n_embedding = 512
        max_length = max_caption_length - 1  # 40 - 1 = 39
        n_h = 512

        # 2.模型推理

        # 开启动态图模式, 进行调试
        tf.config.experimental_run_functions_eagerly(True)

        image_caption_infer = ImageCaption(max_length=max_length,
                                           n_h=n_h,
                                           n_image_feature=n_image_feature,
                                           n_embedding=n_embedding,
                                           vocab_obj=vocab_obj,
                                           use_pretrain=True
                                           )

        image_caption_dict = data_process_obj.load_image_caption_dict()

        image_dir_list = list(image_caption_dict.keys())

        # case1: 一张图片
        image_dir = image_dir_list[1]
        batch_image_feature = image_caption_dict[image_dir]['feature']  # shape(2048, )
        batch_image_feature = np.expand_dims(batch_image_feature, axis=0)  # shape(1,2048)

        candidates = image_caption_infer.inference(batch_image_feature)

        print('candidates: ', candidates)
        print('reference: ', image_caption_dict[image_dir]['caption'])
        im = Image.open(image_dir)
        im.show()

        # case2: 多张图片
        # image_dir_batch = image_dir_list[0:5]
        # batch_image_feature = np.array(
        #     [list(image_caption_dict[image_dir]['feature']) for image_dir in image_dir_batch])
        # references = [image_caption_dict[image_dir]['caption'] for image_dir in image_dir_batch]
        #
        # candidates = image_caption_infer.inference(batch_image_feature)
        #
        # print('candidates: ', candidates)
        # print('reference: ', references)


    def test_evaluate(self):


        # 1. 数据集的预处理, 运行 dataset_xrh.py 中的 DataPreprocess 中的 do_mian()

        data_process_obj = DataPreprocess()

        vocab_obj = BuildVocab(load_vocab_dict=True)

        n_image_feature = 2048

        max_caption_length = 30
        n_embedding = 512
        max_length = max_caption_length - 1  # 40 - 1 = 39
        n_h = 512
        n_vocab = len(vocab_obj.word_to_id)  # 词表大小
        
        print('n_h:{}, n_embedding:{}, max_length:{}, n_vocab:{}'.format(n_h, n_embedding, max_length, n_vocab))

        # 2.模型推理

        # 开启动态图模式, 进行调试
        # tf.config.experimental_run_functions_eagerly(True)

        model_path = 'models/image_caption_naive_lstm_hid_512_emb_512_thres_0_len_30.h5'

        image_caption_infer = ImageCaption(max_length=max_length,
                                           n_h=n_h,
                                           n_image_feature=n_image_feature,
                                           n_embedding=n_embedding,
                                           vocab_obj=vocab_obj,
                                           model_path=model_path,
                                           use_pretrain=True
                                           )

        image_caption_dict = data_process_obj.load_image_caption_dict()

        image_dir_list = list(image_caption_dict.keys())

        m = 1619  # 测试数据集的图片个数

        image_dir_batch = image_dir_list[:m]

        print('test image num:{}'.format(len(image_dir_batch)))

        batch_image_feature = np.array(
            [list(image_caption_dict[image_dir]['feature']) for image_dir in image_dir_batch])

        references = [image_caption_dict[image_dir]['caption'] for image_dir in image_dir_batch]

        candidates = image_caption_infer.inference(batch_image_feature)

        print('candidates: ', candidates[:10])
        print('reference: ', references[:10])

        evaluate_obj = Evaluate()

        bleu_score, _ = evaluate_obj.evaluate_bleu(references, candidates)

        print('bleu_score:{}'.format(bleu_score))

if __name__ == '__main__':
    test = Test()

    # test.test_training()

    # test.test_inference()

    test.test_evaluate()