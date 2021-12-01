#!/usr/bin/python
# -*- coding: UTF-8 -*-

#  适用于 tensorflow >= 2.0, keras 被直接集成到 tensorflow 的内部
#  ref: https://keras.io/about/

from tensorflow.keras.layers import Bidirectional, Concatenate, Dot, Input, LSTM
from tensorflow.keras.layers import RepeatVector, Dense, Lambda, Softmax, Reshape

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from tensorflow.keras.utils import plot_model

from src.attention.v1.lib.nmt_utils import *
from src.attention.v1.lib.utils_xrh import *

from sklearn.model_selection import train_test_split

from lib.bleu_xrh import *

from deprecated import deprecated

@deprecated(version='0.0', reason="You should use class MachineTranslationV1")
class BasicMachineTranslation:
    """
    基于 LSTM + seq2seq + attention 的翻译模型
    基础的面向过程实现
    Author: xrh
    Date: 2019-12-16
    ref: https://github.com/enggen/Deep-Learning-Coursera
    """

    def model_implementation_naive(self, Tx, Ty, machine_vocab, human_vocab, Xoh, Yoh, m):
        # 对于 attention 模块, 使用全局的 网络层(Keras Layers) 对象，以在多个 model 中共享权重
        repeator = RepeatVector(Tx)
        concatenator = Concatenate(axis=2)
        densor = Dense(1, activation="relu")
        activator = Softmax(axis=1)
        dotor = Dot(axes=1)

        def one_step_attention(a, s_prev):  # 与RNN 类似，是一个 循环结构
            """
            attention 模块的实现
            使用全局的 网络层(Keras Layers) 对象，以在多个 model 中共享权重
            :param a: encoder-lstm 所有时间步的输出 shape (m, Tx, 2*n_a)
            :param s_prev: decoder-lstm 中上一个时间步的 隐藏状态 shape (m, n_s)
            :return:
            """
            s_prev = repeator(s_prev)
            concat = concatenator([a, s_prev])  # shape: (m, Tx, 2*n_a+n_s)
            e = densor(concat)  # shape: (m, Tx, 1)
            alphas = activator(e)  # shape:  (m, Tx, 1)
            context = dotor([alphas, a])  # shape: (m, 1, 2*n_a)

            return context

        n_a = 64
        n_s = 128

        pre_activation_LSTM_cell = Bidirectional(LSTM(n_a, return_sequences=True, return_state=True))
        # ref: https://keras.io/api/layers/recurrent_layers/lstm/#lstm-class
        # ref: https://keras.io/api/layers/recurrent_layers/bidirectional/
        post_activation_LSTM_cell = LSTM(n_s, return_state=True)
        output_layer = Dense(len(machine_vocab), activation='softmax')

        def model(Tx, Ty, n_a, n_s, human_vocab_size):
            """
            实现 encoder 和 decoder
            :param Tx: 输入序列的长度
            :param Ty: 输出序列的长度
            :param n_a: encoder-lstm 中隐藏状态的维度
            :param n_s: decoder-lstm 中隐藏状态的维度
            :param human_vocab_size: 输入序列的字典大小
            :return: Keras model instance
            """

            # 定义模型的输入
            X = Input(shape=(Tx, human_vocab_size))  # shape: (m,Tx,human_vocab_size) , m- batch 大小

            #  decoder-LSTM 的初始状态
            # 隐藏状态向量 hidden state
            s0 = Input(shape=(n_s,), name='s0')  # shape of s:  (m, n_s=64)
            # 细胞状态向量
            c0 = Input(shape=(n_s,), name='c0')  # shape of c:  (m, n_s=64)

            s = s0
            c = c0

            outputs = []

            # lstm-encoder 实现
            a, forward_h, forward_c, backward_h, backward_c = pre_activation_LSTM_cell(
                inputs=X)  # shape of a : (m,Tx, 2*n_a)

            # lstm-decoder 实现
            # 对Ty 时间步进行展开，由于每次只运行一个 时间步 所以要输出 hidden state 和 cell state 作为下一个时间步的LSTM的输入
            for t in range(Ty):
                context = one_step_attention(a, s)  # shape: (m, 1, 2*n_a)
                # shape of s: (m, n_s)

                s, _, c = post_activation_LSTM_cell(inputs=context, initial_state=[s, c])  # 输入只有一个时间步

                out = output_layer(s)  # shape: (m, machine_vocab)

                outputs.append(out)

            model = Model(inputs=[X, s0, c0], outputs=outputs)
            # shape of outs : ( Ty, m, machine_vocab)

            return model

        model = model(Tx, Ty, n_a, n_s, len(human_vocab))

        opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        # tips :
        # n_a = 64
        # n_s = 128
        # m = 10000

        outputs = list(Yoh.swapaxes(0, 1))  # Yoh.swapaxes(0,1) 第0维度 和 第1 维度交换，原来为(m,T_y,11) 变换后 为：(T_y,m,11)

        s0 = np.zeros((m, n_s))
        c0 = np.zeros((m, n_s))

        history = model.fit([Xoh, s0, c0], outputs, epochs=40, batch_size=2048, validation_split=0.1)


class MachineTranslationV1:
    """
    基于 LSTM + seq2seq + attention 的翻译模型

    对比 class BasicMachineTranslation 改进如下:

    1. 面向对象实现

    2. 更改了 attention 机制
       (1) 对于 decoder, 第一个时间步的输入的 隐藏状态向量(s) 和 细胞状态向量(c) 为 encoder 最后一个时间步的输出
       (2) 对于 decoder, 每一个时间步的输入除了 attention 模块对 encoder 所有时间步的输出的加权和 之外,
           还引入了 decoder 上一个时间步的输出

    3. 基于 beamsearch 的推理
       (1) 实现了 encoder 和 decoder 的解耦, 提升推理速度
       (2) decoder 采用分时间步解码的方式, 即每一个时间步做一次模型的推理得到解码结果

    Author: xrh
    Date: 2019-12-16
    ref:
    1.https://github.com/enggen/Deep-Learning-Coursera
    2.论文 Neural machine translation by jointly learning to align and translate
    """

    def __init__(self, Tx, Ty, n_a, n_s, machine_vocab, inv_machine_vocab, human_vocab,
                 use_pretrain=False, model_path='models/lstm_seq2seq_attention.h5'):
        """
        模型初始化
        :param Tx: 输入序列的长度
        :param Ty: 输出序列的长度
        :param n_a: encoder-lstm 中隐藏状态的维度 n_a = 64
        :param n_s: decoder-lstm 中隐藏状态的维度 n_s = 128
        :param machine_vocab: 输出序列的字典
        :param inv_machine_vocab: 输出序列的逆字典
        :param human_vocab: 输入序列的字典

        :param use_pretrain: 使用训练好的模型
        :param model_path: 预训练模型的路径
        """
        self.Tx = Tx
        self.Ty = Ty

        self.n_a = n_a
        self.n_s = n_s

        self.machine_vocab = machine_vocab
        self.inv_machine_vocab = inv_machine_vocab
        self.human_vocab = human_vocab

        self.model_path = model_path

        # 输出序列的字典大小
        self.machine_vocab_size = len(machine_vocab)
        # 输入序列的字典大小
        self.human_vocab_size = len(human_vocab)

        # 对组成计算图的所有网络层进行声明和初始化
        self.__init_computation_graph()

        # 用于训练的计算图
        self.model_train = self.__encoder_decoder_model(Tx=self.Tx, Ty=self.Ty, human_vocab_size=self.human_vocab_size)

        if use_pretrain:  # 载入训练好的模型

            print('load pretrained model sucess ')

            self.model_train.load_weights(self.model_path)


    def __init_computation_graph(self):
        """
        对组成计算图的所有网络层进行声明和初始化
        :return:
        """

        # Model 也可以作为一个网络层
        self.model_one_step_attention = self.__one_step_attention_model(self.Tx, self.n_a, self.n_s)

        self.pre_activation_LSTM_cell = Bidirectional(LSTM(self.n_a, return_sequences=True, return_state=True),
                                                      name='encoder_lstm')

        self.concatenate_s = Concatenate(name='concatenate_s')
        self.concatenate_c = Concatenate(name='concatenate_c')

        self.concatenate_context = Concatenate()

        self.post_activation_LSTM_cell = LSTM(self.n_s, return_state=True, name='decoder_lstm')
        self.output_layer = Dense(len(self.machine_vocab), activation='softmax', name='decoder_output')

        self.lambda_argmax = Lambda(TensorUtils.argmax_tensor, arguments={'axis': -1}, name='argmax_tensor')
        self.lambda_one_hot = Lambda(TensorUtils.one_hot_tensor, arguments={'num_classes': len(self.machine_vocab)},
                                     name='one_hot_tensor')

        self.reshape = Reshape(target_shape=(1, len(self.machine_vocab)))

    def __one_step_attention_model(self, Tx, n_a, n_s):
        """
         attention 模块的实现
        把 几个网络层(keras layer) 包装为 model ,并通过重新定义 model 的输入的方式 来共享 layer 的权重
        :param Tx: 输入序列的长度
        :param n_a: encoder-lstm 中隐藏状态的维度
        :param n_s: decoder-lstm 中隐藏状态的维度
        :return: keras model
            model inputs: [a0, s_prev0]
            model outputs : context 向量, 作为 decoder-lstm 的输入 shape: (m, 1, 2*n_a)
        """

        repeator = RepeatVector(Tx)
        concatenator = Concatenate(axis=2)
        densor = Dense(1, activation="relu")
        activator = Softmax(axis=1)
        dotor = Dot(axes=1)

        a0 = Input(shape=(Tx, 2 * n_a), name='a')  # shape: (m, Tx, 2 * n_a)
        s_prev0 = Input(shape=(n_s,), name='s_prev')  # shape: (m, Tx, n_s)

        a = a0  # 否则报错 ： ValueError: Graph disconnected: cannot obtain value for tensor Tensor ....
        #             The following previous layers were accessed without issue: []

        s_prev = s_prev0

        s_prev = repeator(s_prev)  # shape: (m, Tx, n_s)

        concat = concatenator([a, s_prev])  # shape: (m, Tx, 2*n_a+n_s)

        e = densor(concat)  # shape: (m, Tx, 1)

        alphas = activator(e)  # shape:  (m, Tx, 1)

        context = dotor([alphas, a])  # shape: (m, 1, 2*n_a)

        model = Model(inputs=[a0, s_prev0], outputs=context)  # Model 也可以作为一个网络层

        return model

    def __encoder_decoder_model(self, Tx, Ty, human_vocab_size):
        """
        实现 encoder 和 decoder
        1. 解码时加入上一个时刻的输出单词,
           考虑场景: 若前一个时刻的词是 '-', 则当前词必须为数字
           在 decoder中, 经过 softmax 输出后 取最大的 那一个字符的 one-hot 向量 与 context 拼接后输入 decoder-lstm
        2. 修改 decoder-LSTM 的初始 隐藏状态的输入 ，由原来的 0 向量，改为 encoder-LSTM 最后一个时间步的隐状态（注意进行拼接）
        3. 把所有的 keras layer object 声明为类变量，以便 后面重构 decoder 可以使用训练好的网络结构
        :param Tx: 输入序列的长度
        :param Ty: 输出序列的长度
        :param human_vocab_size: 输入序列的字典大小
        :return: Keras model instance
        """

        X = Input(shape=(Tx, human_vocab_size))  # shape: (m,Tx,human_vocab_size)

        pred0 = Input(shape=(1, self.machine_vocab_size), name='pred0')  # shape: (m ,1, 11)

        pred = pred0

        # print('pred: after Input', pred)

        outputs = []

        # lstm-encoder
        a, forward_h, forward_c, backward_h, backward_c = self.pre_activation_LSTM_cell(
            inputs=X)  # shape of a : (m,Tx, 2*n_a)

        # 最后一个时间步的 隐藏状态向量
        s = self.concatenate_s(inputs=[forward_h, backward_h])  # shape  (m, 2*n_a=128)
        # 最后一个时间步的 细胞状态向量
        c = self.concatenate_c(inputs=[forward_c, backward_c])  # shape  (m, 2*n_a=128)

        # lstm-decoder
        for t in range(Ty):  # 遍历 Ty 个时间步

            context = self.model_one_step_attention(inputs=[a, s])  # shape of context :  (m, 1, 2*n_a=128)
            # print('context after one_step_attention: ', context)

            context = self.concatenate_context(inputs=[context, pred])  # shape of context: (m,1,128+11=139)

            # print('context after Concatenate:  ', context)

            s, _, c = self.post_activation_LSTM_cell(inputs=context, initial_state=[s, c])  # 输入 context 只有1个时间步

            out = self.output_layer(inputs=s)  # shape (m, machine_vocab)

            pred = self.lambda_argmax(inputs=out)  # shape (m, 1)
            pred = self.lambda_one_hot(inputs=pred)  # shape (m, machine_vocab_size)

            # print(pred)

            pred = self.reshape(inputs=pred)  # shape: (m ,1, machine_vocab_size)
            # print(pred)

            outputs.append(out)  # shape : (Ty, m, machine_vocab_size)

        model = Model(inputs=[X, pred0], outputs=outputs)
        # shape of outputs : ( Ty ,m ,machine_vocab)

        return model

    def fit(self, Xoh, Yoh, epoch_num=120, batch_size=2048):
        """
        训练模型
        :param Xoh: 输入序列 (one-hot化)
        :param Yoh: 输出序列 (one-hot化)
        :param epoch_num: 模型训练的 epoch 个数,  一般训练集所有的样本模型都见过一遍才算一个 epoch
        :param batch_size: 选择 min-Batch梯度下降时, 每一次输入模型的样本个数 (默认 = 2048)
        :return:
        """

        m = np.shape(Xoh)[0] # 训练样本总数

        model = self.__encoder_decoder_model(self.Tx, self.Ty, self.human_vocab_size)

        # 打印 模型(计算图) 的所有网络层
        # print(model.summary())

        # 画出计算图
        # plot_model(model, to_file='doc/model2.png')

        opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        outputs = list(Yoh.swapaxes(0, 1))  # Yoh.swapaxes(0,1) 第0维度 和 第1 维度交换，原来为(m,T_y,11) 变换后 为：(T_y,m,11)

        pred0 = np.zeros((m, 1, self.machine_vocab_size))

        history = model.fit([Xoh, pred0], outputs, epochs=epoch_num, batch_size=batch_size, validation_split=0.1)

        # 将训练好的模型保存到文件
        model.save(self.model_path)

        print('save model sucess ')

    def __inference_encoder_model(self, Tx, human_vocab_size):
        """
        推理过程中 解耦后的 encoder
        :param Tx: 输入序列的长度
        :param human_vocab_size: 输入序列的字典大小
        :return: model -- Keras model instance
        """

        X = Input(shape=(Tx, human_vocab_size))  # shape: (N,Tx,human_vocab_size)

        # 使用 已经训练好的 网络层
        a, forward_h, forward_c, backward_h, backward_c = self.pre_activation_LSTM_cell(
            inputs=X)  # shape of a : (N, Tx, 2*n_a)

        # encoder 最后一个时间步的 隐藏状态向量
        s = self.concatenate_s([forward_h, backward_h])  # shape of s:  (N, n_a+n_a) = (m,128)
        # encoder 最后一个时间步的 细胞状态向量
        c = self.concatenate_c([forward_c, backward_c])

        # context = self.model_one_step_attention(inputs=[a, s])  # shape of context :  (N, 1, 128)

        outputs = [a, s, c]

        model = Model(inputs=[X], outputs=outputs)  # 生成计算图(模型)

        return model

    def __inference_onestep_decoder_model(self, n_s, Tx):
        """
        推理过程中 解耦后的 一个时间步的 decoder
        :param n_s: decoder-lstm 中隐藏状态的维度
        :return: model -- Keras model instance
        """

        a = Input(shape=(Tx, n_s), name='a')  # shape (N, Tx, 2*n_a)

        # 输入的隐藏状态向量
        s0 = Input(shape=(n_s,), name='s')  # shape of s:  (m, n_s=64)
        # 输入的细胞状态向量
        c0 = Input(shape=(n_s,), name='c')  # shape of c:  (m, n_s=64)

        # decoder 中上一个时间步的输出
        pred = Input(shape=(1, self.machine_vocab_size), name='pred')  # shape of pred (m ,1, 11)

        s = s0  # unmutable object: a new tensor is generated
        c = c0

        # print('pred: after Input',pred)

        context = self.model_one_step_attention(inputs=[a, s])  # shape of context :  (m, 1, 2*n_a=128)
        # print('context after one_step_attention: ', context)

        context = self.concatenate_context(inputs=[context, pred])  # shape of context: (m,128+11=139)

        s, _, c = self.post_activation_LSTM_cell(inputs=context, initial_state=[s, c])
        # s 最后一个时间步的隐藏状态向量 shape (m, n_s)
        # c 最后一个时间步的细胞状态向量 shape (m, n_s)

        out = self.output_layer(inputs=s)  # shape (m, machine_vocab_size)

        outputs = [s, c, out]  # 输出 s c out 作为下一个时间步使用

        model = Model(inputs=[a, s0, c0, pred], outputs=outputs)  # 生成计算图(模型)

        return model

    def __inference_beamsearch(self, source_oh, Tx, Ty, n_s, human_vocab_size, machine_vocab_size, k=3):
        """
        实现 beamsearch (带窗口的贪心搜索)
         np.array -> tensor 很自然, 但是 tensor -> np.array 的方式： K.eval(tensor) 需要启动框架计算 计算图, 非常耗费时间；
        尽量多用 numpy 的函数，以减少 tensor 到 np.array 的转换的次数
        :param source_oh: 输入序列(one-hot化)
        :param Tx: 输入序列的长度
        :param Ty: 输出序列的长度
        :param n_s: decoder-lstm 中隐藏状态的维度
        :param human_vocab_size: 输入序列的字典大小
        :param machine_vocab_size: 输出序列的字典大小
        :param k: 窗口大小
        :return: decoder_result
        """

        pred0 = np.zeros((k, 1, self.machine_vocab_size))

        pred = pred0

        decoder_result = []

        # encoder 和 decoder 的解耦
        encoder = self.__inference_encoder_model(Tx, human_vocab_size)
        context, s, c = encoder.predict([source_oh])
        # source_oh shape：(m=3, Tx=30, human_vocab_size=37)
        # shape of context :  (m, 1, 2*n_a=128)
        # shape of s:  (m, 2*n_a)
        # shape of c:  (m, 2*n_a)

        # bearmserach decoder 实现

        for timestep in range(Ty):

            # print('timestep :', timestep)

            onestep_decoder = self.__inference_onestep_decoder_model(n_s, Tx)
            s, c, out = onestep_decoder.predict([context, s, c, pred])
            # out: softmax 层输出的为 11 个分类的概率 shape (m=3, machine_vocab_size=11) 输入的样本数量为3
            # s 最后一个时间步的隐藏状态向量 shape (m, n_s)
            # c 最后一个时间步的细胞状态向量 shape (m, n_s)

            # print('out: \n', out)  # shape:(3, 11)

            # 每次都对 3个相同的样本（k=3）进行 推理，但是每一个 样本对应的 pred 不同 ；
            # beamsearch 中，每一个时间步都会根据上一步的 onestep_decoder 输出结果中 选择最好的k个, 输入此时间步的 onestep_decoder

            if timestep == 0:

                out_top_K = ArrayUtils.partition_topk_array(out,
                                                            k)  # shape:(3,3) 每一行为 k 个标号, 表示从每个样本的11个分类中选出概率最大的 3个 类别
                # print('out_top_K: \n', out_top_K)

                top_K_indices = out_top_K

                r0 = top_K_indices[0]  # shape:(1,3) 因为3个输入样本是一样的，取其中一个即可

                r0 = np.reshape(r0, (k, 1))  # shape:(3,1)

                decoder_result = r0

                one_hot = ArrayUtils.one_hot_array(r0.reshape(-1), machine_vocab_size)  # shape:(3,11)
                # 把 r0.reshape(-1) shape:(3,) 最后一个维度 变为 one-hot 向量

                # print(one_hot)

                one_hot = np.reshape(one_hot, (1, one_hot.shape[0], one_hot.shape[1]))  # shape:(1,3,11)

                one_hot_permute = one_hot.transpose((1, 0, 2))  # shape: (3,1,11) ；
                # 交换 第0维 和 第1维，相当于3个不同的 pred 同时输入下一个时间步的 onestep_decoder
                pred = one_hot_permute

            else:

                out_top_K = ArrayUtils.whole_topk_array(out,
                                                        k)  # shape:(3, 2) 找出 out (shape (3,11)) 中33 个元素中的k个最大的元素的标号(位置)
                # out shape:(3, 11)

                r = out_top_K
                # print('r: \n', r)
                #   [[1 1]
                #    [0 1]
                #    [2 1]]  # topk 的元素在out中的标号为 [2,1], 代表 第2个输入的pred 所输出的11个分类中的第1个类别

                r_pre = decoder_result  # shape:(k,timestep) 上一步 解码的结果 即是 这一步的输入
                # [[2]
                #  [1]
                #  [3]]

                rt = np.zeros((k, timestep + 1), dtype=np.int32)  # 这一步 会在上一步 已有的解码序列的基础上 增加1个 解码位

                for i in range(k):
                    rt[i, :] = np.concatenate((r_pre[r[i][0]], [r[i][1]]), axis=0)
                    # i=2:
                    # r[2][0]=2 说明是第2个输入的pred，前一步的解码情况为： r_pre[r[2][0]]=[3] ，
                    # 再连接上这一步的解码位  r[2][1]=1 得到 解码序列：[3,1]
                    # 一共k 个解码序列 组成 rt

                decoder_result = rt

                # decoder_result:
                # [[1 1]
                #  [2 1]
                #  [3 1]]

                one_hot = ArrayUtils.one_hot_array(decoder_result[:, -1], machine_vocab_size)  # shape:(3, 11)
                # print(one_hot.shape)

                one_hot = np.reshape(one_hot, (1, one_hot.shape[0], one_hot.shape[1]))  # shape:(1,3, 11)
                one_hot_permute = one_hot.transpose((1, 0, 2))  # shape: (3,1,11)

                pred = one_hot_permute

            # print('decoder_result: \n', decoder_result)

        return decoder_result

    def inference(self, example):
        """
        使用训练好的模型进行推理
        :param example: 样本序列
        :return:
        """

        source = np.array(string_to_int(example, self.Tx, self.human_vocab))

        source_oh = ArrayUtils.one_hot_array(source, nb_classes=self.human_vocab_size)

        # beamsearch 的窗口大小
        k = 3

        source_oh = source_oh.reshape(1, source_oh.shape[0], source_oh.shape[1])
        # print(source_oh.shape)
        source_oh = np.repeat(source_oh, k, axis=0)
        # print(source_oh.shape) #(3, 30, 37) m=k=3 将一个样本复制为3个, 输入模型进行推理

        decoder_result = self.__inference_beamsearch(source_oh=source_oh, Tx=self.Tx, Ty=self.Ty, n_s=self.n_s,
                                                     human_vocab_size=self.human_vocab_size,
                                                     machine_vocab_size=self.machine_vocab_size, k=k)

        candidates = []

        for prediction in decoder_result:
            output = ''.join(int_to_string(prediction, self.inv_machine_vocab))

            # print("source:", example)
            # print("output:", output)

            candidates.append(output)

        return candidates

    @deprecated(version='1.0', reason="You should use another function")
    def evaluate_deprecated(self, source_list, reference_list):
        """
        使用 bleu 对翻译结果进行评价
        :param source_list: 待翻译的句子的列表
        :param reference_list: 对照语料, 人工翻译的句子列表
        :return: average_bleu_score : 所有 source 的最佳翻译结果的平均分数 ;
                 best_result_list : 所有 source 的最佳翻译结果
                 best_result = (max_bleu_score, source, reference, best_candidate)
        """

        bleu_score_list = np.zeros(len(source_list))
        best_result_list = []

        for i in range(len(source_list)):

            source = source_list[i] # "3rd of March 2002"
            reference = reference_list[i] # "2002-03-03"

            candidates = self.inference(source) #  ['2002-03-03', '0002-03-03', '1002-03-03']

            max_bleu_score = float('-inf') # 最佳分数
            best_candidate = None # 最好的翻译结果

            reference_arr = reference.split('-')
            # 对 reference 切分(分隔符为 '-' )为   ['2002','03','03']

            for candidate in candidates: # 遍历所有的 candidate, 找到 分数最高的

                candidate_arr = candidate.split('-')
                # 对 candidate 切分(分隔符为 '-' )为   ['2002','03','03']

                bleu_score = BleuScore.compute_bleu_corpus([[reference_arr]], [candidate_arr], N=2)[0]

                # bleu_score = corpus_bleu([[reference_arr]], [candidate_arr], weights=(0.5, 0.5, 0, 0))

                # 因为 reference_arr 的长度为3 , 很容易导致 3-gram-precision=0,
                # 根据 bleu 计算公式 log(0) -> -inf, 导致bleu=0;
                # weights=(0.5, 0.5, 0, 0) 表示 只考虑 1-gram-precision 和 2-gram-precision 的对数加权和

                if bleu_score > max_bleu_score:

                    max_bleu_score = bleu_score
                    best_candidate = candidate

            bleu_score_list[i] = max_bleu_score

            best_result = (max_bleu_score, source, reference, best_candidate)

            print('i:{}, best_result:{}'.format(i,best_result))

            best_result_list.append(best_result)

        average_bleu_score = np.average(bleu_score_list)

        return average_bleu_score, best_result_list

    def evaluate(self, source_list, reference_list):
        """
        使用 bleu 对翻译结果进行评价
        1.推理时采用 beamsearch (窗口大小 k = 3), 我们取 bleu 得分最高的作为此样本的预测序列
        2.词元(term)的粒度
        (1) '1990-09-23' 使用分隔符 '-' 切分为 3个 term ['1990','09','23'],
            计算 bleu 时设置 N_gram 的长度上限为 2(仅仅考虑 1-gram, 2-gram)
        (2) '1978-12-21' 切分为 10个 term ['1', '9', '7', '8', '-', '1', '2', '-', '2', '1']
        :param source_list: 待翻译的句子的列表
        :param reference_list: 对照语料, 人工翻译的句子列表
        :return: average_bleu_score : 所有 source 的最佳翻译结果的平均分数 ;
                 best_result_list : 所有 source 的最佳翻译结果
                 best_result = (max_bleu_score, source, reference, best_candidate)
        """

        bleu_score_list = np.zeros(len(source_list))
        best_result_list = []

        for i in range(len(source_list)):

            source = source_list[i] # "3rd of March 2002"
            reference = reference_list[i] # "2002-03-03"

            candidates = self.inference(source) #  ['2002-03-03', '0002-03-03', '1002-03-03']

            # reference_arr = reference.split('-')
            # 使用分隔符为 '-', 对 reference 切分为  ['2002','03','03']
            # candidate_arr_list = [candidate.split('-')for candidate in candidates]

            reference_arr = list(reference)
            candidate_arr_list = [list(candidate) for candidate in candidates]

            candidates_bleu_score = BleuScore.compute_bleu_corpus( [[reference_arr]]*len(candidates), candidate_arr_list, N=4)

            max_bleu_score = np.max(candidates_bleu_score)
            best_idx = np.argmax(candidates_bleu_score)
            bleu_score_list[i] = max_bleu_score
            best_candidate = candidates[best_idx]

            best_result = (max_bleu_score, source, reference, best_candidate)

            print('i:{}, best_result:{}'.format(i,best_result))

            best_result_list.append(best_result)

        average_bleu_score = np.average(bleu_score_list)

        return average_bleu_score, best_result_list


class MachineTranslationV2:
    """
    基于 LSTM + seq2seq + attention 的翻译模型

    对比 class BasicMachineTranslation 改进如下:

    1. 采用面向对象实现
    2. 更改了 attention 机制
       (1) 对于 decoder, 第一个时间步的输入的 隐藏状态向量(s) 和 细胞状态向量(c) 为 encoder 最后一个时间步的输出
       (2) 对于 decoder, 每一个时间步的输入除了 attention 模块对 encoder 所有时间步的输出的加权和 之外,
           还引入了 decoder 上一个时间步的输出

    3. 基于 beamsearch 的推理
       (1) decoder 采用一体化模型解码的方式, 即构建推理计算图, 一次推理拿到所有时间步的结果

    Author: xrh
    Date: 2019-12-16

    ref:
    1.https://github.com/enggen/Deep-Learning-Coursera
    2.论文 Neural machine translation by jointly learning to align and translate


    """

    def __init__(self, Tx, Ty, n_a, n_h, vocab_target, inv_vocab_target, vocab_source, k=3,
                 use_pretrain=False, model_path='models/lstm_seq2seq_attention.h5'):
        """
        模型初始化

        :param Tx: 编码器输入序列的长度
        :param Ty: 解码器输出序列的长度
        :param n_a: encoder-lstm 中隐藏状态的维度 n_a = 64
        :param n_h: decoder-lstm 中隐藏状态的维度 n_h = 128
        :param vocab_target: 输出序列的字典
        :param inv_vocab_target: 输出序列的逆字典
        :param vocab_source: 输入序列的字典
        :param k: 模型推理时 beamsearch 的窗口

        :param use_pretrain: 使用训练好的模型
        :param model_path: 预训练模型的路径

        """
        self.Tx = Tx
        self.Ty = Ty

        self.n_a = n_a
        self.n_h = n_h

        self.vocab_target = vocab_target
        self.inv_vocab_target = inv_vocab_target
        self.vocab_source = vocab_source

        self.k = k

        self.model_path = model_path

        # 输出序列的字典大小
        self.vocab_target_size = len(vocab_target)
        # 输入序列的字典大小
        self.vocab_source_size = len(vocab_source)

        # 对组成计算图的所有网络层进行声明和初始化
        self.__init_computation_graph()

        # 用于训练的计算图
        self.model_train = self.train_model()

        # 用于推理的计算图
        self.infer_model = self.inference_model()

        if use_pretrain:  # 载入训练好的模型

            print('load pretrained model sucess ')

            self.model_train.load_weights(self.model_path)


    def __init_computation_graph(self):
        """
        对组成计算图的所有网络层进行声明和初始化

        :return:
        """

        # Model 也可以作为一个网络层
        self.model_one_step_attention = self.__one_step_attention_model(self.Tx, self.n_a, self.n_h)

        self.pre_activation_LSTM_cell = Bidirectional(LSTM(self.n_a, return_sequences=True, return_state=True),
                                                      name='encoder_lstm')

        self.concatenate_h = Concatenate(name='concatenate_h')
        self.concatenate_c = Concatenate(name='concatenate_c')

        self.concatenate_context = Concatenate()

        self.post_activation_LSTM_cell = LSTM(self.n_h, return_state=True, name='decoder_lstm')
        self.output_layer = Dense(len(self.vocab_target), activation='softmax', name='decoder_output')

        self.lambda_argmax = Lambda(K.argmax, arguments={'axis': -1}, name='argmax_tensor')
        self.lambda_one_hot = Lambda(TensorUtils.one_hot_tensor, arguments={'num_classes': len(self.vocab_target)},
                                     name='one_hot_tensor')

        # self.reshape = Reshape(target_shape=(1, len(self.vocab_target)))

        self.lambda_expand_dims = Lambda(K.expand_dims, arguments={'axis': 1}, name='expand_dims')

        self.lambda_whole_top_k = Lambda(TensorUtils.whole_top_k_tensor, arguments={'k': self.k},
                                     name='whole_top_k_tensor')


    def __one_step_attention_model(self, Tx, n_a, n_h):
        """
         attention 模块的实现

        把 几个网络层(keras layer) 包装为 model ,并通过重新定义 model 的输入的方式 来共享 layer 的权重

        :param Tx: 输入序列的长度
        :param n_a: encoder-lstm 中隐藏状态的维度
        :param n_h: decoder-lstm 中隐藏状态的维度

        :return: keras model
            model inputs: [a0, s_prev0]
            model outputs : context 向量, 作为 decoder-lstm 的输入 shape: (N, 1, 2*n_a)

        """

        repeator = RepeatVector(Tx)
        concatenator = Concatenate(axis=2)
        densor = Dense(1, activation="relu")
        activator = Softmax(axis=1)
        dotor = Dot(axes=1)

        a0 = Input(shape=(Tx, 2 * n_a), name='a')  # shape: (N, Tx, 2 * n_a)
        s_prev0 = Input(shape=(n_h,), name='s_prev')  # shape: (N, Tx, n_h)

        a = a0  # 否则报错 ： ValueError: Graph disconnected: cannot obtain value for tensor Tensor ....
        #             The following previous layers were accessed without issue: []

        s_prev = s_prev0

        s_prev = repeator(s_prev)  # shape: (N, Tx, n_h)

        concat = concatenator([a, s_prev])  # shape: (N, Tx, 2*n_a+n_h)

        e = densor(concat)  # shape: (N, Tx, 1)

        alphas = activator(e)  # shape:  (N, Tx, 1)

        context = dotor([alphas, a])  # shape: (N, 1, 2*n_a)

        model = Model(inputs=[a0, s_prev0], outputs=context)  # Model 也可以作为一个网络层

        return model

    def train_model(self,):
        """
        将各个 网络层(layer) 拼接为训练计算图, 包括 encoder 和 decoder

        1. 解码时加入上一个时刻的输出单词,
           考虑场景: 若前一个时刻的词是 '-', 则当前词必须为数字
           在 decoder中, 经过 softmax 输出后 取最大的 那一个字符的 one-hot 向量 与 context 拼接后输入 decoder-lstm

        2. 修改 decoder-LSTM 的初始 隐藏状态的输入 ，由原来的 0 向量，改为 encoder-LSTM 最后一个时间步的隐状态（注意进行拼接）

        3. 把所有的 keras layer object 声明为类变量，以便 后面重构 decoder 可以使用训练好的网络结构


        :return: Keras model instance

        """

        X = Input(shape=(self.Tx, self.vocab_source_size))  # shape: (N,Tx,vocab_source_size)

        pred0 = Input(shape=(1, self.vocab_target_size), name='pred0')  # shape: (m ,1, 11)

        pred = pred0

        # print('pred: after Input', pred)

        outputs = []

        # lstm-encoder
        a, forward_h, forward_c, backward_h, backward_c = self.pre_activation_LSTM_cell(
            inputs=X)  # shape of a : (N,Tx, 2*n_a)

        # 最后一个时间步的 隐藏状态向量
        h = self.concatenate_h(inputs=[forward_h, backward_h])  # shape  (N, 2*n_a=128)
        # 最后一个时间步的 细胞状态向量
        c = self.concatenate_c(inputs=[forward_c, backward_c])  # shape  (N, 2*n_a=128)

        # lstm-decoder
        for t in range(self.Ty):  # 遍历 Ty 个时间步  #  TODO: 循环结构有可能导致 OON, 训练计算图应避免写此结构

            context = self.model_one_step_attention(inputs=[a, h])  # shape of context :  (N, 1, 2*n_a=128)
            # print('context after one_step_attention: ', context)

            context = self.concatenate_context(inputs=[context, pred])  # shape of context: (N,1,128+11=139)

            # print('context after Concatenate:  ', context)

            h, _, c = self.post_activation_LSTM_cell(inputs=context, initial_state=[h, c])  # 输入 context 只有1个时间步

            out = self.output_layer(inputs=h)  # shape (N, vocab_target)

            pred = self.lambda_argmax(inputs=out)  # shape (N, )
            pred = self.lambda_one_hot(inputs=pred)  # shape (N, vocab_target_size)

            # print(pred)

            pred = self.lambda_expand_dims(inputs=pred)  # shape: (N ,1, vocab_target_size)
            # print(pred)

            outputs.append(out)  # shape : (Ty, N, vocab_target_size)

        model = Model(inputs=[X, pred0], outputs=outputs)
        # shape of outputs : ( Ty ,m ,vocab_target)

        return model

    def fit(self, Xoh, Yoh, epoch_num=120, batch_size=2048):
        """
        训练模型

        :param Xoh: 输入序列 (one-hot化)
        :param Yoh: 输出序列 (one-hot化)
        :param epoch_num: 模型训练的 epoch 个数,  一般训练集所有的样本模型都见过一遍才算一个 epoch
        :param batch_size: 选择 min-Batch梯度下降时, 每一次输入模型的样本个数 (默认 = 2048)

        :return:
        """

        N = np.shape(Xoh)[0] # 训练样本总数

        # 打印 模型(计算图) 的所有网络层
        # print(self.model_train.summary())

        # 画出计算图
        # plot_model(self.model_train, to_file='docs/images/model_train_attention.png', show_layer_names=True, show_shapes=True)

        opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01/epoch_num)
        self.model_train.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        outputs = list(Yoh.swapaxes(0, 1))  # Yoh.swapaxes(0,1) 第0维度 和 第1 维度交换，原来为(N,T_y,11) 变换后 为：(T_y,N,11)

        pred0 = np.zeros((N, 1, self.vocab_target_size))

        history = self.model_train.fit([Xoh, pred0], outputs, epochs=epoch_num, batch_size=batch_size, validation_split=0.1)

        # 将训练好的模型保存到文件
        self.model_train.save(self.model_path)

        print('save model sucess ')

    def inference_model(self):
        """
        将各个 网络层(layer) 拼接为推理计算图
        并实现 beamsearch (带窗口的贪心搜索)

        :return:

        """

        X = Input(shape=(self.Tx, self.vocab_source_size))  # shape: (k, Tx, vocab_source_size)

        pred0 = Input(shape=(1, self.vocab_target_size), name='pred0')  # shape: (k ,1, vocab_target_size)
        pred = pred0

        # lstm-encoder
        a, forward_h, forward_c, backward_h, backward_c = self.pre_activation_LSTM_cell(
            inputs=X)  # shape of a : (k, Tx, 2*n_a)

        # encoder 最后一个时间步的 隐藏状态向量
        h = self.concatenate_h(inputs=[forward_h, backward_h])  # shape  (k, 2*n_a=128)
        # encoder 最后一个时间步的 细胞状态向量
        c = self.concatenate_c(inputs=[forward_c, backward_c])  # shape  (k, 2*n_a=128)

        outputs = []  # 记录每一个时间步解码器输出的窗口

        # lstm-decoder
        for t in range(self.Ty):  # 遍历 Ty 个时间步

            context = self.model_one_step_attention(inputs=[a, h])  # shape of context :  (k, 1, 2*n_a=128)
            # print('context after one_step_attention: ', context)

            context = self.concatenate_context(inputs=[context, pred])  # shape of context: (k, 1, 128+11=139)

            # print('context after Concatenate:  ', context)

            h, _, c = self.post_activation_LSTM_cell(inputs=context, initial_state=[h, c])  # 输入 context 只有1个时间步

            out = self.output_layer(inputs=h)  # shape (k, vocab_target)

            # for beamsearch
            top_k_index = self.lambda_whole_top_k(inputs=out)  # topk个元素 在矩阵中的标号, shape (k, 2)
            # TODO: 喂入数据后报错, 怀疑原因: 该层的输入为  shape (None, ...), 而输出为 shape (k, ...)

            # topk 的样本标号 shape (k, )
            # sample_id = tf.gather(top_k_index, indices=[0], axis=1)  # 通过索引取数组, 效果相当于 top_k_index[:, 0]

            # topk 的单词标号 shape (k, )
            # word_id = tf.gather(top_k_index, indices=[1], axis=1) #  效果相当于 top_k_index[:, 1]

            sample_id = Lambda(tf.gather, arguments={'indices': [0], 'axis': 1})(top_k_index)
            word_id = Lambda(tf.gather, arguments={'indices': [1], 'axis': 1})(top_k_index)

            sample_id = Lambda(tf.squeeze)(sample_id)
            word_id = Lambda(tf.squeeze)(word_id)

            # h = K.gather(h, sample_id)  # 通过索引取数组, 效果相当于  h = h[sample_id, :]
            # c = K.gather(c, sample_id)

            h = Lambda(K.gather, arguments={'indices': sample_id})(h)
            c = Lambda(K.gather, arguments={'indices': sample_id})(c)

            word_id_one_hot = self.lambda_one_hot(inputs=word_id)  # shape (k, vocab_target_size)

            pred = self.lambda_expand_dims(inputs=word_id_one_hot)  # shape: (k ,1, vocab_target_size)

            outputs.append(top_k_index)  # shape : (Ty, k, 2)

        model = Model(inputs=[X, pred0], outputs=outputs)
        # shape of outputs : ( Ty ,m ,vocab_target)

        return model

    def beam_search_seq_gen(self, top_k_index_list):
        """
        根据每一个时间步的 beamsearch 的预测结果, 生成解码序列

        :param top_k_index_list: topk 个元素在矩阵中的标号, shape (Ty, k, 2)
        :return:
        """

        # decoder_seq  shape (k, Ty)

        decoder_seq = []  # 解码序列

        for t in range(self.Ty):  # 遍历所有的时间步

            top_k_index = top_k_index_list[t] # shape (k, 2)
            sample_id = top_k_index[:, 0]  # topk 的样本标号 shape (k, )
            word_id = top_k_index[:, 1]  # topk 的单词标号 shape (k, )

            if t == 0:
                decoder_seq = np.expand_dims(word_id, axis=1)  # shape (k, 1)
                # [[20]
                #  [11]
                #  [32]]

            else:

                pre_decoder_seq = decoder_seq  # shape:(k, t) 上一步的解码序列
                # [[20]
                #  [11]
                #  [32]]

                decoder_seq = np.zeros((self.k, t + 1), dtype=np.int32)  # 这一步 会在上一步 已有的解码序列的基础上 增加1个 解码位

                # top_k_index:
                #   [[1 10]
                #    [0 12]
                #    [0 21]]

                # sample_id: [1, 0, 0]
                # word_id: [10, 12, 21]

                for i in range(self.k):  # 遍历所有的样本

                    # 前一步的解码情况作为前缀
                    prefix = pre_decoder_seq[sample_id[i]]  # sample_id[0]: 1 , prefix: [11]

                    # 这一步的解码位
                    c = word_id[i]  # c: 10

                    # 前缀
                    decoder_seq[i, :] = np.concatenate((prefix, [c]), axis=0)  # [11, 10]

            return decoder_seq

    def inference(self, Xoh):
        """
        使用训练好的模型进行推理

        :param Xoh: 输入序列 (one-hot化) shape: (N, Tx, vocab_source_size)

        :return:  candidate_group_list
        """

        # 打印 模型(计算图) 的所有网络层
        print(self.infer_model.summary())

        # 画出计算图
        plot_model(self.infer_model, to_file='../../../docs/images/infer_model_attention.png', show_layer_names=True, show_shapes=True)


        candidate_group_list = []

        # beamsearch 的窗口大小 k = 3
        pred0 = np.zeros((self.k, 1, self.vocab_target_size))

        for source_oh in Xoh:  # 遍历所有输入序列

            # source_oh shape (Tx, vocab_source_size)

            source_oh = np.expand_dims(source_oh, axis=0) #  shape (1, Tx, vocab_source_size)

            source_oh_batch = np.repeat(source_oh, self.k, axis=0)
            # print(source_oh.shape) #(3, 30, 37) k=3 将一个样本复制为3个, 输入模型进行推理

            top_k_index = self.infer_model.predict([source_oh_batch, pred0])

            decoder_result = self.beam_search_seq_gen(top_k_index)

            candidate_group = []

            for prediction in decoder_result:

                candidate = ''.join(int_to_string(prediction, self.inv_vocab_target))

                # print("output:", output)

                candidate_group.append(candidate)

            candidate_group_list.append(candidate_group)

        return candidate_group_list

    def evaluate(self, source_list, reference_list):
        """
        使用 bleu 对翻译结果进行评价

        1.推理时采用 beamsearch (窗口大小 k = 3), 我们取 bleu 得分最高的作为此样本的预测序列

        2.词元(term)的粒度
        (1) '1990-09-23' 使用分隔符 '-' 切分为 3个 term ['1990','09','23'],
            计算 bleu 时设置 N_gram 的长度上限为 2(仅仅考虑 1-graN, 2-gram)

        (2) '1978-12-21' 切分为 10个 term ['1', '9', '7', '8', '-', '1', '2', '-', '2', '1']

        :param source_list: 待翻译的句子的列表
        :param reference_list: 对照语料, 人工翻译的句子列表

        :return: average_bleu_score : 所有 source 的最佳翻译结果的平均分数 ;
                 best_result_list : 所有 source 的最佳翻译结果

                 best_result = (max_bleu_score, source, reference, best_candidate)

        """

        bleu_score_list = np.zeros(len(source_list))
        best_result_list = []

        for i in range(len(source_list)):

            source = source_list[i] # "3rd of March 2002"
            reference = reference_list[i] # "2002-03-03"

            candidates = self.inference(source) #  ['2002-03-03', '0002-03-03', '1002-03-03']

            # reference_arr = reference.split('-')
            # 使用分隔符为 '-', 对 reference 切分为  ['2002','03','03']
            # candidate_arr_list = [candidate.split('-')for candidate in candidates]

            reference_arr = list(reference)
            candidate_arr_list = [list(candidate) for candidate in candidates]

            candidates_bleu_score = BleuScore.compute_bleu_corpus( [[reference_arr]]*len(candidates), candidate_arr_list, N=4)

            max_bleu_score = np.max(candidates_bleu_score)
            best_idx = np.argmax(candidates_bleu_score)
            bleu_score_list[i] = max_bleu_score
            best_candidate = candidates[best_idx]

            best_result = (max_bleu_score, source, reference, best_candidate)

            print('i:{}, best_result:{}'.format(i,best_result))

            best_result_list.append(best_result)

        average_bleu_score = np.average(bleu_score_list)

        return average_bleu_score, best_result_list



class Test:

    def test_BasicMachineTranslation(self):
        # 配置 TensorFlow 在跑程序的时候不让程序占满 GPU 内存
        # physical_devices = tf.config.list_physical_devices('GPU')
        # tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

        m = 10000  # 数据集中的样本总数
        dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

        # 0.了解数据集
        print(dataset[:5])

        #  human date     machine date
        # 待翻译的日期     翻译后的日期
        # ('9 may 1998', '1998-05-09'),
        # ('10.11.19', '2019-11-10'),
        # ('9/10/70', '1970-09-10'),
        # ('saturday april 28 1990', '1990-04-28'),
        # ('thursday january 26 1995', '1995-01-26'),

        print(human_vocab)
        print(machine_vocab)

        Tx = 30
        Ty = 10
        X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
        # X 为待翻译的日期, Y 为翻译后的标准化的日期
        # Xoh 为 X 的 one-hot 化表示 , Yoh 为 Y 的 one-hot 化表示

        print("X.shape:", X.shape)
        print("Y.shape:", Y.shape)
        print("Xoh.shape:", Xoh.shape)
        print("Yoh.shape:", Yoh.shape)

        sol = BasicMachineTranslation()

        sol.model_implementation_naive(Tx=Tx, Ty=Ty, machine_vocab=machine_vocab, human_vocab=human_vocab, Xoh=Xoh,
                                       Yoh=Yoh, m=m)

        # sol.model_implementation_v2(Tx=Tx, Ty=Ty, machine_vocab=machine_vocab, inv_machine_vocab=inv_machine_vocab,
        #                             human_vocab=human_vocab, Xoh=Xoh, Yoh=Yoh, m=m)

    def test_MachineTranslationV1(self):

        m = 10000  # 数据集中的样本总数
        dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

        # 0.了解数据集
        print(dataset[:5])

        #  human date     machine date
        # 待翻译的日期     翻译后的日期
        # ('9 may 1998', '1998-05-09'),
        # ('10.11.19', '2019-11-10'),
        # ('9/10/70', '1970-09-10'),
        # ('saturday april 28 1990', '1990-04-28'),
        # ('thursday january 26 1995', '1995-01-26'),

        print(human_vocab)
        print(machine_vocab)

        # 划分训练集和测试集
        train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=1024)

        Tx = 30
        Ty = 10
        X, Y, Xoh, Yoh = preprocess_data(train_dataset, human_vocab, machine_vocab, Tx, Ty)
        # X 为待翻译的日期(字符以标号表示), Y 为翻译后的标准化的日期(字符以标号表示)
        # Xoh 为 X 的 one-hot 化表示 , Yoh 为 Y 的 one-hot 化表示

        print("X.shape:", X.shape)
        print("Y.shape:", Y.shape)
        print("Xoh.shape:", Xoh.shape)
        print("Yoh.shape:", Yoh.shape)

        # trainer = MachineTranslationV1(Tx=Tx, Ty=Ty, n_a=64, n_s=128, machine_vocab=machine_vocab,
        #                         inv_machine_vocab=inv_machine_vocab, human_vocab=human_vocab)

        # trainer.fit(Xoh=Xoh, Yoh=Yoh, epoch_num=200, batch_size=2048)

        infer = MachineTranslationV1(Tx=Tx, Ty=Ty, n_a=64, n_s=128, machine_vocab=machine_vocab,
                                inv_machine_vocab=inv_machine_vocab, human_vocab=human_vocab, use_pretrain=True)

        example = "december 21 1978"
        candidates = infer.inference(example)

        print("source:", example) # 待翻译的句子
        print("beam search candidates: \n", candidates) # 模型翻译的句子列表
        #  ['9987-12-21', '1988-12-21', '2987-12-21']

        test_dataset_arr = np.array(test_dataset[:200])  # 测试数据全部跑一遍太慢了

        X_test = test_dataset_arr[:, 0]  # 待翻译的日期(源序列)
        Y_test = test_dataset_arr[:, 1]  # 翻译后的日期(目标序列)

        average_bleu_score, best_result_list = infer.evaluate(X_test,Y_test)

        print('average_bleu_score:', average_bleu_score)


    def test_MachineTranslationV2(self):

        m = 10000  # 数据集中的样本总数
        dataset, vocab_source, vocab_target, inv_vocab_target = load_dataset(m)

        # 0.了解数据集
        print(dataset[:5])

        #  human date     machine date
        # 待翻译的日期     翻译后的日期
        # ('9 may 1998', '1998-05-09'),
        # ('10.11.19', '2019-11-10'),
        # ('9/10/70', '1970-09-10'),
        # ('saturday april 28 1990', '1990-04-28'),
        # ('thursday january 26 1995', '1995-01-26'),

        print(vocab_source)
        print(vocab_target)

        # 划分训练集和测试集
        train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=1024)

        Tx = 30
        Ty = 10
        X, Y, Xoh, Yoh = preprocess_data(train_dataset, vocab_source, vocab_target, Tx, Ty)
        # X 为待翻译的日期(字符以标号表示), Y 为翻译后的标准化的日期(字符以标号表示)
        # Xoh 为 X 的 one-hot 化表示 , Yoh 为 Y 的 one-hot 化表示

        print("X.shape:", X.shape)
        print("Y.shape:", Y.shape)
        print("Xoh.shape:", Xoh.shape)
        print("Yoh.shape:", Yoh.shape)

        # 1.模型训练
        n_a = 64
        n_h = 128

        # trainer = MachineTranslation(Tx=Tx, Ty=Ty, n_a=n_a, n_h=n_h, vocab_target=vocab_target,
        #                         inv_vocab_target=inv_vocab_target, vocab_source=vocab_source,
        #                         use_pretrain=False)

        # trainer.fit(Xoh=Xoh, Yoh=Yoh, epoch_num=50, batch_size=512)


        # 2.模型推理

        X_test, Y_test, Xoh_test, Yoh_test = preprocess_data(test_dataset, vocab_source, vocab_target, Tx, Ty)

        # tf.python.framework_ops.disable_eager_execution()

        infer = MachineTranslationV2(Tx=Tx, Ty=Ty, n_a=n_a, n_h=n_h, vocab_target=vocab_target,
                                inv_vocab_target=inv_vocab_target, vocab_source=vocab_source,
                                use_pretrain=True)

        candidate_group_list = infer.inference(Xoh_test[:10])

        print(candidate_group_list)

        # average_bleu_score, best_result_list = infer.evaluate(X_test,Y_test)
        #
        # print('average_bleu_score:', average_bleu_score)

if __name__ == '__main__':

    test = Test()

    # test.test_BasicMachineTranslation()

    test.test_MachineTranslationV1()

    # test.test_MachineTranslationV2()