#!/usr/bin/python
# -*- coding: UTF-8 -*-

#  适用于 tensorflow >= 2.0, keras 被直接集成到 tensorflow 的内部
#  ref: https://keras.io/about/


from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, \
    Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model

from tensorflow.keras.initializers import glorot_uniform

from utils.dataset_xrh import *

from utils.utils_xrh import *

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class ResNet50:
    """
    残差网络

    Author: xrh
    Date: 2019-10-16

    ref:
    1. 论文 Deep Residual Learning for Image Recognition
    2. https://github.com/enggen/Deep-Learning-Coursera

    """

    def __init__(self, input_shape, class_num, model_path='models/ResNet50.h5'):
        """

        :param input_shape: 输入图片的大小 input_shape=(64, 64, 3)
        :param class_num: 分类的类别数
        :param model_path: 预训练模型的路径
        """
        self.input_shape = input_shape
        self.class_num = class_num
        self.model_path = model_path

    def identity_block(self, X, f, filter_num_list, stage, block):
        """
        实现 ResNet 的 identity block

        :param X: 输入 tensor , shape (m, n_H_prev, n_W_prev, n_C_prev)
        :param f: 中间的卷积的卷积核的大小
        :param filter_num_list: 各个卷积核的通道数目
        :param stage: 当前的层 id
        :param block: 层中的块 id
        :return: out - 输出 tensor, shape (m, n_H, n_W, n_C)

        """

        # 当前层的名字
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        F1, F2, F3 = filter_num_list

        X_shortcut = X  # 跳跃(短路主路径)

        # 主路径的第1个组件
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # 主路径的第2个组件
        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # 主路径的第3个组件
        X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

        # 短路后的从路径的结果 与 主路径的结果相加
        X = Add()([X, X_shortcut])
        out = Activation('relu')(X)

        return out

    def convolutional_block(self, X, f, filter_num_list, stage, block, s=2):
        """
        实现 ResNet 的 identity block

        :param X: 输入 tensor , shape (m, n_H_prev, n_W_prev, n_C_prev)
        :param f: 中间的卷积的卷积核的大小
        :param filter_num_list: 各个卷积核的通道数目
        :param stage: 当前的层 id
        :param block: 层中的块 id
        :param s: 卷积核移动的步长 (stride)
        :return: out - 输出 tensor, shape (m, n_H, n_W, n_C)

        """

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        F1, F2, F3 = filter_num_list

        X_shortcut = X  # 跳跃(短路主路径)

        # 主路径的第1个组件
        X = Conv2D(F1, (1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # 主路径的第2个组件
        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # 主路径的第3个组件
        X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

        #  从路径的第1个组件
        X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + 'l',
                            kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=bn_name_base + 'l')(X_shortcut)

        # 短路后的从路径的结果 与 主路径的结果相加
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def resnet50_model(self, input_shape=(64, 64, 3), class_num=6):
        """
        实现 ResNet50, 网络结构如下:

        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        :param input_shape: 输入图片的大小 input_shape=(64, 64, 3) 前面要增加样本个数(m)的维度, 实际 shape(m, 64, 64, 3)
        :param class_num: 分类的类别数

        :return: model

        """
        X_input = Input(input_shape)

        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)

        # Stage 1
        X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = self.convolutional_block(X, f=3, filter_num_list=[64, 64, 256], stage=2, block='a', s=1)
        X = self.identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        X = self.identity_block(X, 3, [64, 64, 256], stage=2, block='c')

        # Stage 3 (≈4 lines)
        X = self.convolutional_block(X, f=3, filter_num_list=[128, 128, 512], stage=3, block='a', s=2)
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='b')
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='c')
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='d')

        # Stage 4 (≈6 lines)
        X = self.convolutional_block(X, f=3, filter_num_list=[256, 256, 1024], stage=4, block='a', s=2)
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

        # Stage 5 (≈3 lines)
        X = self.convolutional_block(X, f=3, filter_num_list=[512, 512, 2048], stage=5, block='a', s=2)
        X = self.identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
        X = self.identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

        # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
        X = AveragePooling2D(pool_size=(2, 2), padding='valid')(X)

        # output layer
        X = Flatten()(X)
        X = Dense(class_num, activation='softmax', name='fc' + str(class_num), kernel_initializer=glorot_uniform(seed=0))(X)
        #   6 classes

        model = Model(inputs=X_input, outputs=X, name='ResNet50')

        return model

    def fit(self, X_train, Y_train, epoch_num=20, batch_size=32):
        """
        训练模型

        :param X_train: 输入图片
        :param Y_train: 输出标签
        :param epoch_num: 模型训练的 epoch 个数,  一般训练集所有的样本模型都见过一遍才算一个 epoch
        :param batch_size: 选择 min-Batch梯度下降时, 每一次输入模型的样本个数 (默认 = 32)

        :return:
        """

        m = np.shape(X_train)[0]  # 训练样本总数

        input_shape = np.shape(X_train)[1:]

        # one-hot 化
        Y_train_oh = ArrayUtils.one_hot_array(Y_train, self.class_num)

        assert self.class_num == np.shape(Y_train_oh)[-1]  # 设定的类别需要与样本标签类别一致

        assert self.input_shape == input_shape  # 设定的输入维度应与训练数据相同

        model = self.resnet50_model(self.input_shape, self.class_num)

        # 打印 模型(计算图) 的所有网络层
        # print(model.summary())

        # 画出计算图
        # plot_model(model, to_file='models/resnet_model.png')

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(X_train, Y_train_oh, epochs=epoch_num, batch_size=batch_size)

        # save the model
        model.save(self.model_path)

        print('save model dir:{} complete'.format(self.model_path))

    def evaluate(self, X_test, Y_test):
        """
        模型评价

        :param X_test:
        :param Y_test:
        :return:
        """
        model = self.resnet50_model(self.input_shape, self.class_num)

        # 载入训练好的模型
        model.load_weights(self.model_path)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # one-hot 化
        Y_test_oh = ArrayUtils.one_hot_array(Y_test, self.class_num)

        result = model.evaluate(X_test, Y_test_oh)

        accuracy = result[1]

        return accuracy

    def predict(self, X_test):
        """
        模型预测

        :param X_test:

        :return: labels -预测的标签 shape: (N,)
        """
        model = self.resnet50_model(self.input_shape, self.class_num)

        # 载入训练好的模型
        model.load_weights(self.model_path)

        # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        prob = model.predict(X_test)  # shape(120,6)

        labels = np.argmax(prob, axis=1)  # axis=1 干掉第1个维度, shape: (N,)

        return labels

class Test:

    def test_signs_dataset(self):

        signs_dataset_dir = 'datasets/signs'

        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset(signs_dataset_dir)

        # 图片特征的标准化
        X_train = X_train_orig / 255.
        X_test = X_test_orig / 255.

        Y_train = Y_train_orig
        Y_test =Y_test_orig

        # one-hot 化
        # Y_train_oh = ArrayUtils.one_hot_array(Y_train_orig, 6)
        # Y_test_oh = ArrayUtils.one_hot_array(Y_test_orig, 6)

        print("number of training examples = " + str(X_train.shape[0]))
        print("number of test examples = " + str(X_test.shape[0]))
        print("X_train shape: " + str(X_train.shape))
        print("Y_train shape: " + str(Y_train.shape))
        print("X_test shape: " + str(X_test.shape))
        print("Y_test shape: " + str(Y_test.shape))

        res_net = ResNet50(input_shape=(64, 64, 3), class_num=6)

        # res_net.fit(X_train=X_train, Y_train=Y_train)

        acurracy = res_net.evaluate(X_test, Y_test)
        print("Test Accuracy = {}".format(acurracy))

        y_predict = res_net.predict(X_test)
        print('test accuracy :', accuracy_score(y_predict, Y_test))



if __name__ == '__main__':

    test = Test()

    test.test_signs_dataset()
