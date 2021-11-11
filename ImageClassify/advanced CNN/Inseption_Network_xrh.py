#!/usr/bin/python
# -*- coding: UTF-8 -*-

#  适用于 tensorflow >= 2.0, keras 被直接集成到 tensorflow 的内部
#  ref: https://keras.io/about/

from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, \
    Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Concatenate
from tensorflow.keras.models import Model

from tensorflow.keras.initializers import glorot_uniform

from utils.dataset_xrh import *

from utils.utils_xrh import *

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Inception:
    """
    实现了 Inception 网络的其中一部分

    Author: xrh
    Date: 2019-10-16

    ref:
    1. 论文 Going Deeper with Convolutions

    """

    def __init__(self, input_shape=(64, 64, 3), class_num=6, model_path='models/Inception.h5'):
        """

        :param input_shape: 输入图片的大小 input_shape=(64, 64, 3)
        :param class_num: 分类的类别数
        :param model_path: 预训练模型的路径
        """
        self.input_shape = input_shape
        self.class_num = class_num
        self.model_path = model_path



    def conv2d_bn(self, x,
                  n_c,
                  n_h,
                  n_w,
                  padding='same',
                  strides=(1, 1),
                  name=None):
        """
        实现 卷积层 + BatchNormalization层

        :param x: 输入 tensor shape (m, n_H_prev, n_W_prev, n_C_prev)
        :param n_c: 输出通道个数
        :param n_h: 卷积核的高度
        :param n_w: 卷积核的宽度
        :param padding: padding 填充
        :param strides: 步长
        :param name: 当期层的名字
        :return:
        """
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(
            n_c, (n_h, n_w),
            strides=strides,
            padding=padding,
            use_bias=False,
            name=conv_name)(x)
        x = BatchNormalization(axis=3, scale=False, name=bn_name)(x)  # BN 作用在输出通道(特征图), 对每一个通道, 要训练一个单独的BN
        x = Activation('relu', name=name)(x)
        return x

    def inception_v1_model(self, input_shape=(64, 64, 3)):
        """
        inception_v1 网络截取了一部分

        :param input_shape:  输入图片的大小 input_shape=(64, 64, 3) 前面要增加样本个数(m)的维度, 实际 shape(m, 64, 64, 3)
        :return:
        """

        # Define the input as a tensor with shape input_shape
        X_input = Input(input_shape)  #

        #   conv->maxpool->conv->maxpool
        X = self.conv2d_bn(X_input, 64, 7, 7, strides=(2, 2))  # 29x29
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)  # 14x14
        X = self.conv2d_bn(X, 64, 1, 1)
        X = self.conv2d_bn(X, 192, 3, 3)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)  # 6x6

        # inception(3a)
        # mixed 0 :28 x 28 x 192  -> 28 x 28 x 256 (64+128+32+32=256)
        branch_0 = self.conv2d_bn(X, 64, 1, 1)  # 通道序列[64,96,128,16,32,32]其实和 论文 中table1 的顺序相同

        branch_1 = self.conv2d_bn(X, 96, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 128, 3, 3)

        branch_2 = self.conv2d_bn(X, 16, 1, 1)
        branch_2 = self.conv2d_bn(branch_2, 32, 5, 5)

        branch_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(X)
        branch_3 = self.conv2d_bn(branch_3, 32, 1, 1)

        channel_axis = 3  # 通道维度的标号

        X = Concatenate(
            axis=channel_axis,
            name='mixed0')([branch_0, branch_1, branch_2, branch_3])

        # inception(3b)
        # mixed 1 :28 x 28 x 256  -> 28 x 28 x 480  (128+192+96+64=480)
        branch_0 = self.conv2d_bn(X, 128, 1, 1)

        branch_1 = self.conv2d_bn(X, 128, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 192, 3, 3)

        branch_2 = self.conv2d_bn(X, 32, 1, 1)
        branch_2 = self.conv2d_bn(branch_2, 96, 5, 5)

        branch_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(X)
        branch_3 = self.conv2d_bn(branch_3, 64, 1, 1)

        X = Concatenate(
            axis=channel_axis,
            name='mixed1')([branch_0, branch_1, branch_2, branch_3])

        # AVGPOOL
        X = AveragePooling2D(pool_size=(2, 2), padding='valid')(X)

        # output layer
        X = Flatten()(X)
        X = Dense(self.class_num, activation='softmax', name='fc' + str(self.class_num), kernel_initializer=glorot_uniform(seed=0))(X)
        #   6 classes

        # Create model
        model = Model(inputs=X_input, outputs=X, name='inception_v1')

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

        model = self.inception_v1_model(self.input_shape)

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
        model = self.inception_v1_model(self.input_shape)

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
        model = self.inception_v1_model(self.input_shape)

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

        print("number of training examples = " + str(X_train.shape[0]))
        print("number of test examples = " + str(X_test.shape[0]))
        print("X_train shape: " + str(X_train.shape))
        print("Y_train shape: " + str(Y_train.shape))
        print("X_test shape: " + str(X_test.shape))
        print("Y_test shape: " + str(Y_test.shape))

        inception_net = Inception(input_shape=(64, 64, 3), class_num=6)

        # inception_net.fit(X_train=X_train, Y_train=Y_train)

        y_predict = inception_net.predict(X_test)
        print('test accuracy :', accuracy_score(y_predict, Y_test))

    def test_cafir_dataset(self):

        dataset_dir = 'datasets/cafir-10/cifar-10-batches-py'

        data = get_CIFAR10_data(dataset_dir, subtract_mean=False)  # subtract_mean 是否对样本特征进行normalize
        for k, v in data.items():
            print('%s: ' % k, v.shape)

        inception_net = Inception(input_shape=(32, 32, 3), class_num=10)

        # inception_net.fit(X_train=data['X_train'], Y_train=data['y_train'], epoch_num=30, batch_size=512)

        y_predict = inception_net.predict(data['X_test'])
        print('test accuracy :', accuracy_score(y_predict, data['y_test']))





if __name__ == '__main__':

    test = Test()

    # test.test_signs_dataset()

    test.test_cafir_dataset()