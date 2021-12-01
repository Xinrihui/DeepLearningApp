#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


class TensorUtils(object):
    """
    使用 tensorflow 后端构造自定义函数, 与 网络层(keras layer) Lambda 结合使用后
    可以生成更多无状态的网络层 (无需进行反向传播)

    eg.

    lambda_argmax=Lambda(tensors_tookits.argmax_tensor, arguments={'axis': -1 },name='argmax_tensor')
    lambda_one_hot=Lambda(tensors_tookits.one_hot_tensor, arguments={'num_classes': len(machine_vocab) },name='one_hot_tensor')

    pred=lambda_argmax(out)
    pred=lambda_one_hot(pred)

    Author: xrh
    Date: 2019-12-16

    """

    @staticmethod
    def argmax_tensor(x, axis):
        """
        求张量 x, 某一个维度的最大值

        ref: https://keras-zh.readthedocs.io/backend/

        :param x:
        :param axis: 目标维度
        :return:
        """
        return K.argmax(x, axis)

    @staticmethod
    def argmin_tensor(x, axis):
        """
        求张量 x, 某一个维度的最小值

        ref: https://keras-zh.readthedocs.io/backend/

        :param x:
        :param axis:目标维度
        :return:
        """
        return K.argmin(x, axis)

    @staticmethod
    def one_hot_tensor(x, num_classes):
        """
        对张量 x 进行 onehot 化

        x shape (m,1) -> (m, num_classes)

        ref:  https://fdalvi.github.io/blog/2018-04-07-keras-sequential-onehot/

        :param x:
        :param num_classes: 类别的数目
        :return:
        """

        return K.one_hot(K.cast(x, 'uint8'), num_classes)

    @staticmethod
    def top_k_tensor(input, k):
        """
        返回张量 input, 第 0个维度的 topk 个元素的标号

        :param input:
        :param k:
        :return:
        """

        return tf.nn.top_k(input,k).indices

    @staticmethod
    def whole_top_k_tensor(input, k):
        """
        返回整个张量的 topk 个元素的标号 (index)

        :param input:
        :param k:
        :return:
        """

        flatten = K.flatten(input)
        global_top_k = tf.nn.top_k(flatten, k)
        #         print('global topk values:',K.eval(global_top_k.values))
        #         print('glaobal topk indices:',K.eval(global_top_k.indices))
        indices = global_top_k.indices

        indices_row = K.cast(tf.floor(indices / input.shape[-1]), dtype=tf.int32)
        #     K.eval(indices_row)

        indices_col = indices % input.shape[-1]  # dtype='int32'
        # indices_col=tf.mod(indices,a.shape[-1]) #  tensorflow 的数学运算 https://blog.csdn.net/zywvvd/article/details/78593618

        #     K.eval(indices_col)

        indices = K.concatenate(
            [K.reshape(indices_row, (1, indices_row.shape[0])), K.reshape(indices_col, (1, indices_col.shape[0]))],
            axis=0)
        indices = K.transpose(indices)

        return indices


class ArrayUtils(object):

    @staticmethod
    def topk_array(x, k, axis=1):
        """
        使用 numpy 复现 tf.nn.topk

        perform topK based on np.argsort

        ref: https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array

        :param x: to be sorted
        :param K: select and sort the top K items
        :param axis: dimension to be sorted.
        :return:
        """

        full_sort = np.argsort(x, axis=axis)

        return full_sort[:, -k:]

    @staticmethod
    def partition_topk_array(x, K, axis=1):
        """
        perform topK based on np.argpartition

        :param x: to be sorted
        :param K: select and sort the top K items
        :param axis: 0 or 1. dimension to be sorted.
        :return:
        """
        a_part = np.argpartition(x, -K, axis=axis)
        if axis == 0:
            row_index = np.arange(x.shape[1 - axis])
            a_sec_argsort_K = np.argsort(x[a_part[-K:, :], row_index], axis=axis)
            return a_part[-K:, :][a_sec_argsort_K, row_index]
        else:
            column_index = np.arange(x.shape[1 - axis])[:, None]
            #         print('column_index ',column_index)
            #         print('matrix[column_index, a_part[:, -K:]] ',matrix[column_index, a_part[:, -K:]]) #选取矩阵中的一组元素
            a_sec_argsort_K = np.argsort(x[column_index, a_part[:, -K:]], axis=axis)
            #         print('a_sec_argsort_K ',a_sec_argsort_K)
            return a_part[:, -K:][column_index, a_sec_argsort_K]  # 乾坤大挪移，变换矩阵中的元素位置

    @staticmethod
    def whole_topk_array(x, k):
        """
        输出整个多维数组 x 的 k 个最大的元素的下标，但是这 k个元素并不会按照大小排序

        :param x:
        :param k:
        :return:
        """

        flatten = x.flatten()
        #     print(flatten)

        global_top_k = np.argpartition(flatten, -k)[-k:]

        indices = global_top_k

        indices_row = np.floor(indices / x.shape[-1])

        indices_col = indices % x.shape[-1]  # dtype='int32'

        indices = np.concatenate(
            [np.reshape(indices_row, (1, indices_row.shape[0])), np.reshape(indices_col, (1, indices_col.shape[0]))],
            axis=0)
        indices = np.transpose(indices)

        return indices.astype(np.int32)  # numpy 数据类型转换 ；查看数据类型： arr.dtype

    @staticmethod
    def one_hot_array(x, nb_classes):
        """
        对数组 x 进行 onehot 化

        :param x:
        :param nb_classes:
        :return:
        """

        res = np.eye(nb_classes)[np.array(x).reshape(-1)]
        #     print(res.shape)
        return res.reshape(list(x.shape) + [nb_classes])


class Test:

    def test_TensorUtils(self):

        a = np.array([1,2,0])
        #  K.eval() 执行计算图
        # print('one_hot_tensor: \n', K.eval(TensorUtils.one_hot_tensor(a, num_classes=3)))


        a = np.array(
            [[1, 2, 3, 4, 5],
             [1, 2, 2, 2, 2],
             [1, 3, 3, 3, 6]]
        )
        k = 3

        print('top_k_tensor: \n', K.eval(TensorUtils.top_k_tensor(a,k)))

        # print('whole_top_k_tensor: \n', K.eval(TensorUtils.whole_top_k_tensor(a,k)))

        # tensorflow >= 2.0 为动态图, 直接返回结果
        top_k_index = TensorUtils.whole_top_k_tensor(a, k)

        print('whole_top_k_tensor: \n', top_k_index)

        word_id = top_k_index[:, 1]  # shape (k, )
        print(word_id)
        word_id_one_hot = TensorUtils.one_hot_tensor(word_id, num_classes=5)
        print(word_id_one_hot)

        h = tf.constant(
           [[1,2],
            [3,4],
            [5,6]]
        )

        sample_id = top_k_index[:, 0]  # topk 的样本标号 shape (k, )
        print(sample_id)

        h = K.gather(h, sample_id)  # 通过索引取数组, 效果相当于  h = h[sample_id, :]

        print(h)

    def test_ArrayUtils(self):


        ##--part1--:
        k = 3
        # arr = np.array([[1, 98, 2, 99, 100]])

        a = np.array(
            [[1, 2, 3, 4, 5],
             [1, 2, 8, 2, 2],
             [9, 3, 3, 3, 6]]
        )

        print(ArrayUtils.partition_topk_array(a, k, axis=1)) # 最大的 三个元素 (已排序)

        ##--part1-- end --##

        ##--part2--: 输出整个矩阵的 topk

        print(ArrayUtils.whole_topk_array(a,k))

        ##--part2-- end --##

        ##--part3--: 使用 numpy 复现  tf.one_hot()


        b = ArrayUtils.one_hot_array(a,11)

        print(b)
        print(np.shape(b))

        ##--part3-- end --##


if __name__ == '__main__':

    test = Test()

    test.test_TensorUtils()

    # test.test_ArrayUtils()