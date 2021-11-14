#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import pickle
import pandas as pd
import re
import string
from collections import *

import tensorflow as tf
from tensorflow.keras.models import Model


class DataPreprocess:
    """
    利用 tf.data 数据流水线的数据集预处理

    主流程见 do_main()

    Author: xrh
    Date: 2021-11-01

    """

    def __init__(self,
                 base_dir='../dataset/',
                 cache_data_folder='cache_data',
                 tokenizer_file='tokenizer.bin',
                 _null_str='<NULL>',
                 _start_str='<START>',
                 _end_str='<END>',
                 _unk_str='<UNK>',
                 tensor_int_type=tf.int32,
                 ):
        """
        :param base_dir:  数据集的路径
        :param cache_data_folder: 预处理结果文件夹
        :param tokenizer_dir: 词典路径
        :param  _null_str: 空字符
        :param  _start_str: 句子的开始字符
        :param  _end_str: 句子的结束字符
        :param  _unk_str: 未登录字符
        :param  tensor_int_type: 操作系统不同, windows 选择 tf.int32 , linux 选择 tf.int64

        """

        self.cache_data_dir = os.path.join(base_dir, cache_data_folder)
        self.tokenizer_path = os.path.join(self.cache_data_dir, tokenizer_file)

        self.tensor_int_type = tensor_int_type

        self.caption_file_dir = os.path.join(base_dir, 'Flicker8k/Flickr8k.token.txt')
        self.image_folder_dir = os.path.join(base_dir, 'Flicker8k/Flicker8k_Dataset/')

        self.train_image_file_dir = os.path.join(base_dir, 'Flicker8k/Flickr_8k.trainImages.txt')
        self.valid_image_file_dir = os.path.join(base_dir, 'Flicker8k/Flickr_8k.devImages.txt')
        self.test_image_file_dir = os.path.join(base_dir, 'Flicker8k/Flickr_8k.testImages.txt')

        self.dataset_dir = os.path.join(base_dir, 'cache_data/train_dataset.json')
        self.image_caption_dict_dir = os.path.join(base_dir, 'cache_data/image_caption_dict.bin')

        self._null_str = _null_str
        self._start_str = _start_str
        self._end_str = _end_str
        self._unk_str = _unk_str

        # 删除不在各个语言字典中的字符
        self.remove_unk_re = re.compile(r'[^\u4e00-\u9fa5\u0030-\u0039\u0021-\u007e\u00C0-\u00FF]')

        """
        [^**]	表示不匹配此字符集中的任何一个字符
        \u4e00-\u9fa5	汉字的unicode范围
        \u0030-\u0039	数字的unicode范围
        \u0041-\u005a	大写字母unicode范围
        \u0061-\u007a	小写字母unicode范围
        \u0021-\u007e   英文字母unicode范围(含数字与符号)
        \u00C0-\u00FF   德文/法文的unicode范围
        \uAC00-\uD7AF	韩文的unicode范围
        \u3040-\u31FF	日文的unicode范围

        """

        # 需要删除的标点符号, 包含中文符号
        remove_chars = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·"""

        remove_chars = remove_chars.replace("'", "")  # 英语中有 ', 不删除语料库中的 '
        remove_chars = remove_chars.replace("-", "")  #

        self.remove_chars_re = re.compile('[%s]' % re.escape(remove_chars))

        # 删除 数字 字符
        self.remove_digits_re = re.compile(r'\s[0-9]+\s')

    def load_image_name(self, image_file_path):
        """
        导入文件中的图片的名字

        :param image_file_path: 文件路径
        :return:
        """
        with open(image_file_path) as file:

            line_list = file.readlines()

            image_name_list = []

            for line in line_list:

                name = line.rstrip("\n")

                image_name_list.append(os.path.join(self.image_folder_dir, name))

        return image_name_list

    def load_captions_data(self, clean_punctuation=True, clean_digits=True, lowercase=True):
        """
        读取 图片描述文本, 并将它们和对应的图片进行映射

        1.图片描述文本 可以选择是否清除其中的标点符号

        :param clean_punctuation:  是否清除文本中的标点符号
        :param clean_digits: 是否清除文本中的数字
        :param clean_digits: 是否全部小写
        :return:

        caption_mapping: 字典, key 为图片的路径, value 为图片描述的文本列表
        text_data: 所有图片描述的文本

        """

        with open(self.caption_file_dir) as caption_file:

            caption_data = caption_file.readlines()
            caption_mapping = {}
            text_data = []

            for line in caption_data:

                line = line.rstrip("\n")

                # Image name and captions are separated using a tab
                img_name, caption = line.split("\t")

                # Each image is repeated five times for the five different captions. Each
                # image name has a prefix `#(caption_number)`
                img_name = img_name.split("#")[0]

                img_name = os.path.join(self.image_folder_dir, img_name.strip())

                if img_name.endswith("jpg"):

                    # 清除句子前后的空格
                    caption = caption.strip()

                    # 删除非法的字符
                    caption = self.remove_unk_re.sub(' ', caption)

                    if clean_punctuation:
                        # 清除句子中的标点符号
                        caption = self.remove_chars_re.sub(' ', caption)

                    if clean_digits:
                        # 清除句子中的数字
                        caption = self.remove_digits_re.sub(' ', caption)

                    if lowercase:
                        # 全部转成小写
                        caption = caption.lower()

                    # We will add a start and an end token to each caption
                    caption = self._start_str + " " + caption + " " + self._end_str
                    text_data.append(caption)

                    if img_name in caption_mapping:
                        caption_mapping[img_name].append(caption)
                    else:
                        caption_mapping[img_name] = [caption]

            return caption_mapping, text_data


    def __calc_max_length(self, tensor):
        """
        Find the maximum length of any caption in the dataset
        :param tensor:
        :return:
        """

        return max(len(t) for t in tensor)

    def tokenize_corpus(self, text_data, n_vocab):
        """
        对语料库中的句子进行标记化, 同时生成词典

        :param text_data: 句子的列表
        :param n_vocab: 词表大小

        :return:
        """
        top_k = n_vocab - 1  # 考虑 <NULL> 也是一个单词

        # top_k: 保留出现次数最多的 topk 个单词
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, oov_token=self._unk_str, filters='',
                                                          lower=False)

        tokenizer.fit_on_texts(text_data)

        # print(tokenizer.index_word[0])

        tokenizer.word_index[self._null_str] = 0
        tokenizer.index_word[0] = self._null_str

        print('word_index length: {}'.format(len(tokenizer.word_index)))

        # Create the tokenized vectors
        caption_vector = tokenizer.texts_to_sequences(text_data)

        # Pad each vector to the max_length of the captions
        # If you do not provide a max_length value, pad_sequences calculates it automatically
        caption_vector_pad = tf.keras.preprocessing.sequence.pad_sequences(caption_vector, padding='post')

        # Calculates the max_length, which is used to store the attention weights
        max_length = self.__calc_max_length(caption_vector)
        print('max_caption_length:{}'.format(max_length))

        print('tokenizer:')

        print('<NULL> id: ', tokenizer.word_index[self._null_str])
        print('<UNK> id: ', tokenizer.word_index[self._unk_str])
        print('<START> id: ', tokenizer.word_index[self._start_str])
        print('<END> id: ', tokenizer.word_index[self._end_str])

        print('id:0 word:', tokenizer.index_word[0])
        print('id:1 word:', tokenizer.index_word[1])
        print('id:2 word:', tokenizer.index_word[2])
        print('id:3 word:', tokenizer.index_word[3])
        print('id:4 word:', tokenizer.index_word[4])

        print('cache the tokenizer in {}'.format(self.tokenizer_path))

        save_dict = {}
        save_dict['tokenizer'] = tokenizer
        with open(self.tokenizer_path, 'wb') as f:
            pickle.dump(save_dict, f)

        return caption_vector_pad, max_length, tokenizer


    def image_embedding_InceptionV3(self, image_path_list, batch_num=8):
        """
        使用预训练的 CNN 对图片进行映射

        :param image_path_list:
        :return:
        """

        def load_image(image_path):
            img = tf.io.read_file(image_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, (299, 299))

            img = tf.keras.applications.inception_v3.preprocess_input(img)

            return img, image_path

        model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet')

        new_input = model.input
        hidden_layer = model.layers[-3].output

        print('the embedding layer output tensor: ', hidden_layer)

        model_emb_pict = Model(new_input, hidden_layer)

        image_dataset = tf.data.Dataset.from_tensor_slices(image_path_list)

        image_dataset = image_dataset.map(
            load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_num)

        for img, path in tqdm(image_dataset):

            batch_features = model_emb_pict(img)

            batch_features = tf.reshape(batch_features,
                                        (batch_features.shape[0], -1, batch_features.shape[
                                            3]))  # 把向量拍平,  shape (None, 8, 8, 2048) -> shape (None, 64, 2048)

            for bf, p in zip(batch_features, path):
                path_of_feature = p.numpy().decode("utf-8")  # 以文件的形式缓存在图片所在的目录下, 每一张图片对应一个同名的特征向量文件
                np.save(path_of_feature, bf.numpy())

    def image_embedding_VGG16(self, image_path_list, batch_num=8):
        """
        使用预训练的 CNN 对图片进行映射

        :param image_path_list:
        :param batch_num: 取决于 GPU的显存大小

        :return:
        """

        def load_image(image_path):
            img = tf.io.read_file(image_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, (224, 224))

            img = tf.keras.applications.vgg16.preprocess_input(img)

            return img, image_path

        model = tf.keras.applications.vgg16.VGG16(weights='imagenet')

        new_input = model.input
        hidden_layer = model.layers[-6].output

        print('the embedding layer output tensor: ', hidden_layer)

        model_emb_pict = Model(new_input, hidden_layer)

        image_dataset = tf.data.Dataset.from_tensor_slices(image_path_list)

        image_dataset = image_dataset.map(
            load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_num)

        for img, path in tqdm(image_dataset):

            batch_features = model_emb_pict(img)

            batch_features = tf.reshape(batch_features,
                                        (batch_features.shape[0], -1, batch_features.shape[
                                            3]))  # 把向量拍平,  shape (None, 16, 16, 512) -> shape (None, 196, 512)

            for bf, p in zip(batch_features, path):
                path_of_feature = p.numpy().decode("utf-8")  # 以文件的形式缓存在图片所在的目录下, 每一张图片对应一个同名的特征向量文件
                np.save(path_of_feature, bf.numpy())

    def image_embedding_VGG19(self, image_path_list, batch_num=8):
        """
        使用预训练的 CNN 对图片进行映射

        :param image_path_list:
        :param batch_num: 取决于 GPU的显存大小

        :return:
        """

        def load_image(image_path):
            img = tf.io.read_file(image_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, (224, 224))

            img = tf.keras.applications.vgg19.preprocess_input(img)

            return img, image_path

        model = tf.keras.applications.vgg19.VGG19(weights='imagenet')

        new_input = model.input
        hidden_layer = model.layers[-6].output

        print('the embedding layer output tensor: ', hidden_layer)

        model_emb_pict = Model(new_input, hidden_layer)

        image_dataset = tf.data.Dataset.from_tensor_slices(image_path_list)

        image_dataset = image_dataset.map(
            load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_num)

        for img, path in tqdm(image_dataset):

            batch_features = model_emb_pict(img)

            batch_features = tf.reshape(batch_features,
                                        (batch_features.shape[0], -1, batch_features.shape[
                                            3]))  # 把向量拍平,  shape (None, 16, 16, 512) -> shape (None, 196, 512)

            for bf, p in zip(batch_features, path):
                path_of_feature = p.numpy().decode("utf-8")  # 以文件的形式缓存在图片所在的目录下, 每一张图片对应一个同名的特征向量文件
                np.save(path_of_feature, bf.numpy())


    def build_caption_vector_dict(self, image_path_list, caption_dict, caption_vector_pad):
        """
        制作 图片路径 到 标记化后的句子 的字典

        :param image_path_list:
        :param caption_dict:
        :param caption_vector_pad:
        :return:
        """

        image_path_expand_list = []  # 膨胀后的图片路径的列表

        # caption_list = []

        for image_path in image_path_list:
            caption_num = len(caption_dict[image_path])

            image_path_expand_list.extend([image_path] * caption_num)  # 一张图片会对应多个描述句子

            # caption_list.extend(caption_dict[image_path])

        caption_vector_dict = defaultdict(list)  # 图片路径 到 标记化后的句子 的字典

        for img, cap in zip(image_path_expand_list, caption_vector_pad):
            caption_vector_dict[img].append(cap.tolist())  # cap 为 array, 转换为 list , 方便后面的统一处理

        return caption_vector_dict

    def train_val_split(self, caption_dict, train_size=0.8, shuffle=True):
        """
        将数据集划分为 训练数据集 和 验证数据集(测试数据)

        :param caption_dict: 字典, key 为图片的名字, value 为图片描述的文本列表
        :param train_size: 训练数据的比例
        :param shuffle: 是否混洗
        :return:
            train_caption_dict : 字典, key 为图片的路径, value 为描述图片的文本列表
            validation_caption_dict : 字典, key 为图片的路径 value 为描述图片的文本列表
        """

        # 1. 图片的路径列表
        all_images = list(caption_dict.keys())

        # 2.进行混洗
        if shuffle:
            np.random.shuffle(all_images)

        # 3. 训练集的大小
        train_size = int(len(caption_dict) * train_size)

        train_caption_dict = {
            img_name: caption_dict[img_name] for img_name in all_images[:train_size]
        }
        valid_caption_dict = {
            img_name: caption_dict[img_name] for img_name in all_images[train_size:]
        }

        return train_caption_dict, valid_caption_dict

    def train_val_test_split(self, caption_dict, train_images, valid_images, test_images):

        train_caption_dict = {
            img_name: caption_dict[img_name] for img_name in train_images
        }

        valid_caption_dict = {
            img_name: caption_dict[img_name] for img_name in valid_images
        }

        test_caption_dict = {
            img_name: caption_dict[img_name] for img_name in test_images
        }

        return train_caption_dict, valid_caption_dict, test_caption_dict

    def __map_func(sefl, image_path, caption_in):
        """
        Load the numpy files

        :param image_path:
        :param caption_in:
        :return:
        """
        image_tensor = np.load(image_path.decode('utf-8') + '.npy')

        return image_tensor, caption_in

    def tf_data_pipline(self, caption_vector_dict, do_persist=False, dataset_file=None):
        """
        利用 tf.data 数据流水线建立数据集,
        将图片向量化后和对应的句子集合合并为一个元组
        
        :param caption_vector_dict:
        :return:
        """

        image_path_expand_list = []  # 膨胀后的图片路径的列表

        caption_vector_list = []

        for image_path in caption_vector_dict.keys():

            caption_num = len(caption_vector_dict[image_path])

            image_path_expand_list.extend([image_path] * caption_num)  # 一张图片会对应多个描述句子

            caption_vector_list.extend(caption_vector_dict[image_path])

        #  膨胀后的图片路径的列表的长度与图片描述的列表的长度相同
        assert len(image_path_expand_list) == len(caption_vector_list)

        print('dataset tuple num: ', len(caption_vector_list))


        caption_arr = np.array(caption_vector_list)

        # 输入 和 输出的 caption 要错开一位
        caption_out = caption_arr[:, 1:]  # shape(N,max_caption_length-1)
        caption_in = caption_arr[:, :-1]  # shape(N,max_caption_length-1)

        # 特征和标签分开, 之后再合并

        image_features_dataset = tf.data.Dataset.from_tensor_slices(
            (image_path_expand_list, caption_in))

        labels_dataset = tf.data.Dataset.from_tensor_slices(caption_out)


        # Use map to load the numpy files in parallel
        image_features_dataset_map = image_features_dataset.map(lambda item1, item2: tf.numpy_function(
            self.__map_func, [item1, item2], [tf.float32, self.tensor_int_type]),
                                                                            num_parallel_calls=tf.data.AUTOTUNE)

        # 特征 和 标签 合并
        dataset = tf.data.Dataset.zip((image_features_dataset_map, labels_dataset))

        if do_persist:
            dataset_path = os.path.join(self.cache_data_dir, dataset_file)
            tf.data.experimental.save(dataset, dataset_path)

        return dataset

    def build_image_caption_dict(self, caption_dict,  do_persist=False, image_caption_dict_file=None):
        """

        1.一张图片对应多段描述, 因此需要组合 图片路径, 图片向量 和 图片的描述, 返回组合后的字典

        :param caption_dict: 字典, key 为图片的路径, value 为图片描述的文本列表
        :param do_persist: 将结果持久化到磁盘
        :param image_caption_dict_file:

        :return:
            image_caption_dict
            = {
                '.../.../XXX.jpg' : {
                                'feature': 编码后的图片向量
                                'caption': 图片描述的文本列表
                              }
              }
        """

        image_caption_dict = {}

        for k, v_list in caption_dict.items():
            image_path = k
            image_feature = np.load(image_path + '.npy')
            image_caption_dict[image_path] = {'feature': image_feature, 'caption': v_list}

        if do_persist:
            save_dict = {}
            save_dict['image_caption_dict'] = image_caption_dict

            image_caption_dict_path = os.path.join(self.cache_data_dir, image_caption_dict_file)

            with open(image_caption_dict_path, 'wb') as f:
                pickle.dump(save_dict, f)

        return image_caption_dict

    def do_mian_split_random(self, cnn_batch_size, n_vocab):
        """
        数据集预处理的主流程

        将数据集随机划分为 训练集和验证集

        :param cnn_batch_size:
        :param n_vocab:
        :return:
        """

        caption_dict, text_data = self.load_captions_data(lowercase=True, clean_digits=True)

        image_path_list = list(caption_dict.keys())

        # self.image_embedding_VGG16(image_path_list, batch_num=cnn_batch_size)

        caption_vector_pad, max_length, tokenizer = self.tokenize_corpus(text_data, n_vocab=n_vocab)

        caption_vector_dict = self.build_caption_vector_dict(image_path_list, caption_dict, caption_vector_pad)

        seed = 1

        np.random.seed(seed)  # 设置随机数种子
        # 图片路径到标记化后的句子的字典
        train_caption_vector_dict, valid_caption_vector_dict = self.train_val_split(caption_vector_dict)

        np.random.seed(seed)  # 设置随机数种子, 保证出现相同的混洗结果
        train_caption_dict, valid_caption_dict = self.train_val_split(caption_dict)

        train_dataset = self.tf_data_pipline(train_caption_vector_dict, do_persist=True, dataset_file='train_image_tensor_caption.bin')
        valid_dataset = self.tf_data_pipline(valid_caption_vector_dict, do_persist=True, dataset_file='valid_image_tensor_caption.bin')

        test_image_caption_dict = self.build_image_caption_dict(valid_caption_dict, do_persist=True, image_caption_dict_file='test_image_caption_dict.bin')

    def do_mian_split_default(self, cnn_batch_size, n_vocab):
        """
        数据集预处理的主流程

        按照标准的划分方法 将数据集划分为 训练集, 验证集, 和测试集

        :param cnn_batch_size:
        :param n_vocab:
        :return:
        """

        caption_dict, text_data = self.load_captions_data(lowercase=True, clean_digits=True)

        image_path_list = list(caption_dict.keys())

        # self.image_embedding_VGG19(image_path_list, batch_num=cnn_batch_size)
        self.image_embedding_InceptionV3(image_path_list, batch_num=cnn_batch_size)

        caption_vector_pad, max_length, tokenizer = self.tokenize_corpus(text_data, n_vocab=n_vocab)

        caption_vector_dict = self.build_caption_vector_dict(image_path_list, caption_dict, caption_vector_pad)

        train_image_list = self.load_image_name(self.train_image_file_dir)
        valid_image_list = self.load_image_name(self.valid_image_file_dir)
        test_image_list = self.load_image_name(self.test_image_file_dir)

        # 图片路径到标记化后的句子的字典
        train_caption_vector_dict, valid_caption_vector_dict, test_caption_vector_dict = self.train_val_test_split(caption_vector_dict, train_image_list, valid_image_list, test_image_list)

        train_caption_dict, valid_caption_dict, test_caption_dict = self.train_val_test_split(caption_dict,  train_image_list, valid_image_list, test_image_list)

        train_dataset = self.tf_data_pipline(train_caption_vector_dict, do_persist=True, dataset_file='train_image_tensor_caption.bin')
        valid_dataset = self.tf_data_pipline(valid_caption_vector_dict, do_persist=True, dataset_file='valid_image_tensor_caption.bin')

        valid_image_caption_dict = self.build_image_caption_dict(valid_caption_dict, do_persist=True, image_caption_dict_file='valid_image_caption_dict.bin')
        test_image_caption_dict = self.build_image_caption_dict(test_caption_dict, do_persist=True, image_caption_dict_file='test_image_caption_dict.bin')


class Vocab:

    def __init__(self, tokenizer_path, _unk_str='<UNK>'):

        self._unk_str = _unk_str

        with open(tokenizer_path, 'rb') as f:
            save_dict = pickle.load(f)

        self.tokenizer = save_dict['tokenizer']

    def map_id_to_word(self, id):
        """
        输入单词标号, 返回单词

        1.若单词标号未在 逆词典中, 返回 '<UNK>'

        :param id:
        :return:
        """
        if id not in self.tokenizer.index_word:
            return self._unk_str
        else:
            return self.tokenizer.index_word[id]

    def map_word_to_id(self, word):
        """
        输入单词, 返回单词标号

        考虑未登录词:
        1.若输入的单词不在词典中, 返回 '<UNK>' 的标号

        :param word: 单词
        :return:
        """

        if word not in self.tokenizer.word_index:
            return self.tokenizer.word_index[self._unk_str]
        else:
            return self.tokenizer.word_index[word]


class FlickerDataset:
    """
    包装了 Flicker 数据集,  我们通过此类来访问该数据集

    1.使用之前先对数据集进行预处理(class DataPreprocess),
    预处理后的数据集在 dataset/cache_data 目录下

    Author: xrh
    Date: 2021-11-01

    """
    def __init__(self,
                 base_dir='../dataset/',
                 cache_data_folder='cache_data',
                 tokenizer_file='tokenizer.bin',
                 train_dataset_file='train_image_tensor_caption.bin',
                 valid_dataset_file='valid_image_tensor_caption.bin',
                 valid_image_caption_dict_file='valid_image_caption_dict.bin',
                 test_image_caption_dict_file='test_image_caption_dict.bin',
                 mode='train'):
        """

        :param base_dir: 数据集的根路径
        :param cache_data_folder: 预处理结果文件夹
        :param tokenizer_dir: 词典路径
        :param mode: 当前处在的模式, 可以选择
                    'train' - 训练模式
                    'infer' - 推理模式

        """
        self.cache_data_dir = os.path.join(base_dir, cache_data_folder)

        self.tokenizer_path = os.path.join(self.cache_data_dir, tokenizer_file)

        self.vocab = Vocab(self.tokenizer_path)

        if mode == 'train':

            train_dataset_path = os.path.join(self.cache_data_dir, train_dataset_file)
            valid_dataset_path = os.path.join(self.cache_data_dir, valid_dataset_file)

            # 训练数据
            self.train_dataset = tf.data.experimental.load(train_dataset_path)

            # 测试数据
            self.valid_dataset = tf.data.experimental.load(valid_dataset_path)

            valid_image_caption_dict_path = os.path.join(self.cache_data_dir, valid_image_caption_dict_file)

            with open(valid_image_caption_dict_path, 'rb') as f:
                save_dict = pickle.load(f)

            self.valid_image_caption_dict = save_dict['image_caption_dict']


        elif mode == 'infer':

            test_image_caption_dict_path = os.path.join(self.cache_data_dir, test_image_caption_dict_file)

            with open(test_image_caption_dict_path, 'rb') as f:
                save_dict = pickle.load(f)

            self.test_image_caption_dict = save_dict['image_caption_dict']


class Test:

    def test_DataPreprocess(self):

        process_obj = DataPreprocess()

        # process_obj.do_mian_split_random(cnn_batch_size=32, n_vocab=8868)

        process_obj.do_mian_split_default(cnn_batch_size=32, n_vocab=8868)



    def test_FlickerDataset(self):

        dataset_obj = FlickerDataset(mode='train')

        # 查看 1 个批次的数据
        for ele in tqdm(dataset_obj.train_dataset.take(1)):
            #     print(ele)

            print(len(ele))

            image_tensor = ele[0][0]
            caption_in = ele[0][1]
            caption_out = ele[1]

            print('image_tensor shape: ', np.shape(image_tensor))

            print(caption_in)
            print(caption_out)

        for ele in tqdm(dataset_obj.valid_dataset.take(1)):
            #     print(ele)

            print(len(ele))

            image_tensor = ele[0][0]
            caption_in = ele[0][1]
            caption_out = ele[1]

            print('image_tensor shape: ', np.shape(image_tensor))

            print(caption_in)
            print(caption_out)

        dataset_obj = FlickerDataset(mode='infer')

        print(list(dataset_obj.test_image_caption_dict.items())[0])

        print('word a index: ', dataset_obj.vocab.map_word_to_id('a'))

        print('word <UNK> index: ', dataset_obj.vocab.map_word_to_id('<UNK>'))

        print('word -= index: ', dataset_obj.vocab.map_word_to_id('-='))


if __name__ == '__main__':
    test = Test()

    #TODO：运行之前 把 jupyter notebook 停掉, 否则会出现争抢 GPU 导致报错

    test.test_DataPreprocess()

    # test.test_FlickerDataset()
