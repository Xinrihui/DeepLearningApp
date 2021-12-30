#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf

from tensorflow.keras.layers import StringLookup

from lib.tf_data_prepare_xrh import *


class Vocab:
    """
    使用 python 的 dict 的 语料库词典

    """

    def __init__(self, vocab_list_path, vocab_list=None, _unk_str='[UNK]'):
        """
        词典的初始化

        :param vocab_path: 词典持久化的路径
        :param vocab_list:  (建立新的词典时必填)

        """
        self.vocab_list_path = vocab_list_path
        self._unk_str = _unk_str

        if vocab_list is not None:  # 建立新的词典

            self.vocab_list = vocab_list

            self.n_vocab = len(self.vocab_list)  # 字典的长度

            with open(self.vocab_list_path, 'w', encoding='utf-8') as f:

                for token in vocab_list:
                    print(token, file=f)

        else:  # 读取已有的词典

            with open(self.vocab_list_path, 'r', encoding='utf-8') as f:

                lines = f.readlines()

            self.vocab_list = [word.strip() for word in lines]

            self.n_vocab = len(self.vocab_list)  # 字典的长度

        self.word_to_id = {}
        for i, word in enumerate(self.vocab_list):
            self.word_to_id[word] = i

        self.id_to_word = {}
        for i, word in enumerate(self.vocab_list):
            self.id_to_word[i] = word

    def map_id_to_word(self, id):
        """
        输入单词标号, 返回单词

        1.若单词标号未在 逆词典中, 返回 '<UNK>'

        :param id:
        :return:
        """
        if id not in self.id_to_word:
            return self._unk_str
        else:
            return self.id_to_word[id]

    def map_word_to_id(self, word):
        """
        输入单词, 返回单词标号

        考虑未登录词:
        1.若输入的单词不在词典中, 返回 '<UNK>' 的标号

        :param word: 单词
        :return:
        """

        if word not in self.word_to_id:
            return self.word_to_id[self._unk_str]
        else:
            return self.word_to_id[word]


class VocabTf:
    """
    使用 tf 的 StringLookup 构建语料库词典

    """

    def __init__(self, vocab_list_path, vocab_list=None):
        """
        词典的初始化

        :param vocab_path: 词典持久化的路径
        :param vocab_list:  (建立新的词典时必填)

        """
        self.vocab_list_path = vocab_list_path

        if vocab_list is not None:  # 建立新的词典

            self.vocab_list = vocab_list

            self.n_vocab = len(self.vocab_list)  # 字典的长度

            with open(self.vocab_list_path, 'w', encoding='utf-8') as f:

                for token in vocab_list:
                    print(token, file=f)

        else:  # 读取已有的词典

            with open(self.vocab_list_path, 'r', encoding='utf-8') as f:

                lines = f.readlines()

            self.vocab_list = [word.strip() for word in lines]

            self.n_vocab = len(self.vocab_list)  # 字典的长度

    def map_id_to_word(self, ids):
        """
        输入单词标号列表, 返回单词列表

        1.若单词标号未在 逆词典中, 返回 '<UNK>'

        :param ids:
        :return:
        """

        id_to_word = StringLookup(
            vocabulary=self.vocab_list,
            mask_token='',
            invert=True)

        return id_to_word(ids)

    def map_word_to_id(self, words):
        """
        输入单词列表, 返回单词标号列表

        考虑未登录词:
        1.若输入的单词不在词典中, 返回 '<UNK>' 的标号

        :param word: 单词
        :return:
        """

        word_to_id = StringLookup(
            vocabulary=self.vocab_list,
            mask_token='',
        )

        return word_to_id(words)


class Test:

    def test_VocabTf(self, config_path='../config/transformer_seq2seq.ini', tag='DEFAULT'):

        config = configparser.ConfigParser()
        config.read(config_path, 'utf-8')
        current_config = config[tag]

        dataset_train_obj = WMT14_Eng_Ge_Dataset(cache_data_folder=current_config['cache_data_folder'], use_tf_vocab=True, mode='train')

        if current_config['return_mode'] == 'mid':
            # 查看 1 个批次的数据
            for source, target in tqdm(dataset_train_obj.train_dataset.take(1)):

                source_list = source[:10]
                print('source:')
                print(source_list)

                source_vector = dataset_train_obj.tokenizer_source.tokenize(source).to_tensor()
                print('source_vector:')
                print(source_vector)

                decode_result = dataset_train_obj.vocab_source.map_id_to_word(source_vector)

                decode_result = tf.strings.reduce_join(decode_result, axis=1,
                                                       separator=' ')

                decode_result = [sentence.numpy().decode('utf-8') for sentence in decode_result]

                print('decode_result:')
                print(decode_result)
