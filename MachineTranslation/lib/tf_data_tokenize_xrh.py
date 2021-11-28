#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
from tqdm import tqdm
import pickle
import re
from collections import *
import string
import configparser

import tensorflow as tf

from tensorflow.keras.layers import TextVectorization, StringLookup

# from tensorflow.keras.layers.experimental import preprocessing

import tensorflow_text as tf_text


class DataPreprocess:
    """
    利用 tf.data 数据流水线 + TextVectorization 的数据集预处理

    主流程见 do_main()

    Author: xrh
    Date: 2021-11-15

    ref:
    https://keras.io/examples/nlp/neural_machine_translation_with_transformer/
    https://tensorflow.google.cn/text/tutorials/nmt_with_attention

    https://www.tensorflow.org/datasets/overview
    https://www.tensorflow.org/text
    https://tensorflow.google.cn/api_docs/python/tf/keras/layers/TextVectorization


    """

    def __init__(self,
                 config_path, tag,
                 base_dir='../dataset/WMT-14-English-Germa',
                 cache_data_folder='cache_data',

                 tensor_int_type=tf.int32,
                 ):
        """
        :param base_dir:  数据集的路径
        :param cache_data_folder: 预处理结果文件夹

        :param  tensor_int_type: 操作系统不同, windows 选择 tf.int32 , linux 选择 tf.int64

        """

        self.cache_data_dir = os.path.join(base_dir, cache_data_folder)
        self.tensor_int_type = tensor_int_type

        config = configparser.ConfigParser()
        config.read(config_path, 'utf-8')

        current_config = config[tag]
        print('current config tag:{}'.format(tag))

        self.train_source_corpus_dir = os.path.join(base_dir, current_config['train_source_corpus'])
        self.train_target_corpus_dir = os.path.join(base_dir, current_config['train_target_corpus'])

        self.valid_source_corpus_dir = os.path.join(base_dir,  current_config['valid_source_corpus'])
        self.valid_target_corpus_dir = os.path.join(base_dir,  current_config['valid_target_corpus'])

        self.test_source_corpus_dir = os.path.join(base_dir,  current_config['test_source_corpus'])
        self.test_target_corpus_dir = os.path.join(base_dir,  current_config['test_target_corpus'])


        self._null_str = current_config['_null_str']
        self._start_str = current_config['_start_str']
        self._end_str = current_config['_end_str']
        self._unk_str = current_config['_unk_str']


        # 删除在各个语系外的字符
        self.remove_unk = r'[^\p{Latin}|[:print:]]'
        # \p{Latin} 匹配拉丁语系
        # [[:print:]] 匹配打印字符 (≡ [A-Za-z0-9!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~] )
        # ref
        # [1] https://segmentfault.com/a/1190000021141670
        # [2] https://github.com/google/re2/wiki/Syntax#perl

        # 需要删除的标点符号
        punctuation = string.punctuation
        punctuation = punctuation.replace("'", "")  # 英语中有', 不删除 '
        punctuation = punctuation.replace("-", "")  #

        self.remove_punc = r'[%s]' % re.escape(punctuation)

        # 删除独立的数字字符
        # eg. '11 999 avc 10 abc-10 10-abc abc10 50 813 aa 4x4 20'
        #      ->
        #     ' avc abc-10 10-abc abc10 aa 4x4  '
        self.remove_digits = r'^(\d+ )+|( \d+)+ |(\d+)$'

        # 删除特定的单词
        self.remove_words = r'(##AT##-##AT##|&apos|&quot)'

    def load_corpus_data(self, corpus_file_dir):
        """
        读取 图片描述文本, 并将它们和对应的图片进行映射

        :param corpus_file_dir: 语料文本

        :return:

        text_data: 所有句子的列表

        """

        with open(corpus_file_dir, encoding='utf-8') as corpus_file:

            lines = corpus_file.readlines()
            text_data = []

            for line in lines:

                sentence = line.strip()

                # 句子的左右端点统一补充 1个空格
                # sentence = " " + sentence + " "

                text_data.append(sentence)

            return text_data


    def __tf_lower_and_split_punct(self, text):
        """
        对每一个句子的预处理

        :param text:
        :return:
        """

        # 删除在各个语系外的字符
        text = tf.strings.regex_replace(text, self.remove_unk, ' ')

        # 清除指定单词
        text = tf.strings.regex_replace(text, self.remove_words, '')

        # NKFC unicode 标准化 +  大小写折叠
        text = tf_text.case_fold_utf8(text)

        # 清除句子中的标点符号
        text = tf.strings.regex_replace(text, self.remove_punc, ' ')  # 空1格

        # 清除句子中的独立数字
        text = tf.strings.regex_replace(text, self.remove_digits, '')

        # 清除左右端点的空格
        text = tf.strings.strip(text)

        # 左右端点添加 开始 和 结束的控制字符
        text = tf.strings.join([self._start_str, text, self._end_str], separator=' ')

        return text

    def preprocess_corpus(self, text_data, batch_size=64):
        """
        对语料库中的句子进行预处理, 包括删除标点符号

        :param text_data:
        :param buffer_size:
        :param batch_size:

        :return:
        """

        text_dataset = tf.data.Dataset.from_tensor_slices(text_data).batch(batch_size)  # 分块后可以加速计算, 后面每次读取都是一块

        text_preprocess_dataset = text_dataset.map(lambda batch_text:
            self.__tf_lower_and_split_punct(batch_text), num_parallel_calls=tf.data.AUTOTUNE)

        return text_preprocess_dataset


    def tokenize_corpus(self, text_data, n_vocab, tokenizer_file, do_persist=True, batch_size=64):
        """
        对语料库中的句子进行标记化, 同时生成词典

        :param text_data: 句子的列表
        :param n_vocab: 词表大小
        :param tokenizer_file: 标记模型的文件
        :param do_persist: 是否持久化标记模型
        :param batch_size:

        :return:
        """

        text_preprocess_dataset = self.preprocess_corpus(text_data, batch_size=batch_size)

        tokenizer = TextVectorization(
                    standardize=None,
                    # output_sequence_length=max_seq_length,
                    max_tokens=n_vocab)
        # (1) max_tokens: 保留出现次数最多的 top-k 个单词
        # (2) 若 output_sequence_length=None, 自动将本 batch 中最长的句子的长度作为 padding 的长度
        #  在 padding 时 每一个 batch 中的序列的长度可能不同, 在同个 batch 中自然是统一的

        tokenizer.adapt(text_preprocess_dataset)

        vocab_list = tokenizer.get_vocabulary()

        print('vocab_list length: {}'.format(len(vocab_list)))

        print('vocab_list: ', vocab_list[:20])

        # 标记化所有的句子, 包括 padding
        # text_vector_dataset = text_preprocess_dataset.map(lambda batch_text: tokenizer(batch_text))
        # print('tokenize text complete ')

        if do_persist:
            # 持久化 tokenizer
            tokenizer_path = os.path.join(self.cache_data_dir, tokenizer_file)
            model = tf.keras.Sequential(tokenizer)
            model.save(tokenizer_path)

        return text_preprocess_dataset, tokenizer, vocab_list

    def tf_data_pipline(self, source_dataset, target_dataset, mode='mid', tokenizer_source=None, tokenizer_target=None, do_persist=False, dataset_file=None):
        """
        利用 tf.data 数据流水线 建立数据集,

        :param source_dataset:
        :param target_target:
        :param tokenizer_source:
        :param tokenizer_target:

        :param do_persist:
        :param dataset_file:
        :return:
        """

        assert len(source_dataset) == len(target_dataset)

        print('dataset batch num: ', len(source_dataset))

        dataset = None

        if mode == 'final':  # 返回最终的数据集, 模型可以直接加载进入训练

            # 将文本标记化, 每一个 batch 中的 序列的长度相同
            source_vector = source_dataset.map(lambda batch_text: tokenizer_source(batch_text))
            target_vector = target_dataset.map(lambda batch_text: tokenizer_target(batch_text))

            print('tokenize text complete ')

            # 输入 和 输出的 target 要错开一位
            target_out = target_vector.map(lambda batch_text: batch_text[:, 1:], num_parallel_calls=tf.data.AUTOTUNE)
            target_in = target_vector.map(lambda batch_text: batch_text[:, :-1], num_parallel_calls=tf.data.AUTOTUNE)

            features = tf.data.Dataset.zip((source_vector, target_in))
            labels = target_out

            dataset = tf.data.Dataset.zip((features, labels))

        elif mode == 'mid':  # 返回中间态的数据集, 模型需要做进一步预处理

            dataset = tf.data.Dataset.zip((source_dataset, target_dataset))

        if do_persist:
            dataset_path = os.path.join(self.cache_data_dir, dataset_file)
            tf.data.experimental.save(dataset, dataset_path)

        return dataset


    def build_source_target_dict(self, source_text, target_text,  do_persist=False, source_target_dict_file=None):
        """

        1.待翻译的源语句会对应多个人工翻译的标准句子, 因此需要组合源语句和目标语句, 并返回组合后的字典

        :param source_text:
        :param target_text:
        :param do_persist: 将结果持久化到磁盘
        :param source_target_dict_file:

        :return:
            source_target_dict
            = {
                '<START> republican leaders justified their... <END>' :
                 ['<START> Die Führungskräfte der Republikaner rechtfertigen... <END>' ]

              }
        """

        source_text = source_text.unbatch()  # 将批数据还原为一行一行
        target_text = target_text.unbatch()

        source_target_dict = {}

        for source, target in zip(source_text, target_text):

            key = source.numpy().decode('utf-8')
            source_target_dict[key] = [target.numpy().decode('utf-8')]


        if do_persist:

            save_dict = {}
            save_dict['source_target_dict'] = source_target_dict
            source_target_dict_path = os.path.join(self.cache_data_dir, source_target_dict_file)

            with open(source_target_dict_path, 'wb') as f:
                pickle.dump(save_dict, f)

        return source_target_dict

    def __get_seq_length(self, seq_list):
        """
        统计序列的长度

        :param seq_list:
        :return:
        """
        # length_dict = { 句子标号: 句子的长度,  }
        length_dict = {}

        for i, seq in enumerate(seq_list):

            seq_arr = seq.split(" ")

            length_dict[i] = len(seq_arr)

        return length_dict

    def statistic_seq_length(self, length_dict):
        """
        统计语料中句子的长度的分布

        :param length_dict: 序列的长度字典
                    length_dict = { 句子标号: 句子的长度,  }
        :return:
        """

        length_counter = Counter(length_dict.values())
        # length_counter = { 句子的长度: 该长度的句子的个数 }

        print('most common seq length: (seq length, count num)')

        print(length_counter.most_common(10))

        length_counter_list = sorted(length_counter.items(), key=lambda t: t[0], reverse=True)

        print('seq length count:  (seq length, count num)')

        print(length_counter_list)

        return length_counter_list

    def filter_by_seq_length(self, source_seq_list, target_seq_list, max_seq_length):
        """
        过滤掉超出设定长度的序列

        :param source_seq_list:
        :param target_seq_list:
        :param max_seq_length:
        :return:
        """
        assert len(source_seq_list) == len(target_seq_list)  # 平行语料, 序列的个数必须相同

        source_length_dict = self.__get_seq_length(source_seq_list)
        target_length_dict = self.__get_seq_length(target_seq_list)

        # 统计语料中句子的长度的分布
        _ = self.statistic_seq_length(source_length_dict)

        res_source_seq_list = []
        res_target_seq_list = []

        for i in range(len(source_seq_list)):

            if source_length_dict[i] <= max_seq_length and target_length_dict[i] <= max_seq_length:

                res_source_seq_list.append(source_seq_list[i])
                res_target_seq_list.append(target_seq_list[i])

        return res_source_seq_list, res_target_seq_list


    def do_mian(self, batch_size, n_vocab_source, n_vocab_target, max_seq_length, test_max_seq_length):
        """
        数据集预处理的主流程

        :param batch_size: 一个批次的大小
        :param n_vocab_source: 源语言的词典大小
        :param n_vocab_target: 目标语言的词典大小
        :param max_seq_length: 最大序列的长度(训练集和验证集)
        :param test_max_seq_length: 最大序列的长度(测试集)

        :return:
        """

        # 1.训练数据处理
        print('preprocess the train dataset...')

        train_source_text = self.load_corpus_data(corpus_file_dir=self.train_source_corpus_dir)
        train_target_text = self.load_corpus_data(corpus_file_dir=self.train_target_corpus_dir)

        # 过滤掉长度 大于 max_seq_length 的序列
        train_source_text, train_target_text = self.filter_by_seq_length(train_source_text, train_target_text, max_seq_length)

        train_source_dataset, tokenizer_source, vocab_source_list = self.tokenize_corpus(text_data=train_source_text, n_vocab=n_vocab_source, batch_size=batch_size, tokenizer_file='tokenizer_source_model')
        train_target_dataset, tokenizer_target, vocab_target_list = self.tokenize_corpus(text_data=train_target_text, n_vocab=n_vocab_target, batch_size=batch_size, tokenizer_file='tokenizer_target_model')


        # 建立字典
        print('build the vocab...')
        vocab_source = Vocab(vocab_list=vocab_source_list, vocab_list_path=os.path.join(self.cache_data_dir, 'vocab_source.bin'))
        vocab_target = Vocab(vocab_list=vocab_target_list, vocab_list_path=os.path.join(self.cache_data_dir, 'vocab_target.bin'))

        # 训练数据集
        train_dataset = self.tf_data_pipline(train_source_dataset, train_target_dataset, mode='mid',
                                             do_persist=True, dataset_file='train_dataset.bin')


        # 2.验证数据处理
        print('preprocess the valid dataset...')

        valid_source_text = self.load_corpus_data(corpus_file_dir=self.valid_source_corpus_dir)
        valid_target_text = self.load_corpus_data(corpus_file_dir=self.valid_target_corpus_dir)

        # 过滤掉长度 大于 max_seq_length 的序列
        valid_source_text, valid_target_text = self.filter_by_seq_length(valid_source_text, valid_target_text, max_seq_length)

        valid_source_dataset = self.preprocess_corpus(valid_source_text,  batch_size=batch_size)
        valid_target_dataset = self.preprocess_corpus(valid_target_text,  batch_size=batch_size)

        # 验证数据集
        valid_dataset = self.tf_data_pipline(valid_source_dataset, valid_target_dataset, mode='mid',
                                             do_persist=True, dataset_file='valid_dataset.bin')

        # 验证数据 source_target_dict
        self.build_source_target_dict(valid_source_dataset, valid_target_dataset, do_persist=True, source_target_dict_file='valid_source_target_dict.bin')

        # 3.测试数据处理
        print('preprocess the test dataset...')

        test_source_text = self.load_corpus_data(corpus_file_dir=self.test_source_corpus_dir)
        test_target_text = self.load_corpus_data(corpus_file_dir=self.test_target_corpus_dir)

        # 过滤掉长度 大于 test_max_seq_length 的序列
        test_source_text, test_target_text = self.filter_by_seq_length(test_source_text, test_target_text, max_seq_length=test_max_seq_length)

        test_source_dataset = self.preprocess_corpus(test_source_text, batch_size=batch_size)
        test_target_dataset = self.preprocess_corpus(test_target_text, batch_size=batch_size)

        # 测试数据 source_target_dict
        self.build_source_target_dict(test_source_dataset, test_target_dataset, do_persist=True, source_target_dict_file='test_source_target_dict.bin')


class Vocab:

    def __init__(self, vocab_list_path, vocab_list=None):
        """
        词典的初始化

        :param vocab_path: 词典持久化的路径
        :param vocab_list:  (建立新的词典时必填)

        """
        self.vocab_list_path = vocab_list_path
        
        if vocab_list is not None:  # 建立新的词典

            save_dict = {}

            self.vocab_list = vocab_list

            self.n_vocab = len(self.vocab_list)  # 字典的长度

            save_dict['vocab_list'] = vocab_list

            with open(self.vocab_list_path, 'wb') as f:

                pickle.dump(save_dict, f)

        else:  # 读取已有的词典

            with open(self.vocab_list_path, 'rb') as f:
                save_dict = pickle.load(f)

            self.vocab_list = save_dict['vocab_list']
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


class WMT14_Eng_Ge_Dataset:
    """
    包装了 WMT14 English Germa 数据集,  我们通过此类来访问该数据集

    1.使用之前先对数据集进行预处理(class DataPreprocess),
    预处理后的数据集在 dataset/cache_data 目录下

    Author: xrh
    Date: 2021-11-15

    """
    def __init__(self,
                 base_dir='../dataset/WMT-14-English-Germa',
                 cache_data_folder='cache_data',

                 vocab_source_file='vocab_source.bin',
                 vocab_target_file='vocab_target.bin',

                 tokenizer_source_file='tokenizer_source_model',
                 tokenizer_target_file='tokenizer_target_model',

                 train_dataset_file='train_dataset.bin',
                 valid_dataset_file='valid_dataset.bin',

                 valid_source_target_dict_file='valid_source_target_dict.bin',
                 test_source_target_dict_file='test_source_target_dict.bin',

                 mode='train'
                 ):
        """

        :param base_dir: 数据集的根路径
        :param cache_data_folder: 预处理结果文件夹
        :param vocab_source_file: 源语言词典路径
        :param vocab_target_file: 目标语言词典路径
        :param train_dataset_file:
        :param valid_dataset_file:
        :param valid_source_target_dict_file:
        :param test_source_target_dict_file:
        :param mode: 当前处在的模式, 可以选择
                    'train' - 训练模式
                    'infer' - 推理模式

        """
        self.cache_data_dir = os.path.join(base_dir, cache_data_folder)

        self.vocab_source = Vocab(os.path.join(self.cache_data_dir, vocab_source_file))
        self.vocab_target = Vocab(os.path.join(self.cache_data_dir, vocab_target_file))

        self.tokenizer_source = tf.keras.models.load_model(os.path.join(self.cache_data_dir, tokenizer_source_file))
        self.tokenizer_target = tf.keras.models.load_model(os.path.join(self.cache_data_dir, tokenizer_target_file))

        preffix = ''

        if mode == 'train':

            train_dataset_path = os.path.join(self.cache_data_dir, '{}{}'.format(preffix, train_dataset_file))
            valid_dataset_path = os.path.join(self.cache_data_dir, '{}{}'.format(preffix, valid_dataset_file))

            # 训练数据
            self.train_dataset = tf.data.experimental.load(train_dataset_path)

            # 验证数据
            self.valid_dataset = tf.data.experimental.load(valid_dataset_path)

            valid_source_target_dict_path = os.path.join(self.cache_data_dir, '{}{}'.format(preffix, valid_source_target_dict_file))

            with open(valid_source_target_dict_path, 'rb') as f:
                save_dict = pickle.load(f)

            self.valid_source_target_dict = save_dict['source_target_dict']


        elif mode == 'infer':

            test_source_target_dict_path = os.path.join(self.cache_data_dir, '{}{}'.format(preffix, test_source_target_dict_file))

            with open(test_source_target_dict_path, 'rb') as f:
                save_dict = pickle.load(f)

            self.test_source_target_dict = save_dict['source_target_dict']


class Test:

    def test_DataPreprocess(self, config_path='config.ini', tag='DEFAULT'):

        config = configparser.ConfigParser()
        config.read(config_path, 'utf-8')
        current_config = config[tag]

        process_obj = DataPreprocess(config_path=config_path, tag=tag, cache_data_folder=current_config['cache_data_folder'])

        process_obj.do_mian(batch_size=int(current_config['batch_size']), n_vocab_source=int(current_config['n_vocab_source']),
                            n_vocab_target=int(current_config['n_vocab_target']), max_seq_length=int(current_config['max_seq_length']), test_max_seq_length=int(current_config['test_max_seq_length']))

    def test_WMT14_Eng_Ge_Dataset(self, config_path='config.ini', tag='DEFAULT'):

        config = configparser.ConfigParser()
        config.read(config_path, 'utf-8')
        current_config = config[tag]

        dataset_train_obj = WMT14_Eng_Ge_Dataset(cache_data_folder=current_config['cache_data_folder'], mode='train')
        dataset_infer_obj = WMT14_Eng_Ge_Dataset(cache_data_folder=current_config['cache_data_folder'], mode='infer')


        # 查看 1 个批次的数据
        for source, target in tqdm(dataset_train_obj.train_dataset.take(1)):

            print('source:')
            print(source[:10])

            source_vector = dataset_train_obj.tokenizer_source(source).to_tensor()
            print('source_vector:')
            print(source_vector)

            print('target:')
            print(target[:10])



        print('test_source_target_dict: ')
        print(list(dataset_infer_obj.test_source_target_dict.items())[0])

        print('vocab_source: ')

        print('word [START] index: ', dataset_infer_obj.vocab_source.map_word_to_id('[START]'))
        print('word [END] index: ', dataset_infer_obj.vocab_source.map_word_to_id('[END]'))
        print('word [UNK] index: ', dataset_infer_obj.vocab_source.map_word_to_id('[UNK]'))
        print('word [NULL] index: ', dataset_infer_obj.vocab_source.map_word_to_id(''))
        print('word a index: ', dataset_infer_obj.vocab_source.map_word_to_id('a'))
        print('word -= index: ', dataset_infer_obj.vocab_source.map_word_to_id('-='))


        print('vocab_target: ')

        print('word [START] index: ', dataset_infer_obj.vocab_target.map_word_to_id('[START]'))
        print('word [END] index: ', dataset_infer_obj.vocab_target.map_word_to_id('[END]'))
        print('word [UNK] index: ', dataset_infer_obj.vocab_target.map_word_to_id('[UNK]'))
        print('word [NULL] index: ', dataset_infer_obj.vocab_target.map_word_to_id(''))

        print('word nämlich index: ', dataset_infer_obj.vocab_target.map_word_to_id('nämlich'))

        print('word würden index: ', dataset_infer_obj.vocab_target.map_word_to_id('würden'))

        print('word üblich index: ', dataset_infer_obj.vocab_target.map_word_to_id('üblich'))

        print('word Ämter index: ', int(dataset_infer_obj.vocab_target.map_word_to_id('Ämter')))

        print('word ämter index: ', int(dataset_infer_obj.vocab_target.map_word_to_id('ämter')))

if __name__ == '__main__':
    test = Test()

    #TODO：运行之前 把 jupyter notebook 停掉, 否则会出现争抢 GPU 导致报错

    # test.test_DataPreprocess(tag='TEST')

    test.test_WMT14_Eng_Ge_Dataset(tag='TEST')
