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


import tensorflow_text as tf_text

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

from lib.tokenizer_xrh import *

class CorpusNormalize:
    """
    使用 tensorflow 库函数 对语料库中的句子进行标准化

    Author: xrh
    Date: 2021-12-1

    """

    def __init__(self, _start_str, _end_str):
        """

        :param _start_str: 代表句子开始的控制字符
        :param _end_str: 代表句子结束的控制字符
        """
        self._start_str = _start_str
        self._end_str = _end_str

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


    def add_control_token(self, text):
        """
        对每一个句子 只在首尾加入控制字符, 不做其他标准化操作

        :param text:
        :return:
        """

        # 左右端点添加 开始 和 结束的控制字符
        text = tf.strings.join([self._start_str, text, self._end_str], separator=' ')

        return text

    def standard_normalize(self, text):
        """
        对每一个句子的标准化操作, 并在首尾加入控制字符

        :param text:
        :return:
        """
        # 删除在各个语系外的字符
        text = tf.strings.regex_replace(text, self.remove_unk, ' ')

        # 清除指定单词
        text = tf.strings.regex_replace(text, self.remove_words, '')

        # NKFC unicode 标准化 +  大小写折叠
        text = tf_text.case_fold_utf8(text)

        # text = tf_text.normalize_utf8(text)  # NKFC unicode 标准化
        # text = tf.strings.lower(text) # 小写化

        # 清除句子中的标点符号
        text = tf.strings.regex_replace(text, self.remove_punc, ' ')  # 空1格

        # 清除句子中的独立数字
        text = tf.strings.regex_replace(text, self.remove_digits, '')

        # 清除左右端点的空格
        text = tf.strings.strip(text)

        text = self.add_control_token(text)

        return text




class DataPreprocess:
    """
    利用 tf.data.Dataset 数据流水线 + TextVectorization 的数据集预处理

    主流程见 do_main()

    Author: xrh
    Date: 2021-11-15

    ref:
    https://tensorflow.google.cn/text/tutorials/nmt_with_attention
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

        self.current_config = current_config

        print('current config tag:{}'.format(tag))

        self.tokenize_mode = current_config['tokenize_mode']
        self.return_mode = current_config['return_mode']
        self.normalize_mode = current_config['normalize_mode']

        self.max_seq_length = int(self.current_config['max_seq_length'])
        self.increment = int(self.current_config['increment'])

        self.reverse_source = bool(int(self.current_config['reverse_source']))

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

        self.corpus_normalize = CorpusNormalize(_start_str=self._start_str, _end_str=self._end_str)


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

    def preprocess_corpus(self, text_data, normalize_mode=None, batch_size=64):
        """
        对语料库中的句子进行预处理, 包括 标准化文本

        :param text_data:
        :param normalize_mode: 预处理模式
        :param batch_size:

        :return:
        """

        if normalize_mode is None:
            normalize_mode = self.normalize_mode

        text_dataset = tf.data.Dataset.from_tensor_slices(text_data).batch(batch_size)  # 分块后可以加速计算, 后面每次读取都是一块

        if normalize_mode == 'add_control':

            text_preprocess_dataset = text_dataset.map(lambda batch_text:
                self.corpus_normalize.add_control_token(batch_text), num_parallel_calls=tf.data.AUTOTUNE)

        elif normalize_mode == 'standard':

            text_preprocess_dataset = text_dataset.map(lambda batch_text:
                self.corpus_normalize.standard_normalize(batch_text), num_parallel_calls=tf.data.AUTOTUNE)

        else:
            text_preprocess_dataset = text_dataset

        return text_preprocess_dataset

    def build_subword_tokenizer(self, text_data, n_vocab, tokenizer_file, do_persist=True, batch_size=64):
        """
        构建 subword 分词器

        :param text_data: 句子的列表
        :param n_vocab: 词表大小
        :param tokenizer_file: 标记模型的文件
        :param do_persist: 是否持久化标记模型
        :param batch_size:

        :return:
        """

        text_preprocess_dataset = self.preprocess_corpus(text_data, batch_size=batch_size)

        bert_tokenizer_params = dict(lower_case=False)
        reserved_tokens = [self._null_str, self._unk_str, self._start_str, self._end_str]

        bert_vocab_args = dict(
            # The target vocabulary size
            vocab_size=n_vocab,
            # Reserved tokens that must be included in the vocabulary
            reserved_tokens=reserved_tokens,
            # Arguments for `text.BertTokenizer`
            bert_tokenizer_params=bert_tokenizer_params,
            # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
            learn_params={},
        )

        vocab_list = bert_vocab.bert_vocab_from_dataset(
            text_preprocess_dataset.prefetch(buffer_size=tf.data.AUTOTUNE),
            **bert_vocab_args
        )

        print('vocab_list length: {}'.format(len(vocab_list)))

        print('vocab_list: ', vocab_list[:20])

        reserved_tokens = [self._null_str, self._unk_str, self._start_str, self._end_str]

        tokenizer = SubwordTokenizer(fixed_seq_length=self.max_seq_length+self.increment, reserved_tokens=reserved_tokens, vocab_list=vocab_list)
        # (1) -output_sequence_length=self.max_seq_length+20, 前面的处理中已经删除了在语料库中长度超过 max_seq_length 的句子
        #       +increment 的原因是考虑句子的前后补充了2个控制字符, 另外将 word 变为 subword 必然导致句子的长度增加

        if do_persist:
            # 持久化 tokenizer
            tokenizer_path = os.path.join(self.cache_data_dir, tokenizer_file)
            tf.saved_model.save(tokenizer, tokenizer_path)

        return text_preprocess_dataset, tokenizer, vocab_list

    def build_space_tokenizer(self, text_data, n_vocab, tokenizer_file, do_persist=True, batch_size=64):
        """

        构建传统的空格分词器(tokenizer)

        :param text_data: 句子的列表
        :param n_vocab: 词表大小
        :param tokenizer_file: 标记模型的文件
        :param do_persist: 是否持久化标记模型
        :param batch_size:

        :return:
        """

        text_preprocess_dataset = self.preprocess_corpus(text_data, batch_size=batch_size)

        tokenizer = SpaceTokenizer(corpus=text_preprocess_dataset, max_tokens=n_vocab, fixed_seq_length=self.max_seq_length + self.increment)
        # (1) max_tokens: 保留出现次数最多的 top-k 个单词
        # (2) -fixed_seq_length=None, 自动将本 batch 中最长的句子的长度作为序列的长度,
        #      在 padding 时 不同的 batch 的序列的长度可能不同, 但是在同个 batch 中自然是统一的;
        #     -fixed_seq_length=max_seq_length+increment, 前面的处理中已经删除了在语料库中
        #      长度超过 max_seq_length 的句子, +increment 的原因是考虑句子的前后补充了 2个控制字符

        vocab_list = tokenizer.vocab_list

        print('vocab_list length: {}'.format(len(vocab_list)))

        print('vocab_list: ', vocab_list[:20])

        if do_persist:
            # 持久化 tokenizer
            tokenizer_path = os.path.join(self.cache_data_dir, tokenizer_file)

            tf.saved_model.save(tokenizer, tokenizer_path)


        return text_preprocess_dataset, tokenizer, vocab_list

    def tf_data_pipline(self, source_dataset, target_dataset, tokenizer_source=None, tokenizer_target=None, do_persist=False, dataset_file=None):
        """
        利用 tf.data.Dataset 数据流水线 建立数据集,

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

        if self.return_mode == 'final':  # 返回最终的符号化后的句子, 模型可以直接加载进入训练

            # 将文本标记化, 每一个 batch 中的 序列的长度相同
            source_vector = source_dataset.map(lambda batch_text: tokenizer_source.tokenize_fixed(batch_text))
            target_vector = target_dataset.map(lambda batch_text: tokenizer_target.tokenize_fixed(batch_text))

            print('tokenize text complete!')

            # 将源序列倒置
            if self.reverse_source:
                source_vector = source_vector.map(lambda batch_text: batch_text[:, ::-1],
                                               num_parallel_calls=tf.data.AUTOTUNE)

            # 输入 和 输出的 target 要错开一位
            target_out = target_vector.map(lambda batch_text: batch_text[:, 1:], num_parallel_calls=tf.data.AUTOTUNE)
            target_in = target_vector.map(lambda batch_text: batch_text[:, :-1], num_parallel_calls=tf.data.AUTOTUNE)

            features = tf.data.Dataset.zip((source_vector, target_in))
            labels = target_out

            dataset = tf.data.Dataset.zip((features, labels))

        elif self.return_mode == 'mid':  # 返回文本形式的句子, 模型需要做进一步预处理

            dataset = tf.data.Dataset.zip((source_dataset, target_dataset))

        else:
            raise Exception('the value of return_mode is {}, which is illegal'.format(self.return_mode))

        if do_persist:
            dataset_path = os.path.join(self.cache_data_dir, dataset_file)
            tf.data.experimental.save(dataset, dataset_path)

        return dataset


    def build_source_target_dict(self, source_text, target_text, do_persist=False, source_target_dict_file=None):
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
        print('source sentence length distribution:')
        _ = self.statistic_seq_length(source_length_dict)

        # 统计语料中句子的长度的分布
        print('target sentence length distribution:')
        _ = self.statistic_seq_length(target_length_dict)

        res_source_seq_list = []
        res_target_seq_list = []

        for i in range(len(source_seq_list)):

            if source_length_dict[i] <= max_seq_length and target_length_dict[i] <= max_seq_length:

                res_source_seq_list.append(source_seq_list[i])
                res_target_seq_list.append(target_seq_list[i])

        assert len(res_source_seq_list) == len(res_target_seq_list)  # 平行语料, 序列的个数必须相同

        print('seq length <={} num: {}'.format(max_seq_length, len(res_source_seq_list)))

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
        print('--------------------------------')
        print('preprocess the train dataset...')

        train_source_text = self.load_corpus_data(corpus_file_dir=self.train_source_corpus_dir)
        train_target_text = self.load_corpus_data(corpus_file_dir=self.train_target_corpus_dir)

        # 过滤掉长度 大于 max_seq_length 的序列
        train_source_text, train_target_text = self.filter_by_seq_length(train_source_text, train_target_text, max_seq_length)


        if self.tokenize_mode == 'space':

            print('build {} tokenizer ...'.format(self.tokenize_mode))
            train_source_dataset, tokenizer_source, vocab_source_list = self.build_space_tokenizer(text_data=train_source_text, n_vocab=n_vocab_source, batch_size=batch_size, tokenizer_file='tokenizer_source_model')

            train_target_dataset, tokenizer_target, vocab_target_list = self.build_space_tokenizer(text_data=train_target_text, n_vocab=n_vocab_target, batch_size=batch_size, tokenizer_file='tokenizer_target_model')

        elif self.tokenize_mode == 'subword':

            print('build {} tokenizer ...'.format(self.tokenize_mode))
            train_source_dataset, tokenizer_source, vocab_source_list = self.build_subword_tokenizer(text_data=train_source_text, n_vocab=n_vocab_source, batch_size=batch_size, tokenizer_file='tokenizer_source_model')

            train_target_dataset, tokenizer_target, vocab_target_list = self.build_subword_tokenizer(text_data=train_target_text, n_vocab=n_vocab_target, batch_size=batch_size, tokenizer_file='tokenizer_target_model')

        else:
            raise Exception('the value of tokenize_mode is {}, which is illegal'.format(self.tokenize_mode))

        # 建立字典
        print('--------------------------------')
        print('build the vocab...')
        vocab_source = VocabTf(vocab_list=vocab_source_list, vocab_list_path=os.path.join(self.cache_data_dir, 'vocab_source.txt'))
        vocab_target = VocabTf(vocab_list=vocab_target_list, vocab_list_path=os.path.join(self.cache_data_dir, 'vocab_target.txt'))

        # vocab_source = Vocab(vocab_list=vocab_source_list, vocab_list_path=os.path.join(self.cache_data_dir, 'vocab_source.txt'))
        # vocab_target = Vocab(vocab_list=vocab_target_list, vocab_list_path=os.path.join(self.cache_data_dir, 'vocab_target.txt'))


        # 训练数据集
        train_dataset = self.tf_data_pipline(train_source_dataset, train_target_dataset, tokenizer_source=tokenizer_source, tokenizer_target=tokenizer_target,
                                             do_persist=True, dataset_file='train_dataset.bin')


        # 2.验证数据处理
        print('--------------------------------')
        print('preprocess the valid dataset...')

        valid_source_text = self.load_corpus_data(corpus_file_dir=self.valid_source_corpus_dir)
        valid_target_text = self.load_corpus_data(corpus_file_dir=self.valid_target_corpus_dir)

        # 过滤掉长度 大于 max_seq_length 的序列
        valid_source_text, valid_target_text = self.filter_by_seq_length(valid_source_text, valid_target_text, max_seq_length)

        valid_source_dataset = self.preprocess_corpus(valid_source_text,  batch_size=batch_size)
        valid_target_dataset = self.preprocess_corpus(valid_target_text,  batch_size=batch_size)

        # 验证数据集
        valid_dataset = self.tf_data_pipline(valid_source_dataset, valid_target_dataset, tokenizer_source=tokenizer_source, tokenizer_target=tokenizer_target,
                                             do_persist=True, dataset_file='valid_dataset.bin')


        # 验证数据 source_target_dict
        self.build_source_target_dict(source_text=self.preprocess_corpus(valid_source_text, normalize_mode='add_control', batch_size=batch_size),
                                      target_text=self.preprocess_corpus(valid_target_text, normalize_mode='add_control', batch_size=batch_size),
                                      do_persist=True, source_target_dict_file='valid_source_target_dict.bin')


        # 3.测试数据处理
        print('--------------------------------')
        print('preprocess the test dataset...')

        test_source_text = self.load_corpus_data(corpus_file_dir=self.test_source_corpus_dir)
        test_target_text = self.load_corpus_data(corpus_file_dir=self.test_target_corpus_dir)

        # 过滤掉长度 大于 test_max_seq_length 的序列
        test_source_text, test_target_text = self.filter_by_seq_length(test_source_text, test_target_text, max_seq_length=test_max_seq_length)


        # 测试数据 source_target_dict
        self.build_source_target_dict(source_text=self.preprocess_corpus(test_source_text, normalize_mode='add_control', batch_size=batch_size),
                                      target_text=self.preprocess_corpus(test_target_text, normalize_mode='add_control', batch_size=batch_size),
                                      do_persist=True, source_target_dict_file='test_source_target_dict.bin')


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

                 vocab_source_file='vocab_source.txt',
                 vocab_target_file='vocab_target.txt',

                 tokenizer_source_file='tokenizer_source_model',
                 tokenizer_target_file='tokenizer_target_model',

                 train_dataset_file='train_dataset.bin',
                 valid_dataset_file='valid_dataset.bin',

                 valid_source_target_dict_file='valid_source_target_dict.bin',
                 test_source_target_dict_file='test_source_target_dict.bin',

                 use_tf_vocab=True,
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

        :param use_tf_vocab: 是否使用 tf 的 StringLookup 构建的语料库词典

        :param mode: 当前处在的模式, 可以选择
                    'train' - 训练模式
                    'infer' - 推理模式

        """
        self.cache_data_dir = os.path.join(base_dir, cache_data_folder)

        if use_tf_vocab:

            self.vocab_source = VocabTf(os.path.join(self.cache_data_dir, vocab_source_file))
            self.vocab_target = VocabTf(os.path.join(self.cache_data_dir, vocab_target_file))

        else:

            self.vocab_source = Vocab(os.path.join(self.cache_data_dir, vocab_source_file))
            self.vocab_target = Vocab(os.path.join(self.cache_data_dir, vocab_target_file))


        self.tokenizer_source = tf.saved_model.load(os.path.join(self.cache_data_dir, tokenizer_source_file))
        self.tokenizer_target = tf.saved_model.load(os.path.join(self.cache_data_dir, tokenizer_target_file))

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

    def test_DataPreprocess(self, config_path='../config/transformer_seq2seq.ini', tag='DEFAULT'):

        config = configparser.ConfigParser()
        config.read(config_path, 'utf-8')
        current_config = config[tag]

        process_obj = DataPreprocess(config_path=config_path, tag=tag, cache_data_folder=current_config['cache_data_folder'])

        process_obj.do_mian(batch_size=int(current_config['batch_size']), n_vocab_source=int(current_config['n_vocab_source']),
                            n_vocab_target=int(current_config['n_vocab_target']), max_seq_length=int(current_config['max_seq_length']), test_max_seq_length=int(current_config['test_max_seq_length']))

    def test_WMT14_Eng_Ge_Dataset(self, config_path='../config/transformer_seq2seq.ini', tag='DEFAULT'):

        config = configparser.ConfigParser()
        config.read(config_path, 'utf-8')
        current_config = config[tag]

        dataset_train_obj = WMT14_Eng_Ge_Dataset(cache_data_folder=current_config['cache_data_folder'], mode='train')
        dataset_infer_obj = WMT14_Eng_Ge_Dataset(cache_data_folder=current_config['cache_data_folder'], mode='infer')

        if current_config['return_mode'] == 'mid':

            # 查看 1 个批次的数据
            for source, target in tqdm(dataset_train_obj.train_dataset.take(1)):
                print('source:')
                print(source[:10])

                source_vector = dataset_train_obj.tokenizer_source.tokenize(source).to_tensor()
                print('source_vector:')
                print(source_vector)

                print('detokenize source vector')
                detoken = dataset_train_obj.tokenizer_source.detokenize(source_vector)
                print(tf.strings.reduce_join(detoken, separator=' ', axis=-1))

                print('target:')
                print(target[:10])

                target_vector = dataset_train_obj.tokenizer_target.tokenize(target).to_tensor()
                print('target_vector:')
                print(target_vector)


        elif current_config['return_mode'] == 'final':

            dataset = dataset_train_obj.train_dataset.unbatch()
            # 保证所有的 batch 中序列的长度均一致, 才能进入后面的 shuffle

            dataset = dataset.shuffle(int(current_config['buffer_size'])).batch(2)

            # 查看 1 个批次的数据
            for batch_feature, batch_label in tqdm(dataset.take(1)):

                source_vector = batch_feature[0]

                target_in_vector = batch_feature[1]
                target_out_vector = batch_label

                print('source_vector:')
                print(source_vector)

                source = dataset_train_obj.vocab_source.map_id_to_word(source_vector)
                source = tf.strings.reduce_join(source[:, ::-1], axis=1, separator=' ')  # 单词序列 join 成句子
                print('source:')
                print(source)

                print('target_in_vector:')
                print(target_in_vector)

                target_in = dataset_train_obj.vocab_target.map_id_to_word(target_in_vector)
                target_in = tf.strings.reduce_join(target_in, axis=1, separator=' ')  # 单词序列 join 成句子
                print('target_in:')
                print(target_in)

                print('target_out_vector:')
                print(target_out_vector)


        print('rows of source_target_dict: ')

        for id, (source, target_list) in enumerate(list(dataset_infer_obj.test_source_target_dict.items())[:2]):

            print(id)
            print('source :')
            print(source)
            print(dataset_infer_obj.vocab_source.map_word_to_id(source.split()))

            print('target :')
            print(target_list[0])
            print(dataset_infer_obj.vocab_target.map_word_to_id(target_list[0].split()))


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


if __name__ == '__main__':
    test = Test()

    #TODO：运行之前 把 jupyter notebook 停掉, 否则会出现争抢 GPU 导致报错

    # test.test_DataPreprocess(tag='TEST')

    test.test_WMT14_Eng_Ge_Dataset(tag='TEST')

    # test.test_VocabTf(tag='TEST')

