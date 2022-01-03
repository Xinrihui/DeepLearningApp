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

import tensorflow_text as tf_text

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

from lib.data_generator.tokenizer_xrh import *

from lib.data_generator.vocab_xrh import *

from lib.data_generator.dynamic_batch_xrh import *


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
        对每一个句子 在首尾加入控制字符, 不做其他标准化操作

        :param text:
        :return:
        """

        # 左右端点添加 开始 和 结束的控制字符
        text = tf.strings.join([self._start_str, text, self._end_str], separator=' ')

        return text

    def normalize(self, text):
        """
        对每一个句子进行 unicode 标准化

        :param text:
        :return:
        """
        # 删除在各个语系外的字符
        # text = tf.strings.regex_replace(text, self.remove_unk, ' ')

        # NKFC unicode 标准化 +  大小写折叠
        # text = tf_text.case_fold_utf8(text)

        text = tf_text.normalize_utf8(text)  # NKFC unicode 标准化

        return text

    def all(self, text):
        """
        对每一个句子进行所有预处理操作, 包括 unicode 标准化和在首尾加入控制字符

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


class DatasetGenerate:
    """
    利用 tf.data.Dataset 数据流水线的数据集生成

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
                 base_dir='../../dataset/WMT-14-English-Germa',
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
        self.preprocess_mode = current_config['preprocess_mode']

        self.fixed_seq_length = int(self.current_config['fixed_seq_length'])

        self.reverse_source = bool(int(self.current_config['reverse_source']))

        self.train_source_corpus_dir = os.path.join(base_dir, current_config['train_source_corpus'])
        self.train_target_corpus_dir = os.path.join(base_dir, current_config['train_target_corpus'])

        self.valid_source_corpus_dir = os.path.join(base_dir, current_config['valid_source_corpus'])
        self.valid_target_corpus_dir = os.path.join(base_dir, current_config['valid_target_corpus'])

        self.test_source_corpus_dir = os.path.join(base_dir, current_config['test_source_corpus'])
        self.test_target_corpus_dir = os.path.join(base_dir, current_config['test_target_corpus'])

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

    def preprocess_corpus(self, text_data, preprocess_mode=None, batch_size=64):
        """
        对语料库中的句子进行预处理

        :param text_data:
        :param preprocess_mode: 预处理模式
        :param batch_size:

        :return:
        """

        if preprocess_mode is None:  # 若未定义局部的 preprocess_mode 则使用全局的
            preprocess_mode = self.preprocess_mode

        text_dataset = tf.data.Dataset.from_tensor_slices(text_data).batch(batch_size)  # 分块后可以加速计算, 后面每次读取都是一块

        if preprocess_mode == 'none':

            text_preprocess_dataset = text_dataset

        elif preprocess_mode == 'normalize':

            text_preprocess_dataset = text_dataset.map(lambda batch_text:
                                                       self.corpus_normalize.normalize(batch_text),
                                                       num_parallel_calls=tf.data.AUTOTUNE)


        elif preprocess_mode == 'add_control_token':

            text_preprocess_dataset = text_dataset.map(lambda batch_text:
                                                       self.corpus_normalize.add_control_token(batch_text),
                                                       num_parallel_calls=tf.data.AUTOTUNE)

        elif preprocess_mode == 'all':

            text_preprocess_dataset = text_dataset.map(lambda batch_text:
                                                       self.corpus_normalize.all(batch_text),
                                                       num_parallel_calls=tf.data.AUTOTUNE)

        else:

            raise Exception('the value of preprocess_mode is {}, which is illegal'.format(preprocess_mode))

        return text_preprocess_dataset

    def build_subword_tokenizer(self, preprocess_dataset, n_vocab, tokenizer_file, do_persist=True, batch_size=64):
        """
        构建 subword 分词器

        :param preprocess_dataset: 预处理后的数据集
        :param n_vocab: 词表大小
        :param tokenizer_file: 标记模型的文件
        :param do_persist: 是否持久化标记模型
        :param batch_size:

        :return:
        """

        reserved_tokens = [self._null_str, self._unk_str, self._start_str, self._end_str]

        bert_tokenizer_params = dict(lower_case=True)

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
            preprocess_dataset.unbatch().batch(2048).prefetch(buffer_size=tf.data.AUTOTUNE),
            **bert_vocab_args
        )

        print('vocab_list length: {}'.format(len(vocab_list)))

        print('vocab_list: ', vocab_list[:20])

        tokenizer = SubwordTokenizer(bert_tokenizer_params=bert_tokenizer_params,
                                     fixed_seq_length=self.fixed_seq_length,
                                     reserved_tokens=reserved_tokens, vocab_list=vocab_list,
                                     _start_str=self._start_str, _end_str=self._end_str
                                     )
        # (1) -fixed_seq_length= org_seq_length+increment, 前面的处理中已经删除了在语料库中长度超过 org_seq_length 的句子
        #       +increment 的原因是考虑句子的前后补充了2个控制字符, 另外将 word 变为 subword 必然导致句子的长度增加

        if do_persist:
            # 持久化 tokenizer
            tokenizer_path = os.path.join(self.cache_data_dir, tokenizer_file)
            tf.saved_model.save(tokenizer, tokenizer_path)

        return tokenizer, vocab_list

    def build_space_tokenizer(self, preprocess_dataset, n_vocab, tokenizer_file, do_persist=True, batch_size=64):
        """

        构建传统的空格分词器(tokenizer)

        :param preprocess_dataset: 预处理后的数据集
        :param n_vocab: 词表大小
        :param tokenizer_file: 标记模型的文件
        :param do_persist: 是否持久化标记模型
        :param batch_size:

        :return:
        """

        tokenizer = SpaceTokenizer(corpus=preprocess_dataset, max_tokens=n_vocab,
                                   fixed_seq_length=self.fixed_seq_length)
        # (1) max_tokens: 保留出现次数最多的 top-k 个单词
        # (2) -fixed_seq_length=None, 自动将本 batch 中最长的句子的长度作为序列的长度,
        #      在 padding 时 不同的 batch 的序列的长度可能不同, 但是在同个 batch 中自然是统一的;
        #     -fixed_seq_length = org_seq_length+increment, 前面的处理中已经删除了在语料库中
        #      长度超过 org_seq_length 的句子, +increment=2 的原因是考虑句子的前后补充了 2个控制字符

        vocab_list = tokenizer.vocab_list

        print('vocab_list length: {}'.format(len(vocab_list)))

        print('vocab_list: ', vocab_list[:20])

        if do_persist:
            # 持久化 tokenizer
            tokenizer_path = os.path.join(self.cache_data_dir, tokenizer_file)

            tf.saved_model.save(tokenizer, tokenizer_path)

        return tokenizer, vocab_list

    def tf_data_pipline(self, source_dataset, target_dataset, tokenizer_source=None, tokenizer_target=None,
                        do_persist=False, dataset_file=None):
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

        if self.return_mode == 'fixed_length':  # 返回符号化后的句子, 并保证所有 batch 中序列的长度均相同

            # 将文本标记化, 所有 batch 的 序列的长度均相同
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

        elif self.return_mode == 'tokenized':  # 返回符号化后的句子

            source_vector = source_dataset.map(lambda batch_text: tokenizer_source.tokenize(batch_text).to_tensor())
            target_vector = target_dataset.map(lambda batch_text: tokenizer_target.tokenize(batch_text).to_tensor())

            print('tokenize text complete!')

            dataset = tf.data.Dataset.zip((source_vector, target_vector))


        elif self.return_mode == 'dynamic_batch':  # 返回符号化后的句子, 并动态划分 batch

            source_vector = source_dataset.map(lambda batch_text: tokenizer_source.tokenize(batch_text))
            target_vector = target_dataset.map(lambda batch_text: tokenizer_target.tokenize(batch_text))

            print('tokenize text complete!')

            dataset = tf.data.Dataset.zip((source_vector, target_vector)).unbatch()

            # 划分动态的 batch
            ret = batching_scheme(batch_size=int(self.current_config['token_in_batch']))

            dataset = dataset.map(lambda source, target: (source, target))  # 将数据集(Dataset)转换为生成器(generato)

            dataset = dataset.bucket_by_sequence_length(
                element_length_func=lambda source, target: tf.cast(tf.maximum(tf.shape(source)[0], tf.shape(target)[0]),
                                                                   tf.int32),
                bucket_boundaries=ret["boundaries"],
                bucket_batch_sizes=ret["batch_sizes"],
            )
            # tf.maximum(tf.shape(source)[0], tf.shape(target)[0]) 取源序列长度和目标序列长度的最大值作为划分桶的依据


        elif self.return_mode == 'text':  # 返回文本形式的句子, 模型需要做进一步预处理

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
        过滤掉超出设定长度(句子中用空格分隔的 token 的个数)的序列

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

    def sort_by_seq_length(self, source_seq_list, target_seq_list):
        """
        按照长度(句子中用空格分隔的 token 的个数))对 序列 进行排序,
        注意 source_seq_list 和 target_seq_list 要按照原来的顺序对应上

        :param source_seq_list:
        :param target_seq_list:
        :return:
        """

        assert len(source_seq_list) == len(target_seq_list)  # 平行语料, 序列的个数必须相同

        # 使用 source seq 的长度进行排序
        source_length_dict = self.__get_seq_length(source_seq_list)

        source_ordered_list = sorted(source_length_dict.items(), key=lambda d: d[1], reverse=True)

        source_ordered_idx = [id for id, length in source_ordered_list]

        source_seq_arr = np.array(source_seq_list)
        target_seq_arr = np.array(target_seq_list)

        source_seq_arr = source_seq_arr[source_ordered_idx]
        target_seq_arr = target_seq_arr[source_ordered_idx]

        return list(source_seq_arr), list(target_seq_arr)

    def do_mian(self, batch_size, build_tokenizer, n_vocab_source, n_vocab_target, org_seq_length, test_org_seq_length):
        """
        数据集生成的主流程

        :param batch_size: 一个批次的大小
        :param build_tokenizer: 是否建立分词器
        :param n_vocab_source: 源语言的词典大小
        :param n_vocab_target: 目标语言的词典大小
        :param org_seq_length: 最大序列的长度(训练集和验证集)
        :param test_org_seq_length: 最大序列的长度(测试集)

        :return:
        """

        # 1.训练数据处理
        print('--------------------------------')
        print('preprocess the train dataset...')

        train_source_text = self.load_corpus_data(corpus_file_dir=self.train_source_corpus_dir)
        train_target_text = self.load_corpus_data(corpus_file_dir=self.train_target_corpus_dir)

        # 过滤掉长度 大于 org_seq_length 的序列
        train_source_text, train_target_text = self.filter_by_seq_length(train_source_text, train_target_text,
                                                                         org_seq_length)

        # 按照长度对 序列 进行排序
        train_source_text, train_target_text = self.sort_by_seq_length(train_source_text, train_target_text)

        train_source_dataset = self.preprocess_corpus(train_source_text, batch_size=batch_size)
        train_target_dataset = self.preprocess_corpus(train_target_text, batch_size=batch_size)

        if build_tokenizer:  # 重新建立分词器

            # 建立分词器
            if self.tokenize_mode == 'space':

                print('build source {} tokenizer ...'.format(self.tokenize_mode))
                tokenizer_source, vocab_source_list = self.build_space_tokenizer(
                    preprocess_dataset=train_source_dataset, n_vocab=n_vocab_source, batch_size=batch_size,
                    tokenizer_file='tokenizer_source_model')

                print('build target {} tokenizer ...'.format(self.tokenize_mode))
                tokenizer_target, vocab_target_list = self.build_space_tokenizer(
                    preprocess_dataset=train_target_dataset, n_vocab=n_vocab_target, batch_size=batch_size,
                    tokenizer_file='tokenizer_target_model')

            elif self.tokenize_mode == 'subword':

                print('build source {} tokenizer ...'.format(self.tokenize_mode))
                tokenizer_source, vocab_source_list = self.build_subword_tokenizer(
                    preprocess_dataset=train_source_dataset, n_vocab=n_vocab_source, batch_size=batch_size,
                    tokenizer_file='tokenizer_source_model')

                print('build target {} tokenizer ...'.format(self.tokenize_mode))
                tokenizer_target, vocab_target_list = self.build_subword_tokenizer(
                    preprocess_dataset=train_target_dataset, n_vocab=n_vocab_target, batch_size=batch_size,
                    tokenizer_file='tokenizer_target_model')

            else:
                raise Exception('the value of tokenize_mode is {}, which is illegal'.format(self.tokenize_mode))

            # 建立字典
            print('--------------------------------')
            print('build the vocab...')
            vocab_source = VocabTf(vocab_list=vocab_source_list,
                                   vocab_list_path=os.path.join(self.cache_data_dir, 'vocab_source.txt'))
            vocab_target = VocabTf(vocab_list=vocab_target_list,
                                   vocab_list_path=os.path.join(self.cache_data_dir, 'vocab_target.txt'))

            # vocab_source = Vocab(vocab_list=vocab_source_list, vocab_list_path=os.path.join(self.cache_data_dir, 'vocab_source.txt'))
            # vocab_target = Vocab(vocab_list=vocab_target_list, vocab_list_path=os.path.join(self.cache_data_dir, 'vocab_target.txt'))

        else:  # 已有分词器
            tokenizer_source = tf.saved_model.load(os.path.join(self.cache_data_dir, 'tokenizer_source_model'))
            tokenizer_target = tf.saved_model.load(os.path.join(self.cache_data_dir, 'tokenizer_target_model'))

        # 形成训练数据集并持久化
        train_dataset = self.tf_data_pipline(train_source_dataset, train_target_dataset,
                                             tokenizer_source=tokenizer_source, tokenizer_target=tokenizer_target,
                                             do_persist=True, dataset_file='train_dataset.bin')

        # 2.验证数据处理
        print('--------------------------------')
        print('preprocess the valid dataset...')

        valid_source_text = self.load_corpus_data(corpus_file_dir=self.valid_source_corpus_dir)
        valid_target_text = self.load_corpus_data(corpus_file_dir=self.valid_target_corpus_dir)

        # 过滤掉长度 大于 org_seq_length 的序列
        valid_source_text, valid_target_text = self.filter_by_seq_length(valid_source_text, valid_target_text,
                                                                         org_seq_length)

        # 按照长度对 序列 进行排序
        valid_source_text, valid_target_text = self.sort_by_seq_length(valid_source_text, valid_target_text)

        valid_source_dataset = self.preprocess_corpus(valid_source_text, batch_size=batch_size)
        valid_target_dataset = self.preprocess_corpus(valid_target_text, batch_size=batch_size)

        # 形成训练数据集并持久化
        valid_dataset = self.tf_data_pipline(valid_source_dataset, valid_target_dataset,
                                             tokenizer_source=tokenizer_source, tokenizer_target=tokenizer_target,
                                             do_persist=True, dataset_file='valid_dataset.bin')

        # 验证数据 source_target_dict
        self.build_source_target_dict(
            source_text=self.preprocess_corpus(valid_source_text, preprocess_mode='add_control_token',
                                               batch_size=batch_size),
            target_text=self.preprocess_corpus(valid_target_text, preprocess_mode='add_control_token',
                                               batch_size=batch_size),
            do_persist=True, source_target_dict_file='valid_source_target_dict.bin')

        # 3.测试数据处理
        print('--------------------------------')
        print('preprocess the test dataset...')

        test_source_text = self.load_corpus_data(corpus_file_dir=self.test_source_corpus_dir)
        test_target_text = self.load_corpus_data(corpus_file_dir=self.test_target_corpus_dir)

        # 过滤掉长度 大于 test_org_seq_length 的序列
        test_source_text, test_target_text = self.filter_by_seq_length(test_source_text, test_target_text,
                                                                       test_org_seq_length)

        # 按照长度对 序列 进行排序
        test_source_text, test_target_text = self.sort_by_seq_length(test_source_text, test_target_text)

        # 测试数据 source_target_dict
        self.build_source_target_dict(
            source_text=self.preprocess_corpus(test_source_text, preprocess_mode='add_control_token',
                                               batch_size=batch_size),
            target_text=self.preprocess_corpus(test_target_text, preprocess_mode='add_control_token',
                                               batch_size=batch_size),
            do_persist=True, source_target_dict_file='test_source_target_dict.bin')


class WMT14_Eng_Ge_Dataset:
    """
    包装了 WMT14 English Germa 数据集,  我们通过此类来访问该数据集

    1.使用之前先对数据集进行预处理(class DataPreprocess),
    预处理后的数据集在 dataset/cache_data 目录下

    Author: xrh
    Date: 2021-11-15

    """

    def __init__(self,
                 base_dir='../../dataset/WMT-14-English-Germa',
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

            valid_source_target_dict_path = os.path.join(self.cache_data_dir,
                                                         '{}{}'.format(preffix, valid_source_target_dict_file))

            with open(valid_source_target_dict_path, 'rb') as f:
                save_dict = pickle.load(f)

            self.valid_source_target_dict = save_dict['source_target_dict']


        elif mode == 'infer':

            test_source_target_dict_path = os.path.join(self.cache_data_dir,
                                                        '{}{}'.format(preffix, test_source_target_dict_file))

            with open(test_source_target_dict_path, 'rb') as f:
                save_dict = pickle.load(f)

            self.test_source_target_dict = save_dict['source_target_dict']


class Test:

    def test_DatasetGenerate(self, build_tokenizer=True,
                             config_path='../../config/transformer_seq2seq.ini',
                             base_dir='../../dataset/WMT-14-English-Germa',
                             tag='DEFAULT'):

        config = configparser.ConfigParser()
        config.read(config_path, 'utf-8')
        current_config = config[tag]

        process_obj = DatasetGenerate(config_path=config_path, tag=tag,
                                      base_dir=base_dir,
                                      cache_data_folder=current_config['cache_data_folder'])

        process_obj.do_mian(batch_size=int(current_config['batch_size']), build_tokenizer=build_tokenizer,
                            n_vocab_source=int(current_config['n_vocab_source']),
                            n_vocab_target=int(current_config['n_vocab_target']),
                            org_seq_length=int(current_config['org_seq_length']),
                            test_org_seq_length=int(current_config['test_org_seq_length']))

    def test_WMT14_Eng_Ge_Dataset(self,
                                  config_path='../../config/transformer_seq2seq.ini',
                                  base_dir='../../dataset/WMT-14-English-Germa',
                                  tag='DEFAULT'):

        config = configparser.ConfigParser()
        config.read(config_path, 'utf-8')
        current_config = config[tag]

        dataset_train_obj = WMT14_Eng_Ge_Dataset(cache_data_folder=current_config['cache_data_folder'], base_dir=base_dir, mode='train')
        dataset_infer_obj = WMT14_Eng_Ge_Dataset(cache_data_folder=current_config['cache_data_folder'], base_dir=base_dir, mode='infer')

        if current_config['return_mode'] == 'text':

            # 查看 1 个批次的数据
            for source, target in tqdm(dataset_train_obj.train_dataset.take(1)):

                # source = source[:4]
                # target = target[:4]

                print('source:')
                print(source)

                source_vector = dataset_train_obj.tokenizer_source.tokenize(source).to_tensor()
                print('source_vector:')
                print(source_vector)

                print('detokenize source vector')
                detoken = dataset_train_obj.tokenizer_source.detokenize(source_vector)
                print(tf.strings.reduce_join(detoken, separator=' ', axis=-1))

                print('target:')
                print(target)

                target_vector = dataset_train_obj.tokenizer_target.tokenize(target).to_tensor()
                print('target_vector:')
                print(target_vector)

                print('detokenize target vector')
                detoken = dataset_train_obj.tokenizer_target.detokenize(target_vector)
                print(tf.strings.reduce_join(detoken, separator=' ', axis=-1))


        elif current_config['return_mode'] == 'fixed_length':

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

        elif current_config['return_mode'] in ('tokenized', 'dynamic_batch'):

            dataset = dataset_train_obj.train_dataset

            dataset = dataset.shuffle(int(current_config['buffer_size']))
            # shuffle 的粒度为 batch

            # 查看 1 个批次的数据
            for batch_feature in tqdm(dataset.take(3)):
                source_vector = batch_feature[0]
                target_vector = batch_feature[1]

                print('source_vector:')
                print(source_vector)

                print('detokenize source vector')
                detoken = dataset_train_obj.tokenizer_source.detokenize(source_vector)
                print(tf.strings.reduce_join(detoken, separator=' ', axis=-1))

                print('target_vector:')
                print(target_vector)

                print('detokenize target vector')
                detoken = dataset_train_obj.tokenizer_target.detokenize(target_vector)
                print(tf.strings.reduce_join(detoken, separator=' ', axis=-1))

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


if __name__ == '__main__':
    test = Test()

    # TODO：运行之前 把 jupyter notebook 停掉, 否则会出现争抢 GPU 导致报错

    # test.test_DatasetGenerate(build_tokenizer=False,
    #                           config_path='../../config/transformer_seq2seq.ini',
    #                           base_dir='../../dataset/TED-Portuguese-English',
    #                           tag='TEST-1')  # DEFAULT

    test.test_WMT14_Eng_Ge_Dataset(config_path='../../config/transformer_seq2seq.ini',
                                   base_dir='../../dataset/TED-Portuguese-English',
                                   tag='TEST-1')  # DEFAULT
