#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
from tqdm import tqdm
import pickle
import re
from collections import *
import string

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
                 base_dir='../dataset/WMT-14-English-Germa',
                 cache_data_folder='cache_data',
                 source_tokenizer_file='tokenizer_source.bin',
                 target_tokenizer_file='tokenizer_target.bin',
                 _null_str='',
                 _start_str='[START]',
                 _end_str='[END]',
                 _unk_str='[UNK]',
                 tensor_int_type=tf.int32,
                 ):
        """
        :param base_dir:  数据集的路径
        :param cache_data_folder: 预处理结果文件夹
        :param source_tokenizer_file: 源语言的词典路径
        :param target_tokenizer_file: 目标语言的词典路径
        :param  _null_str: 空字符
        :param  _start_str: 句子的开始字符
        :param  _end_str: 句子的结束字符
        :param  _unk_str: 未登录字符
        :param  tensor_int_type: 操作系统不同, windows 选择 tf.int32 , linux 选择 tf.int64

        """

        self.cache_data_dir = os.path.join(base_dir, cache_data_folder)

        self.source_tokenizer_path = os.path.join(self.cache_data_dir, source_tokenizer_file)
        self.target_tokenizer_path = os.path.join(self.cache_data_dir, target_tokenizer_file)

        self.tensor_int_type = tensor_int_type

        # 标准训练数据集
        # self.train_source_corpus_dir = os.path.join(base_dir, 'train.en')
        # self.train_target_corpus_dir = os.path.join(base_dir, 'train.de')

        # 小的训练集用于测试
        self.train_source_corpus_dir = os.path.join(base_dir, 'newstest2012.en')
        self.train_target_corpus_dir = os.path.join(base_dir, 'newstest2012.de')

        self.valid_source_corpus_dir = os.path.join(base_dir, 'newstest2013.en')
        self.valid_target_corpus_dir = os.path.join(base_dir, 'newstest2013.de')

        self.test_source_corpus_dir = os.path.join(base_dir, 'newstest2014.en')
        self.test_target_corpus_dir = os.path.join(base_dir, 'newstest2014.de')


        self._null_str = _null_str
        self._start_str = _start_str
        self._end_str = _end_str
        self._unk_str = _unk_str

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
                sentence = line.rstrip("\n")

                # 句子的左右端点统一补充 1个空格
                sentence = " " + sentence + " "

                text_data.append(sentence)

            return text_data


    def __tf_lower_and_split_punct(self, text):
        """
        对每一个句子的预处理

        :param text:
        :return:
        """

        text = tf_text.normalize_utf8(text)

        # 删除在各个语系外的字符
        text = tf.strings.regex_replace(text, self.remove_unk, ' ')

        # 清除指定单词
        text = tf.strings.regex_replace(text, self.remove_words, '')

        # 清除句子中的标点符号
        text = tf.strings.regex_replace(text, self.remove_punc, ' ')  # 空1格

        # 清除句子中的独立数字
        text = tf.strings.regex_replace(text, self.remove_digits, ' ')  # 空1格

        # 全部转成小写
        text = tf.strings.lower(text)

        # 清除左右端点的空格
        text = tf.strings.strip(text)

        # 左右端点添加 开始 和 结束的控制字符
        text = tf.strings.join([self._start_str, text, self._end_str], separator=' ')

        return text

    def preprocess_corpus(self, text_data, batch_size=64):
        """
        对语料库中的句子进行预处理, 包括删除标点符号

        :param text_data:
        :return:
        """
        text_dataset = tf.data.Dataset.from_tensor_slices(text_data).batch(batch_size)  # 分块后可以加速计算, 后面每次读取都是一块

        text_preprocess_dataset = text_dataset.map(lambda batch_text:
            self.__tf_lower_and_split_punct(batch_text), num_parallel_calls=tf.data.AUTOTUNE)

        return text_preprocess_dataset


    def tokenize_corpus(self, text_data, n_vocab, batch_size=64, max_seq_length=50):
        """
        对语料库中的句子进行标记化, 同时生成词典

        :param text_data: 句子的列表
        :param n_vocab: 词表大小
        :param max_seq_length: 最大的序列的长度

        :return:
        """

        # max_tokens: 保留出现次数最多的 top-k 个单词
        # 若 output_sequence_length=None, 自动将语料库中最长的句子的长度作为 padding 的长度

        text_preprocess_dataset = self.preprocess_corpus(text_data, batch_size)

        tokenizer = TextVectorization(
                    standardize=None,
                    output_sequence_length=max_seq_length,
                    max_tokens=n_vocab)

        tokenizer.adapt(text_preprocess_dataset)

        vocab_list = tokenizer.get_vocabulary()

        print('vocab_list length: {}'.format(len(vocab_list)))

        print('vocab_list: ', vocab_list[:20])

        # 标记化所有的句子, 包括 padding
        text_vector_dataset = text_preprocess_dataset.map(lambda batch_text: tokenizer(batch_text))

        print('tokenize text complete ')

        return  text_vector_dataset, tokenizer, vocab_list


    def tf_data_pipline(self, source_vector, target_vector, reverse_source=True, do_persist=False, dataset_file=None):
        """
        利用 tf.data 数据流水线 建立数据集,

        :param source_vector:
        :param target_vector:
        :param reverse_source: 是否将源句子反转
        :param do_persist:
        :param dataset_file:

        :return:
        """

        assert len(source_vector) == len(target_vector)

        print('dataset batch num: ', len(source_vector))

        # 输入 和 输出的 target 要错开一位
        target_out = target_vector.map(lambda batch_text: batch_text[:, 1:], num_parallel_calls=tf.data.AUTOTUNE)
        target_in = target_vector.map(lambda batch_text: batch_text[:, :-1], num_parallel_calls=tf.data.AUTOTUNE)

        if reverse_source:  # 将源句子反转
            source_vector = source_vector.map(lambda batch_text: batch_text[:, ::-1], num_parallel_calls=tf.data.AUTOTUNE)

        # 特征和标签分开, 之后再合并
        features_dataset = tf.data.Dataset.zip((source_vector, target_in))
        labels_dataset = target_out

        # 特征 和 标签 合并
        dataset = tf.data.Dataset.zip((features_dataset, labels_dataset)).unbatch()  # 清除掉 batch 索引

        if do_persist:
            dataset_path = os.path.join(self.cache_data_dir, dataset_file)
            tf.data.experimental.save(dataset, dataset_path)

        return dataset


    def build_source_target_dict(self, source_text, source_vector, target_text, reverse_source=True, do_persist=False, source_target_dict_file=None):
        """

        1.待翻译的源语句会对应多个人工翻译的标准句子, 因此需要组合源语句和目标语句, 并返回组合后的字典

        :param source_text:
        :param source_vector:
        :param target_text:
        :param reverse_source: 是否将源句子反转
        :param do_persist: 将结果持久化到磁盘
        :param source_target_dict_file:

        :return:
            source_target_dict
            = {
                b'<START> republican leaders justified their... <END>' : {
                        'vector': array([1,.....,0])
                        'target': [b'<START> Die Führungskräfte der Republikaner rechtfertigen... <END>' ]
                     }
              }
        """

        source_text = source_text.unbatch()
        source_vector = source_vector.unbatch()
        target_text = target_text.unbatch()

        source_target_dict = {}

        for source, vector, target in zip(source_text, source_vector, target_text):

                key = source.numpy().decode('utf-8')
                source_target_dict[key] = {}

                source_vec = vector.numpy()

                if reverse_source:
                    source_vec = source_vec[::-1]

                source_target_dict[key]['vector'] = source_vec
                source_target_dict[key]['target'] = [target.numpy().decode('utf-8')]

        if do_persist:

            save_dict = {}
            save_dict['source_target_dict'] = source_target_dict
            source_target_dict_path = os.path.join(self.cache_data_dir, source_target_dict_file)

            with open(source_target_dict_path, 'wb') as f:
                pickle.dump(save_dict, f)

        return source_target_dict


    def do_mian(self, batch_size, n_vocab_source, n_vocab_target, max_seq_length):
        """
        数据集预处理的主流程

        :param batch_size: 一个批次的大小
        :param n_vocab_source: 源语言的词典大小
        :param n_vocab_target: 目标语言的词典大小
        :param max_seq_length: 最大序列的长度

        :return:
        """

        # 1.训练数据处理
        print('preprocess the train dataset...')

        train_source_text = self.load_corpus_data(corpus_file_dir=self.train_source_corpus_dir)
        train_target_text = self.load_corpus_data(corpus_file_dir=self.train_target_corpus_dir)

        train_source_vector, tokenizer_source, vocab_source_list = self.tokenize_corpus(train_source_text, n_vocab_source, batch_size=batch_size, max_seq_length=max_seq_length)
        train_target_vector, tokenizer_target, vocab_target_list = self.tokenize_corpus(train_target_text, n_vocab_target, batch_size=batch_size, max_seq_length=max_seq_length)

        # 建立字典
        print('build the vocab...')
        vocab_source = Vocab(vocab_list=vocab_source_list, vocab_list_path=os.path.join(self.cache_data_dir, 'vocab_source.bin'))
        vocab_target = Vocab(vocab_list=vocab_target_list, vocab_list_path=os.path.join(self.cache_data_dir, 'vocab_target.bin'))

        # 训练数据集
        train_dataset = self.tf_data_pipline(train_source_vector, train_target_vector, reverse_source=False, do_persist=True, dataset_file='train_dataset.bin')
        reverse_train_dataset = self.tf_data_pipline(train_source_vector, train_target_vector, reverse_source=True, do_persist=True, dataset_file='reverse_train_dataset.bin')


        # 2.验证数据处理
        print('preprocess the valid dataset...')

        valid_source_text = self.load_corpus_data(corpus_file_dir=self.valid_source_corpus_dir)
        valid_target_text = self.load_corpus_data(corpus_file_dir=self.valid_target_corpus_dir)

        valid_source = self.preprocess_corpus(valid_source_text, batch_size=batch_size)
        valid_target = self.preprocess_corpus(valid_target_text, batch_size=batch_size)

        valid_source_vector = valid_source.map(lambda batch_text: tokenizer_source(batch_text))
        valid_target_vector = valid_target.map(lambda batch_text: tokenizer_target(batch_text))

        # 验证数据集
        valid_dataset = self.tf_data_pipline(valid_source_vector, valid_target_vector, reverse_source=False, do_persist=True, dataset_file='valid_dataset.bin')
        reverse_valid_dataset = self.tf_data_pipline(valid_source_vector, valid_target_vector, reverse_source=True, do_persist=True, dataset_file='reverse_valid_dataset.bin')


        # 验证数据 source_target_dict
        self.build_source_target_dict(valid_source, valid_source_vector, valid_target, reverse_source=False, do_persist=True, source_target_dict_file='valid_source_target_dict.bin')
        self.build_source_target_dict(valid_source, valid_source_vector, valid_target, reverse_source=True, do_persist=True, source_target_dict_file='reverse_valid_source_target_dict.bin')


        # 3.测试数据处理

        print('preprocess the test dataset...')

        test_source_text = self.load_corpus_data(corpus_file_dir=self.test_source_corpus_dir)
        test_target_text = self.load_corpus_data(corpus_file_dir=self.test_target_corpus_dir)

        test_source = self.preprocess_corpus(test_source_text, batch_size=batch_size)
        test_target = self.preprocess_corpus(test_target_text, batch_size=batch_size)

        test_source_vector = test_source.map(lambda batch_text: tokenizer_source(batch_text))

        # 测试数据 source_target_dict
        self.build_source_target_dict(test_source, test_source_vector, test_target, reverse_source=False, do_persist=True, source_target_dict_file='test_source_target_dict.bin')
        self.build_source_target_dict(test_source, test_source_vector, test_target, reverse_source=True, do_persist=True, source_target_dict_file='reverse_test_source_target_dict.bin')


class Vocab:

    def __init__(self, vocab_list_path, vocab_list=None):
        """
        词典的初始化

        :param vocab_path: 词典持久化的路径
        :param vocab_list:  (建立新的词典时必填)

        """
        self.vocab_list_path = vocab_list_path
        
        if vocab_list is not None:  # 建立新的词典
        
            self.word_to_id, self.id_to_word, self.n_vocab = self.__build_vocab(vocab_list)

        else:  # 读取已有的词典

            self.word_to_id, self.id_to_word, self.n_vocab = self.__load_vocab()

    def __build_vocab(self, vocab_list):
        """
        建立词典

        :param vocab_list: ['', '[UNK]', '[START]', '[END]', '.', 'que', 'de', 'el', 'a', 'no']
        :return:
        """
        n_vocab = len(vocab_list)  # 字典的长度

        word_to_id = StringLookup(
                    vocabulary=vocab_list,
                    mask_token='',
                    )

        id_to_word = StringLookup(
            vocabulary=vocab_list,
            mask_token='',
            invert=True)

        save_dict = {}

        save_dict['vocab_list'] = vocab_list

        with open(self.vocab_list_path, 'wb') as f:

            pickle.dump(save_dict, f)

        return word_to_id, id_to_word, n_vocab


    def __load_vocab(self):
        """
        读取词典

        :param vocab_path:
        :return:
        """

        with open(self.vocab_list_path, 'rb') as f:
            save_dict = pickle.load(f)

        vocab_list = save_dict['vocab_list']

        n_vocab = len(vocab_list)  # 字典的长度

        word_to_id = StringLookup(
                    vocabulary=vocab_list,
                    mask_token='',
                    )

        id_to_word = StringLookup(
            vocabulary=vocab_list,
            mask_token='',
            invert=True)

        return word_to_id, id_to_word, n_vocab

    def map_id_to_word(self, id):
        """
        输入单词标号, 返回单词

        1.若单词标号未在 逆词典中, 返回 '<UNK>'

        :param id:
        :return:
        """
        return self.id_to_word(id)

    def map_word_to_id(self, word):
        """
        输入单词, 返回单词标号

        考虑未登录词:
        1.若输入的单词不在词典中, 返回 '<UNK>' 的标号

        :param word: 单词
        :return:
        """

        return self.word_to_id(word)


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

                 train_dataset_file='train_dataset.bin',
                 valid_dataset_file='valid_dataset.bin',

                 valid_source_target_dict_file='valid_source_target_dict.bin',
                 test_source_target_dict_file='test_source_target_dict.bin',

                 reverse_source=True,
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
        :param reverse_source: 是否将源句子反转
        :param mode: 当前处在的模式, 可以选择
                    'train' - 训练模式
                    'infer' - 推理模式

        """
        self.cache_data_dir = os.path.join(base_dir, cache_data_folder)

        self.vocab_source = Vocab(os.path.join(self.cache_data_dir, vocab_source_file))
        self.vocab_target = Vocab(os.path.join(self.cache_data_dir, vocab_target_file))

        preffix = ''

        if reverse_source:
            preffix = 'reverse_'

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

    def test_DataPreprocess(self):

        process_obj = DataPreprocess(cache_data_folder='cache_data')

        process_obj.do_mian(batch_size=256, n_vocab_source=50000, n_vocab_target=50000, max_seq_length=50)

        # 使用小的训练数据集进行测试
        # process_obj = DataPreprocess(cache_data_folder='cache_small_data')
        #
        # process_obj.do_mian(batch_size=256, n_vocab_source=5000, n_vocab_target=5000, max_seq_length=50)

    def test_WMT14_Eng_Ge_Dataset(self):

        dataset_train_obj = WMT14_Eng_Ge_Dataset(reverse_source=True, cache_data_folder='cache_data', mode='train')

        dataset_infer_obj = WMT14_Eng_Ge_Dataset(reverse_source=True, cache_data_folder='cache_data', mode='infer')

        # 使用小的训练数据集进行测试
        # dataset_train_obj = WMT14_Eng_Ge_Dataset(reverse_source=True, cache_data_folder='cache_small_data', mode='train')
        #
        # dataset_infer_obj = WMT14_Eng_Ge_Dataset(reverse_source=True, cache_data_folder='cache_small_data', mode='infer')



        # 查看 1 个批次的数据
        for batch_feature, batch_label in tqdm(dataset_train_obj.train_dataset.take(1)):

            source = batch_feature[0]

            target_in = batch_feature[1]
            target_out = batch_label

            print('source:')
            print(source)

            print('target_in:')
            print(target_in)

            print('target_out:')
            print(target_out)


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



if __name__ == '__main__':
    test = Test()

    #TODO：运行之前 把 jupyter notebook 停掉, 否则会出现争抢 GPU 导致报错

    test.test_DataPreprocess()

    test.test_WMT14_Eng_Ge_Dataset()
