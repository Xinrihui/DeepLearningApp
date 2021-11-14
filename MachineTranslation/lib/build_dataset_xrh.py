#!/usr/bin/python
# -*- coding: UTF-8 -*-

from deprecated import deprecated
import os

import numpy as np
from tqdm import tqdm
from PIL import Image
import pickle
import pandas as pd
import re
import string
from collections import Counter
import jieba

from tensorflow.keras.preprocessing import sequence


from lib.utils_xrh import *



class DataPreprocess:
    """
    数据集预处理

    主流程见 do_main()

    Author: xrh
    Date: 2021-9-25

    """

    def __init__(self,
                 base_dir='../dataset/anki',
                 _null_str='<NULL>',
                 _start_str='<START>',
                 _end_str='<END>',
                 _unk_str='<UNK>',
                 split_source=False,
                 split_target=False,
                 ):
        """
        :param base_dir:  数据集的根路径
        :param  _null_str: 空字符
        :param  _start_str: 句子的开始字符
        :param  _end_str: 句子的结束字符
        :param  _unk_str: 未登录字符
        :param  split_source: 对源句子进行分词
        :param  split_target: 对目标句子进行分词

        """

        self.base_dir = base_dir

        # self.corups_file_dir = os.path.join(base_dir, 'cmn-eng/cmn.txt')  # 英文-中文

        self.corups_file_dir = os.path.join(base_dir, 'fra-eng/fra.txt')  # 英文-法文

        self.dataset_dir = os.path.join(base_dir, 'cache_data/train_dataset.json')

        self.source_target_dict_dir = os.path.join(base_dir, 'cache_data/valid_source_target_dict.bin')

        self._null_str = _null_str
        self._start_str = _start_str
        self._end_str = _end_str
        self._unk_str = _unk_str

        self.split_source = split_source
        self.split_target = split_target

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

        # 需要删除的标点符号
        # remove_chars = string.punctuation  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

        # 需要删除的标点符号, 包含中文符号
        remove_chars = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·"""

        remove_chars = remove_chars.replace("'", "")  # 英语中有', 不删除 '
        remove_chars = remove_chars.replace("-", "")  #

        self.remove_chars_re = re.compile('[%s]' % re.escape(remove_chars))

        # 删除数字字符
        # eg. 'avc 10 abc-10' -> 'avc  abc-10'
        self.remove_digits_re = re.compile(r'\s[0-9]+\s')




    def load_corups_data(self, topN=None ,clean_punctuation=True, clean_digits=True, lowercase=True):
        """
        读取语料数据

        1. 可以选择是否清除其中的标点符号

        :param topN: 取数据集中前 N 条记录
        :param clean_punctuation:  是否清除文本中的标点符号
        :param clean_digits: 是否清除文本中的数字
        :param clean_digits: 是否全部小写
        :return:

        corups_mapping, source_text_data, target_text_data

        """

        with open(self.corups_file_dir, encoding='utf-8') as corups_file:

            corups_data = corups_file.readlines()
            corups_mapping = {}

            source_text_data = []
            target_text_data = []

            if topN is not None:
                corups_data = corups_data[:topN]

            for line in corups_data:

                line = line.rstrip("\n")
                # English + TAB + The Other Language + TAB + Attribution
                source, target, attribution = line.split("\t")

                source_id = attribution.split("#")[1].split(" ")[0]

                # 清除句子前后的空格
                source = source.strip()
                target = target.strip()

                # 删除非法的字符
                source = self.remove_unk_re.sub(' ', source)
                target = self.remove_unk_re.sub(' ', target)

                if clean_punctuation:
                    # 清除句子中的标点符号
                    source = self.remove_chars_re.sub(' ', source)
                    target = self.remove_chars_re.sub(' ', target)

                if clean_digits:
                    # 清除句子中的数字
                    source = self.remove_digits_re.sub(' ', source)
                    target = self.remove_digits_re.sub(' ', target)

                if lowercase:
                    # 全部转成小写
                    source = source.lower()
                    target = target.lower()

                if self.split_source:
                    source = " ".join(jieba.cut(source))

                if self.split_target:
                    target = " ".join(jieba.cut(target))

                # 对句子加上 开始和结束标识词
                source = self._start_str + " " + source + " " + self._end_str
                target = self._start_str + " " + target + " " + self._end_str

                source_text_data.append(source)
                target_text_data.append(target)

                if source_id in corups_mapping:
                    corups_mapping[source_id][1].append(target)
                else:
                    corups_mapping[source_id] = [source, [target]]

            return corups_mapping, source_text_data, target_text_data

    def train_val_split(self, corups_mapping, train_size=0.8, shuffle=True):
        """
        将数据集划分为 训练数据集 和 验证数据集(测试数据)

        :param corups_mapping: 字典, key: source_id, value: [source, target_list]
        :param train_size: 训练数据的比例
        :param shuffle: 是否混洗
        :return: train_corups_dict, valid_corups_dict

        """

        source_id_list = list(corups_mapping.keys())

        # 是否洗牌
        if shuffle:
            np.random.shuffle(source_id_list)

        # 训练数据的规模
        train_size = int(len(source_id_list) * train_size)

        train_corups_dict = {
            source_id: corups_mapping[source_id] for source_id in source_id_list[:train_size]
        }
        valid_corups_dict = {
            source_id: corups_mapping[source_id] for source_id in source_id_list[train_size:]
        }

        return train_corups_dict, valid_corups_dict

    def zip_source_and_target(self, corups_dict,
                                    vocab_source, vocab_target,
                                    max_source_length, max_target_length,
                                    do_persist=True, dataset_dir=None):
        """
        1.一个 源句子 对应多个 目标句子, 将它们组合作为数据集

        2.对图片描述的末尾 做 <NULL> 元素的填充, 直到该句子满足目标长度

        :param corups_dict: 字典,  key: source_id, value: [source, target_list]
        :param vocab_source: 源语言的词典对象
        :param vocab_target: 目标语言的词典对象
        :param max_source_length:  源语言句子的目标长度
        :param max_target_length:  源语言句子的目标长度
        :param do_persist: 是否将结果持久化到磁盘
        :param dataset_dir: 数据集存储路径
        :return: dataset

        """

        source_encoding_list = []
        target_encoding_list = []

        for key, value in corups_dict.items():

            source_id = key
            source = value[0]
            target_list = value[1]

            source_encoding = [vocab_source.map_word_to_id(token) for token in source.split()]

            for target in target_list:

                target_encoding = [vocab_target.map_word_to_id(token) for token in target.split()]

                source_encoding_list.append(source_encoding)
                target_encoding_list.append(target_encoding)


        #  对不够长的序列进行填充
        source_encoding_list = list(
            sequence.pad_sequences(source_encoding_list, maxlen=max_source_length, padding='post',
                                   value=vocab_source.map_word_to_id(self._null_str)))

        target_encoding_list = list(
            sequence.pad_sequences(target_encoding_list, maxlen=max_target_length, padding='post',
                                   value=vocab_target.map_word_to_id(self._null_str)))

        dataset = pd.DataFrame({'source_encoding': source_encoding_list, 'target_encoding': target_encoding_list })

        if do_persist:  # 使用 json 格式持久化到磁盘

            if dataset_dir is None:
                dataset_dir = self.dataset_dir

            dataset.to_json(dataset_dir)

        return dataset

    def load_dataset(self):
        """
        读取 训练数据集
        :return:
        """

        dataset = pd.read_json(self.dataset_dir)

        return dataset

    def build_source_target_dict(self, corups_dict, vocab_source, max_source_length, do_persist=True, source_target_dict_dir=None):
        """

        1.一个源句子对应多个 目标句子, 因此需要组合 源句子, 编码后的源句子, 目标句子 返回组合后的字典

        :param corups_dict: 字典,  key: source_id, value: [source, target_list]
        :param vocab_source: 源语言的词典对象
        :param max_source_length:  源语言句子的目标长度
        :param do_persist: 将结果持久化到磁盘
        :param source_target_dict_dir: 结果的持久化路径

        :return:
            source_target_dict
            = {
                source_id : [source, source_encoding, target_list]

              }
        """

        source_target_dict = {}

        for key, value in corups_dict.items():

            source_id = key
            source = value[0]
            target_list = value[1]

            source_encoding = [vocab_source.map_word_to_id(token) for token in source.split()]

            source_encoding = list(sequence.pad_sequences([source_encoding], maxlen=max_source_length, padding='post',
                                   value=vocab_source.map_word_to_id(self._null_str)))[0]


            source_target_dict[source_id] = [source, source_encoding, target_list]

        if do_persist:
            save_dict = {}
            save_dict['source_target_dict'] = source_target_dict

            if source_target_dict_dir is None:
                source_target_dict_dir = self.source_target_dict_dir

            with open(source_target_dict_dir, 'wb') as f:
                pickle.dump(save_dict, f)

        return source_target_dict

    def load_source_target_dict(self, source_target_dict_dir=None):
        """
        读取 source_target_dict

        :param source_target_dict_dir: 持久化路径

        :return:

            source_target_dict
            = {
                source_id : [source, source_encoding, target_list]
              }

        """
        if source_target_dict_dir is None:
            source_target_dict_dir = self.source_target_dict_dir

        with open(source_target_dict_dir, 'rb') as f:
            save_dict = pickle.load(f)

        source_target_dict = save_dict['source_target_dict']

        return source_target_dict

    def do_mian(self, freq_threshold):
        """
        数据集预处理的主流程

        :return:
        """
        np.random.seed(1)  # 设置随机数种子

        cache_data_base_dir = os.path.join(self.base_dir, 'cache_data')

        print("freq_threshold:{}".format(freq_threshold))

        corups_mapping, source_text_data, target_text_data = self.load_corups_data(topN=50000)

        print('sample num in whole dataset: ', len(source_text_data))

        print('source_id num: ', len(corups_mapping.keys()))  # 一个源句子会对应多个目标句子

        train_corups_dict, valid_corups_dict = self.train_val_split(corups_mapping, train_size=0.9, shuffle=True)

        print('build the source vocab...')
        vocab_source = BuildVocab(vocab_path=os.path.join(cache_data_base_dir, 'source_vocab.bin'), load_vocab_dict=False, freq_threshold=freq_threshold, text_data=source_text_data)

        max_source_length = vocab_source.get_max_sentence_length(source_text_data)
        print('max_source_length: {}'.format(max_source_length))

        print('build the target vocab...')
        vocab_target = BuildVocab(vocab_path=os.path.join(cache_data_base_dir, 'target_vocab.bin'), load_vocab_dict=False, freq_threshold=freq_threshold, text_data=target_text_data)

        max_target_length = vocab_target.get_max_sentence_length(target_text_data)
        print('max_target_length: {}'.format(max_target_length))


        print('building the train dataset...')
        self.zip_source_and_target(corups_dict=train_corups_dict, vocab_source=vocab_source, vocab_target=vocab_target, max_source_length=max_source_length, max_target_length=max_target_length)

        print('building the valid(test) dataset...')
        self.zip_source_and_target(corups_dict=valid_corups_dict, vocab_source=vocab_source, vocab_target=vocab_target, max_source_length=max_source_length, max_target_length=max_target_length
                                   , dataset_dir=os.path.join(cache_data_base_dir, 'valid_dataset.json')
                                   )

        print('building the valid(test) dict...')
        self.build_source_target_dict(corups_dict=valid_corups_dict, vocab_source=vocab_source, max_source_length=max_source_length)

        print('building the train dict...')
        self.build_source_target_dict(corups_dict=train_corups_dict, vocab_source=vocab_source, max_source_length=max_source_length, source_target_dict_dir=os.path.join(cache_data_base_dir, 'train_source_target_dict.bin'))





class BuildVocab:
    """
    根据数据集建立词典

    1. 控制词的标号
       '<NULL>' 的标号为 0,
       '<START>' 的标号为 1,
       '<END>' 的标号为 2,
       '<UNK>' 的标号为 3, '<UNK>' 必须与 填充的'<NULL>'做区分

    2.标点符号不记录中字典
    3.在语料库中出现次数大于 freq_threshold 次的词才计入词典中

    Author: xrh
    Date: 2021-9-25

    """

    def __init__(self, _null_str='<NULL>',
                       _start_str='<START>',
                       _end_str='<END>',
                       _unk_str='<UNK>',
                        vocab_path=None, load_vocab_dict=True, freq_threshold=0, text_data=None):
        """
        :param  _null_str: 空字符
        :param  _start_str: 句子的开始字符
        :param  _end_str: 句子的结束字符
        :param  _unk_str: 未登录字符

        :param vocab_path: 词典路径
        :param load_vocab_dict: 是否读取现有的词典
        :param freq_threshold : 单词出现次数的下限, 若单词出现的次数小于此值, 则不计入字典中
        :param text_data: 数据集中的所有句子的列表

        """
        self._null_str = _null_str
        self._start_str = _start_str
        self._end_str = _end_str
        self._unk_str = _unk_str

        self.vocab_path = vocab_path

        self.freq_threshold = freq_threshold

        if load_vocab_dict:  # 读取现有的词典

            self.word_to_id, self.id_to_word = self.__load_vocab()

        else:  # 生成新的词典

            # 需要删除的标点符号
            remove_chars = string.punctuation  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
            remove_chars = remove_chars.replace("<", "")  # 不能删除 '<' , 因为'<START>'中也有'<'
            remove_chars = remove_chars.replace(">", "")
            # remove_chars = remove_chars.replace(".", "")  # 不删除 句号. 和 逗号，
            # remove_chars = remove_chars.replace(",", "")

            self.remove_chars_re = re.compile('[%s]' % re.escape(remove_chars))

            # 需要删除的控制词
            self.remove_word_re = re.compile(
                r'{}|{}|{}'.format(self._null_str, self._start_str, self._end_str, self._unk_str))

            self.word_to_id, self.id_to_word = self.__build_vocab(text_data)

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

    def get_max_sentence_length(self, text_data):
        """
        数据集中最长序列的长度

        :param text_data: 数据集中的所有句子的列表
        :return:
        """

        max_caption_length = 0

        for caption in text_data:

            capation_length = len(caption.split())

            if capation_length > max_caption_length:
                max_caption_length = capation_length

        return max_caption_length

    def __build_vocab(self, text_data):
        """
        制作词典

        1.配置  '<NULL>' 的标号为 0, '<START>' 的标号为 1, '<END>' 的标号为 2
        2.标点符号不记录字典
        3.在语料库中出现次数大于 5次的词才计入词典中

        :param text_data: 数据集中的所有句子的列表
        :return:
            word_to_id, id_to_word
        """

        text_data_flat = []

        for sentence in text_data:

            # 删除句子中的标点符号
            sentence_clean = self.remove_chars_re.sub(' ', sentence)

            # 删除位置标记单词
            sentence_clean = self.remove_word_re.sub(' ', sentence_clean)

            # 因为是英文, 无需分词, 所有单词之间已经有空格
            sentence_split = sentence_clean.split()

            for word in sentence_split:
                text_data_flat.append(word)

        vocab_counter = Counter(text_data_flat)

        vocab_counter_major = {}

        for k, v in vocab_counter.items():

            if v >= self.freq_threshold:
                vocab_counter_major[k] = v

        print('origin vocab length:{}, the number of words that appear more than {} times in datasets: {}'.format(len(vocab_counter), self.freq_threshold, len(vocab_counter_major)))

        vocab_major_list = [self._null_str, self._start_str, self._end_str, self._unk_str] + list(vocab_counter_major.keys())  # 补充标记单词,
        # 将 <NULL> 放在第1个, 使得 <NULL> 的标号为0
        # 同理, <START> 的标号为1, <END> 的标号为2, <UNK> 的标号为3

        word_to_id = {word: idx for idx, word in enumerate(vocab_major_list)}

        id_to_word = {idx: word for idx, word in enumerate(vocab_major_list)}

        save_dict = {}

        save_dict['word_to_id'] = word_to_id
        save_dict['id_to_word'] = id_to_word

        with open(self.vocab_path, 'wb') as f:

            pickle.dump(save_dict, f)

        return word_to_id, id_to_word

    def __load_vocab(self):
        """
        读取词典

        :param vocab_path:
        :return:
        """

        with open(self.vocab_path, 'rb') as f:
            save_dict = pickle.load(f)

        word_to_id = save_dict['word_to_id']
        id_to_word = save_dict['id_to_word']

        return word_to_id, id_to_word




class Test:

    def test_DataPreprocess(self):

        process_obj = DataPreprocess()

        process_obj.do_mian(freq_threshold=0)

        # source_target_dict = process_obj.load_source_target_dict()

        print()


if __name__ == '__main__':
    test = Test()

    test.test_DataPreprocess()

