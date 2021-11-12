#!/usr/bin/python
# -*- coding: UTF-8 -*-

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import sequence

# from deprecated import deprecated
import os

import numpy as np
from tqdm import tqdm
from PIL import Image
import pickle
import pandas as pd
import re
import string
from collections import Counter

from sklearn.decomposition import PCA

from lib.utils_xrh import *



class DataPreprocess:
    """
    数据集预处理

    主流程见 do_main()

    Author: xrh
    Date: 2021-9-25

    """

    def __init__(self,
                 base_dir='../dataset/',
                 _null_str='<NULL>',
                 _start_str='<START>',
                 _end_str='<END>',
                 _unk_str='<UNK>',
                 ):
        """
        :param base_dir:  数据集的路径
        :param  _null_str: 空字符
        :param  _start_str: 句子的开始字符
        :param  _end_str: 句子的结束字符
        :param  _unk_str: 未登录字符

        """

        self.caption_file_dir = os.path.join(base_dir, 'Flicker8k/Flickr8k.token.txt')
        self.image_folder_dir = os.path.join(base_dir, 'Flicker8k/Flicker8k_Dataset/')

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

        # 删除数字字符
        # eg. 'avc 10 abc-10' -> 'avc  abc-10'
        self.remove_digits_re = re.compile(r'\s[0-9]+\s')

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

        # 1. Get the list of all image names
        all_images = list(caption_dict.keys())

        # 2. Shuffle if necessary
        if shuffle:
            np.random.shuffle(all_images)

        # 3. Split into training and validation sets
        train_size = int(len(caption_dict) * train_size)

        train_caption_dict = {
            img_name: caption_dict[img_name] for img_name in all_images[:train_size]
        }
        validation_caption_dict = {
            img_name: caption_dict[img_name] for img_name in all_images[train_size:]
        }

        # 4. Return the splits
        return train_caption_dict, validation_caption_dict

    def zip_image_embed_and_caption(self, caption_dict, image_embed_dict, vocab_obj, max_sentence_length=40,
                                        do_persist=True, dataset_dir=None):
        """
        1.一张图片对应多段描述, 因此需要组合 编码后的图片 和 图片的描述, 作为训练数据集

        2.对图片描述的末尾 做 <NULL> 元素的填充, 直到该句子满足目标长度

        :param caption_dict: 字典, key 为图片的路径, value 为图片描述的文本列表
        :param image_embed_dict: 字典, key 为图片的名字,  value 为 编码后的图片向量
        :param vocab_obj: 词典对象
        :param max_sentence_length:  图片描述句子的目标长度

        :param do_persist: 是否将结果持久化到磁盘
        :param dataset_dir: 数据集存储路径
        :return:

        image_feature_list shape:(m, n_image_feature)
        caption_encoding_list shape:(m, max_sentence_length)

        """

        image_dir_list = []
        image_feature_list = []
        caption_list = []
        caption_encoding_list = []

        for k, v_list in caption_dict.items():

            image_dir = k

            image_name = image_dir[len(self.image_folder_dir):]

            image_feature = image_embed_dict[image_name]

            for caption in v_list:
                caption_encoding = [vocab_obj.map_word_to_id(ele) for ele in caption.split()]

                image_dir_list.append(image_dir)
                image_feature_list.append(image_feature)
                caption_list.append(caption)
                caption_encoding_list.append(caption_encoding)

        #  对不够长的序列进行填充
        caption_encoding_list = list(
            sequence.pad_sequences(caption_encoding_list, maxlen=max_sentence_length, padding='post',
                                   value=vocab_obj.map_word_to_id(self._null_str)))

        dataset = pd.DataFrame({'image_dir': image_dir_list, 'image_feature': image_feature_list,
                                'caption': caption_list, 'caption_encoding': caption_encoding_list})

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

    def build_image_caption_dict(self, caption_dict, image_encoding_dict, do_persist=True, image_caption_dict_dir=None):
        """

        1.一张图片对应多段描述, 因此需要组合 图片路径, 图片向量 和 图片的描述, 返回组合后的字典

        :param caption_dict: 字典, key 为图片的路径, value 为图片描述的文本列表
        :param image_encoding_dict: 字典, key 为图片的名字,  value 为 编码后的图片向量
        :param do_persist: 将结果持久化到磁盘
        :param image_caption_dict_dir: 结果的持久化路径

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
            image_dir = k
            image_name = k[len(self.image_folder_dir):]
            image_feature = image_encoding_dict[image_name]

            image_caption_dict[image_dir] = {'feature': image_feature, 'caption': v_list}

        if do_persist:
            save_dict = {}
            save_dict['image_caption_dict'] = image_caption_dict

            if image_caption_dict_dir is None:
                image_caption_dict_dir = self.image_caption_dict_dir

            with open(image_caption_dict_dir, 'wb') as f:
                pickle.dump(save_dict, f)

        return image_caption_dict

    def load_image_caption_dict(self, image_caption_dict_dir=None):
        """
        读取 image_caption_dict

        :param image_caption_dict_dir: 持久化路径

        :return:

            image_caption_dict
            = {
                '.../.../XXX.jpg' : {
                                'feature': 编码后的图片向量
                                'caption': 图片描述的文本列表
                              }
              }

        """
        if image_caption_dict_dir is None:
            image_caption_dict_dir = self.image_caption_dict_dir

        with open(image_caption_dict_dir, 'rb') as f:
            save_dict = pickle.load(f)

        image_caption_dict = save_dict['image_caption_dict']

        return image_caption_dict

    def do_mian(self, max_caption_length, freq_threshold):
        """
        数据集预处理的主流程

        :return:
        """
        np.random.seed(1)  # 设置随机数种子

        print("max_caption_length:{}, freq_threshold:{}".format(max_caption_length, freq_threshold))

        caption_mapping, text_data = self.load_captions_data(lowercase=True, clean_digits=True)
        train_caption_dict, valid_caption_dict = self.train_val_split(caption_mapping, shuffle=True)

        print('build the vocab...')
        vocab_obj = BuildVocab(load_vocab_dict=False, freq_threshold=freq_threshold, text_data=text_data)

        max_sentence_length = vocab_obj.get_max_sentence_length(text_data)
        print('max_sentence_length: {}'.format(max_sentence_length))

        cache_data_base_dir = '../dataset/cache_data/'

        print('embedding the picture...')

        # image_emb_obj = EmbeddingImage(load_pretrain=True)
        # train_image_path_list = list(train_caption_dict.keys())
        #
        # train_image_emb_dict, train_image_emb_pca_dict = image_emb_obj.process_embed_image(train_image_path_list,
        #                                                          os.path.join(cache_data_base_dir,'encoded_images_train_inceptionV3.p'))
        #
        # valid_image_path_list = list(valid_caption_dict.keys())
        # valid_image_emb_dict, valid_image_emb_pca_dict = image_emb_obj.process_embed_image(valid_image_path_list,
        #                                                          os.path.join(cache_data_base_dir,'encoded_images_valid_inceptionV3.p'))

        # TODO: 考虑前面对数据集进行了 shuffle
        image_emb_obj = EmbeddingImage(load_pretrain=False)
        train_image_emb_dict, train_image_emb_pca_dict = image_emb_obj.load_embed_image_vector(os.path.join(cache_data_base_dir,'encoded_images_train_inceptionV3.p'))
        valid_image_emb_dict, valid_image_emb_pca_dict = image_emb_obj.load_embed_image_vector(os.path.join(cache_data_base_dir,'encoded_images_valid_inceptionV3.p'))

        print('not used PCA: ')

        print('building the train dataset...')
        self.zip_image_embed_and_caption(train_caption_dict, train_image_emb_dict, vocab_obj, max_sentence_length=max_caption_length)

        print('building the valid(test) dataset...')
        self.zip_image_embed_and_caption(valid_caption_dict, valid_image_emb_dict, vocab_obj,
                                            max_sentence_length=max_caption_length, dataset_dir=os.path.join(cache_data_base_dir,'valid_dataset.json'))

        print('building the valid(test) dict...')
        self.build_image_caption_dict(valid_caption_dict, valid_image_emb_dict)

        # print('building the train dict...')
        # self.build_image_caption_dict(train_caption_dict, train_image_emb_dict, image_caption_dict_dir=os.path.join(cache_data_base_dir,'train_image_caption_dict.bin'))


        print('used PCA: ')

        print('building the train dataset...')
        self.zip_image_embed_and_caption(train_caption_dict, train_image_emb_pca_dict, vocab_obj, max_sentence_length=max_caption_length, dataset_dir=os.path.join(cache_data_base_dir,'train_dataset_pca.json'))

        print('building the valid(test) dataset...')
        self.zip_image_embed_and_caption(valid_caption_dict, valid_image_emb_pca_dict, vocab_obj,
                                            max_sentence_length=max_caption_length, dataset_dir=os.path.join(cache_data_base_dir,'valid_dataset_pca.json'))

        print('building the valid(test) dict...')
        self.build_image_caption_dict(valid_caption_dict, valid_image_emb_pca_dict, image_caption_dict_dir=os.path.join(cache_data_base_dir,'image_caption_dict_pca.bin'))



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
                 vocab_path='../dataset/cache_data/vocab.bin', load_vocab_dict=True, freq_threshold=0,
                 text_data=None):
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

            self.stopwords = set(['\'', '-'])  # 不计入词典的词

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
        3.在语料库中出现次数大于 freq_threshold 次的词才计入词典中

        :param text_data: 数据集中的所有句子的列表
        :return:
            word_to_id, id_to_word
        """

        text_data_flat = []

        for sentence in text_data:

            # 删除位置标记单词
            sentence_clean = self.remove_word_re.sub(' ', sentence)

            # 因为是英文, 无需分词, 所有单词之间已经有空格
            sentence_split = sentence_clean.split()

            for word in sentence_split:

                if word not in self.stopwords:
                    text_data_flat.append(word)

        vocab_counter = Counter(text_data_flat)

        print('The least frequent words in the corpus : {}'.format(sorted(vocab_counter)[:80]))  # 查看出现次数最少的单词

        vocab_counter_major = {}  # 主要的单词

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


class EmbeddingImage:
    """
    使用预训练模型 (InceptionV3) 对图片进行 Embedding

    Author: xrh
    Date: 2021-9-25

    """

    def __init__(self, image_folder_dir='../dataset/Flicker8k/Flicker8k_Dataset/', output_layer_id=-2, load_pretrain=True):
        """

        :param image_folder_dir: 图片文件夹路径
        :param output_layer_id: 预训练模型的输出层的标号, -2 为倒数第 2层
        :param load_pretrain: 是否载入预训练模型
        """

        self.image_folder_dir = image_folder_dir
        self.output_layer_id = output_layer_id

        self.model_emb_pict = None

        if load_pretrain:
            self.__load_pretrain_model()

    def __load_pretrain_model(self):
        """
        载入 预训练的 CNN模型

        :return:
        """
        model = InceptionV3(weights='imagenet')
        new_input = model.input
        hidden_layer = model.layers[self.output_layer_id].output

        print('the embedding layer output shape: ', hidden_layer)

        self.model_emb_pict = Model(new_input, hidden_layer)

    def __normalize_image(self, x):
        """

        对图像进行标准化，使其包含范围为-1到1的像素，这与用于训练 InceptionV3 的图像格式相匹配。

        :param x:
        :return:
        """
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    def __vectorize_image(self, image_path):
        """
        图片向量化

        :param image_path: 图片的路径
        :return:
        """

        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)

        x = np.expand_dims(x, axis=0)

        x = self.__normalize_image(x)  #  TODO: 标准化太重要了, 被坑死

        return x

    def encode_image(self, image_path):
        """
        使用预训练模型对图片进行编码

        :param image_path: 图片的路径
        :return:
        """
        image_vec = self.__vectorize_image(image_path)

        # model_emb_pict: 图片的预训练模型
        image_emb = self.model_emb_pict.predict(image_vec)
        image_emb = np.reshape(image_emb, image_emb.shape[1])

        return image_emb

    def process_embed_image(self, image_path_list, embed_images_path, n_components=512):
        """
        图片编码流程

        :param image_path_list: 图片路径的列表
        :param embed_images_path: 输出编码后的图片向量的路径
        :param n_components: PCA 降维后的维度

        :return:

        image_emb_dict: 字典, key 为图片的名字, value 为图片的 embedding
        image_emb_pca_dict: 字典, key 为图片的名字, value 为图片经过 PCA 降低维度的向量

        """

        image_emb_list = []

        for image_path in tqdm(image_path_list):

            # image_path = image_path_list[i]
            image_emb_list.append((self.encode_image(image_path)).tolist()) # 一张一张图片分别做 embedding 是为了避免爆内存

        image_emb_arr = np.array(image_emb_list)  # 数据集中的图片经过CNN映射后的向量

        # 使用 PCA 降维
        pca = PCA(n_components=n_components)
        image_emb_pca_arr = pca.fit(image_emb_arr).transform(image_emb_arr)

        image_emb_dict = {}
        image_emb_pca_dict = {}

        for i in range(len(image_path_list)):
            image_path = image_path_list[i]

            image_emb_dict[image_path[len(self.image_folder_dir):]] = image_emb_arr[i]
            image_emb_pca_dict[image_path[len(self.image_folder_dir):]] = image_emb_pca_arr[i]


        save_dict = {}
        save_dict['image_emb'] = image_emb_dict
        save_dict['image_emb_pca'] = image_emb_pca_dict

        with open(embed_images_path, "wb") as f:
            pickle.dump(save_dict, f)

        return image_emb_dict, image_emb_pca_dict

    def load_embed_image_vector(self, embed_images_path):
        """
        读取经过 embedding 的图片向量

        :param embed_images_path: 编码后的图片向量的路径
        :return:

       encoding_dict: 字典, key 为图片的名字, value 为图片的 embedding

        """
        with open(embed_images_path, 'rb') as f:
            save_dict = pickle.load(f)

        image_emb_dict = save_dict['image_emb']
        image_emb_pca_dict = save_dict['image_emb_pca']

        return image_emb_dict, image_emb_pca_dict


class Test:

    def test_DataPreprocess(self):
        process_obj = DataPreprocess()

        process_obj.do_mian(max_caption_length=37, freq_threshold=0)


if __name__ == '__main__':
    test = Test()

    #TODO：运行之前 把 jupyter notebook 停掉, 否则会出现争抢 GPU 导致报错
    test.test_DataPreprocess()

