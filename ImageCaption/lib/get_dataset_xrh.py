#!/usr/bin/python
# -*- coding: UTF-8 -*-

from lib.build_dataset_xrh import *

from tensorflow.keras.utils import Sequence

from tensorflow.keras.utils import to_categorical

class BatchDataGenSequence(Sequence):
    """
    使用 Sequence 的数据批量生成器

    与 class BatchDataGenerator 不同的是:

    1.Sequence 是进行多进程处理的更安全的方法,
    2.保证网络在每个时期每个样本只训练一次

    Author: xrh
    Date: 2021-10-05

    """

    def __init__(self, n_h, n_embedding, n_vocab, batch_size, dataset, one_hot=False):
        """

        :param n_h: lstm 的隐状态的维度
        :param n_embedding: 词向量的维度
        :param n_vocab: 词表大小
        :param batch_size: 一批数据的样本数
        :param dataset: 预处理后的数据集
        :param one_hot: 标签是否 one-hot 化
        """

        self.n_h = n_h
        self.n_embedding = n_embedding
        self.n_vocab = n_vocab
        self.batch_size = batch_size

        self.one_hot = one_hot

        self.image_feature = np.array(dataset['image_feature'].tolist())
        self.caption_encoding = np.array(dataset['caption_encoding'].tolist())

        # N 数据集的样本总数
        self.N = np.shape(self.caption_encoding)[0]

        order = list(range(self.N))

        np.random.shuffle(order)  # 洗牌算法打乱顺序

        self.image_feature = self.image_feature[order]
        self.caption_encoding = self.caption_encoding[order]

        self.n_image_feature = self.image_feature.shape[1]
        self.max_caption_length = self.caption_encoding.shape[1]

    def __len__(self):

        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        返回一个 batch 的数据

        :param idx:  当前批次的标号
        :return:
        """

        start = idx * self.batch_size  # 当期 batch 开始的样本标号

        N_batch = min(self.batch_size, (self.N - start))  # 当期 batch 的样本个数, 考虑这一批可能是最后一批

        batch_image_feature = self.image_feature[start: start+N_batch]
        batch_caption_encoding = self.caption_encoding[start: start+N_batch]

        caption_out = batch_caption_encoding[:, 1:]  # shape(N,max_caption_length-1)
        caption_in = batch_caption_encoding[:, :-1]  # shape(N,max_caption_length-1)

        zero_init = np.zeros((N_batch, self.n_h))

        if self.one_hot:

            outputs = to_categorical(caption_out, num_classes=self.n_vocab)  # TODO: One-hot 向量导致 OOM

        else:
            outputs = caption_out

        # outputs = ArrayUtils.one_hot_array(caption_out, self.n_vocab)

        return ((caption_in, batch_image_feature, zero_init), outputs)

    def on_epoch_end(self):
        """
        上一个 epoch 的所有 batch 数据都过了一遍之后, 开启一个新的 epoch

        :return:
        """

        order = list(range(self.N))
        np.random.shuffle(order)  # 洗牌算法打乱顺序

        self.image_feature = self.image_feature[order]
        self.caption_encoding = self.caption_encoding[order]



class BatchDataGenerator:
    """
    数据批量生成器

    当数据量过大时, 受限于内存空间, 不能每次都将全部数据喂给模型, 而是分批输入

    Author: xrh
    Date: 2021-9-25

    """

    def __init__(self, Type='train', dataset_dir='../dataset/Flicker8k/cache_data/{}_dataset.json'):

        self.dataset_dir = dataset_dir.format(Type)

    def read_all(self, n_h, n_vocab, batch_size=32, dataset=None):
        """
        从磁盘中读取整个数据集(json)到内存, 每次随机采样一批数据, 喂入模型进行训练

        :param n_h:
        :param n_vocab:
        :param batch_size:
        :param dataset:
        :return:
        """
        # 只执行一次

        if dataset is None:
            dataset = pd.read_json(self.dataset_dir)  # 读取数据集

        image_feature = np.array(dataset['image_feature'].tolist())
        caption_encoding = np.array(dataset['caption_encoding'].tolist())

        # N 数据集的样本总数
        N = np.shape(caption_encoding)[0]

        while True:  # 每次调用 next() 执行下面的语句

            mask = np.random.choice(N, batch_size)  # 从 range(m) 中随机有放回采样 batch_size 组成list, N - 样本总数
            # TODO: 同一批次中会出现重复的样本, 当 batch_size << N 问题不大

            batch_image_feature = image_feature[mask]
            batch_caption_encoding = caption_encoding[mask]

            m_batch = np.shape(batch_caption_encoding)[0]  # 一个批次的样本的数量

            c0 = np.zeros((m_batch, n_h))

            # 语言模型的输入 和 输出要错开一个时刻,
            # eg.
            #  output: 今天   /是   /个/好日子/<end>
            #   input: <start>/今天/是/个    /好日子/

            caption_out = batch_caption_encoding[:, 1:]  # shape(N,39)
            caption_in = batch_caption_encoding[:, :-1]  # shape(N,39)

            outputs = ArrayUtils.one_hot_array(caption_out, n_vocab)

            yield ((caption_in, batch_image_feature, c0),
                   outputs)  # 必须是 tuple 否则 ValueError: No gradients provided for any variable (Keras 2.4, Tensorflow 2.3.0)

    @deprecated()
    def read_by_chunk(self, image_feature_dir,caption_encoding_dir,n_a, n_vocab, m, batch_size=32):
        """
        读取预处理后的数据集(csv)时, 使用分批次的方式读入内存

        :param n_a:
        :param n_vocab:
        :param m: 数据集的样本总数
        :param batch_size:
        :return:
        """

        # 只执行一次
        image_feature = pd.read_csv(image_feature_dir, header=None, iterator=True)  # csv 是如此之大, 无法一次读入内存
        caption_encoding = pd.read_csv(caption_encoding_dir, header=None, iterator=True)

        steps_per_epoch = m // batch_size  # 每一个 epoch 要生成的多少批数据
        # N - 样本总数
        count = 0

        while True:  # 每次调用 next() 执行下面的语句

            batch_image_feature = image_feature.get_chunk(batch_size).iloc[:, 1:]  # 排除第一列(索引列)
            batch_caption_encoding = caption_encoding.get_chunk(batch_size).iloc[:, 1:]

            batch_image_feature = batch_image_feature.to_numpy()
            batch_caption_encoding = batch_caption_encoding.to_numpy()

            N_batch = np.shape(batch_caption_encoding)[0]  # 一个批次的样本的数量

            c0 = np.zeros((N_batch, n_a))

            # 语言模型的输入 和 输出要错开一个时刻,
            # eg.
            #  output: 今天   /是   /个/好日子/<end>
            #   input: <start>/今天/是/个    /好日子/

            caption_out = batch_caption_encoding[:, 1:]  # shape(N,39)
            caption_in = batch_caption_encoding[:, :-1]  # shape(N,39)

            outputs = ArrayUtils.one_hot_array(caption_out, n_vocab)

            yield ((caption_in, batch_image_feature, c0),
                   outputs)  # 必须是 tuple 否则 ValueError: No gradients provided for any variable (Keras 2.4, Tensorflow 2.3.0)

            count += 1
            if count > steps_per_epoch:  # 所有批次已经走了一遍

                image_feature = pd.read_csv(image_feature_dir, header=None, iterator=True)
                caption_encoding = pd.read_csv(caption_encoding_dir, header=None, iterator=True)

                count = 0



class FlickerDataset:
    """
    包装了 Flicker 数据集,  我们通过此类来访问该数据集

    1.使用之前先对数据集进行预处理, 详见 build_dataset_xrh.py, 预处理后的数据集在 dataset/cache_data 目录下

    Author: xrh
    Date: 2021-9-30

    """

    def __init__(self, base_dir='../dataset/Flicker8k', mode='train', use_PCA=False):
        """

        :param base_dir: 数据集的根路径
        :param mode: 当前处在的模式, 可以选择
                    'train' - 训练模式
                    'infer' - 推理模式

        :param use_PCA: 是否使用对图片向量进行降维后的数据集
        """

        vocab_path = os.path.join(base_dir, 'cache_data/vocab.bin')

        self.vocab_obj = BuildVocab(load_vocab_dict=True, vocab_path=vocab_path)

        suffix=''
        if use_PCA:
            suffix = '_pca'

        self.dataset = {}
        self.image_caption_dict = {}

        if mode == 'train':

            train_dataset_dir = os.path.join(base_dir, 'cache_data/train_dataset{}.json'.format(suffix))
            self.dataset['train'] = pd.read_json(train_dataset_dir)

            valid_dataset_dir = os.path.join(base_dir, 'cache_data/valid_dataset{}.json'.format(suffix))
            self.dataset['valid'] = pd.read_json(valid_dataset_dir)

            # self.image_feature = np.array(self.dataset['image_feature'].tolist())
            # self.caption_encoding = np.array(self.dataset['caption_encoding'].tolist())

            self.N_train = np.array(self.dataset['train']['caption_encoding'].tolist()).shape[0]  #  N_train - 训练集样本总数
            self.N_valid = np.array(self.dataset['valid']['caption_encoding'].tolist()).shape[0] #  N_valid - 验证集样本总数

            self.feature_dim = np.array(self.dataset['valid']['image_feature'].tolist()).shape[1]  # feature_dim - 图片向量的维度
            self.caption_length = np.array(self.dataset['valid']['caption_encoding'].tolist()).shape[1]  # caption_length - 图片描述的长度

        elif mode == 'infer':

            data_process = DataPreprocess(
                base_dir=base_dir,
            )

            self.image_caption_dict = data_process.load_image_caption_dict(
                image_caption_dict_dir=os.path.join(base_dir, 'cache_data/image_caption_dict{}.bin'.format(suffix)))



    def sample_minibatch(self, Type='train', batch_size=128, max_length=30):
        """
        从数据集中采样 1个 batch 的样本用于训练

        :param batch_size:  1个 batch的样本个数

        :return:
        """

        dataset = self.dataset[Type]

        image_feature = np.array(dataset['image_feature'].tolist())
        caption_encoding = np.array(dataset['caption_encoding'].tolist())

        N = caption_encoding.shape[0]

        mask = np.random.choice(N, batch_size)  # 从 range(m) 中随机采样 batch_size 组成list

        batch_image_feature = image_feature[mask]
        batch_caption_encoding = caption_encoding[mask]

        batch_caption_encoding = batch_caption_encoding[:, :max_length]

        return batch_caption_encoding, batch_image_feature





class Test:

    def test_BatchDataGenerator(self):

        batch_data_generator = BatchDataGenerator(Type='valid')

        n_h = 300
        n_vocab = 9199
        batch_size = 2

        generator = batch_data_generator.read_all(n_h, n_vocab, batch_size=batch_size)

        print(next(generator))



    def test_FlickerDataset(self):

        dataset_obj = FlickerDataset(use_PCA=True)

        data = dataset_obj.sample_minibatch(Type='valid', batch_size=2)
        # print(data)

        batch_data_generator = BatchDataGenerator()

        n_h = 512
        n_vocab = 8441+4
        batch_size = 2

        train_generator = batch_data_generator.read_all(n_h, n_vocab, batch_size=batch_size, dataset=dataset_obj.dataset['train'])

        batch = next(train_generator)

        print(batch)

if __name__ == '__main__':


    test = Test()

    # test.test_FlickerDataset()

    test.test_BatchDataGenerator()


