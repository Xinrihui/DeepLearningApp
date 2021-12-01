#!/usr/bin/python
# -*- coding: UTF-8 -*-

from lib.build_dataset_xrh import *

from tensorflow.keras.utils import Sequence


from tensorflow.keras.utils import to_categorical


class BatchDataGenSequence(Sequence):
    """
    使用 Sequence 的数据批量生成器

    1.Sequence 是进行多进程处理的更安全的方法,
    2.保证网络在每个时期每个样本只训练一次

    Author: xrh
    Date: 2021-10-10

    """

    def __init__(self, n_h, n_embedding, n_vocab_target, batch_size, dataset, one_hot=False):
        """

        :param n_h: lstm 的隐状态的维度
        :param n_embedding: 词向量的维度
        :param n_vocab_target: 目标语言的词表大小
        :param batch_size: 一批数据的样本数
        :param dataset: 预处理后的数据集
        :param one_hot: 标签是否 one-hot 化
        """

        self.n_h = n_h
        self.n_embedding = n_embedding
        self.n_vocab_target = n_vocab_target
        self.batch_size = batch_size
        self.one_hot = one_hot

        self.source_encoding = np.array(dataset['source_encoding'].tolist())
        self.target_encoding = np.array(dataset['target_encoding'].tolist())

        # N 数据集的样本总数
        self.N = np.shape(self.source_encoding)[0]

        order = list(range(self.N))
        np.random.shuffle(order)  # 洗牌算法打乱顺序

        self.source_encoding = self.source_encoding[order]
        self.target_encoding = self.target_encoding[order]

        self.max_source_length = self.source_encoding.shape[1]
        self.max_target_length = self.target_encoding.shape[1]

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

        batch_source_encoding = self.source_encoding[start: start+N_batch]
        batch_target_encoding = self.target_encoding[start: start+N_batch]

        target_out = batch_target_encoding[:, 1:]  # shape(N,max_target_length-1)
        target_in = batch_target_encoding[:, :-1]  # shape(N,max_target_length-1)

        if self.one_hot:

            outputs = to_categorical(target_out, num_classes=self.n_vocab_target)  # TODO: One-hot 向量导致 OOM

        else:
            outputs = target_out

        return ((batch_source_encoding, target_in), outputs)

    def on_epoch_end(self):
        """
        上一个 epoch 的所有 batch 数据都过了一遍之后, 开启一个新的 epoch

        :return:
        """

        order = list(range(self.N))
        np.random.shuffle(order)  # 洗牌算法打乱顺序

        self.source_encoding = self.source_encoding[order]
        self.target_encoding = self.target_encoding[order]


class AnkiDataset:
    """
    包装了 Anki 数据集,  我们通过此类来访问该数据集

    1.使用之前先对数据集进行预处理, 详见 build_dataset_xrh.py, 预处理后的数据集在 dataset/cache_data 目录下

    Author: xrh
    Date: 2021-10-10

    """

    def __init__(self, base_dir='../dataset/anki', mode='train'):
        """

        :param base_dir: 数据集的根路径
        :param mode: 当前处在的模式, 可以选择
                    'train' - 训练模式
                    'infer' - 推理模式

        """

        self.vocab_source = BuildVocab(load_vocab_dict=True, vocab_path=os.path.join(base_dir, 'cache_data/source_vocab.bin'))
        self.vocab_target = BuildVocab(load_vocab_dict=True, vocab_path=os.path.join(base_dir, 'cache_data/target_vocab.bin'))

        self.dataset = {}
        self.valid_source_target_dict = {}
        self.train_source_target_dict = {}

        if mode == 'train':

            train_dataset_dir = os.path.join(base_dir, 'cache_data/train_dataset.json')
            self.dataset['train'] = pd.read_json(train_dataset_dir)

            valid_dataset_dir = os.path.join(base_dir, 'cache_data/valid_dataset.json')
            self.dataset['valid'] = pd.read_json(valid_dataset_dir)

            self.N_train = np.array(self.dataset['train']['source_encoding'].tolist()).shape[0]  #  N_train - 训练集样本总数
            self.N_valid = np.array(self.dataset['valid']['target_encoding'].tolist()).shape[0] #  N_valid - 验证集样本总数

            self.max_source_length = np.array(self.dataset['valid']['source_encoding'].tolist()).shape[1]  # feature_dim - 图片向量的维度
            self.max_target_length = np.array(self.dataset['valid']['target_encoding'].tolist()).shape[1]  # caption_length - 图片描述的长度

        elif mode == 'infer':

            self.data_process = DataPreprocess(
                base_dir=base_dir,
            )

            self.valid_source_target_dict = self.data_process.load_source_target_dict(
                source_target_dict_dir=os.path.join(base_dir, 'cache_data/valid_source_target_dict.bin'))

            self.train_source_target_dict = self.data_process.load_source_target_dict(
                source_target_dict_dir=os.path.join(base_dir, 'cache_data/train_source_target_dict.bin'))




class Test:


    def test_Dataset(self):

        dataset_train = AnkiDataset(mode='train')

        n_h = 512
        n_embedding = 512
        n_vocab_target = 27651
        batch_size = 2

        batch_data_generator = BatchDataGenSequence(n_h=n_h, n_embedding=n_embedding, n_vocab_target=n_vocab_target, batch_size=batch_size, dataset=dataset_train.dataset['train'])

        ((batch_source_encoding, target_in), outputs) = batch_data_generator.__getitem__(0)

        print(batch_source_encoding)
        print(target_in)
        print(outputs)

        dataset_infer = AnkiDataset(mode='infer')


        valid_source_target_dict = dataset_infer.valid_source_target_dict

        valid_source_id_list = list(valid_source_target_dict.keys())

        print(valid_source_id_list[:10])


if __name__ == '__main__':


    test = Test()

    test.test_Dataset()



