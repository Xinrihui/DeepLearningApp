#!/usr/bin/python
# -*- coding: UTF-8 -*-

#  适用于 tensorflow >= 2.0, keras 被直接集成到 tensorflow 的内部
#  ref: https://keras.io/about/


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import tensorflow.keras as keras

from tensorflow.keras.models import Model

from tensorflow.keras import mixed_precision

from lib.evaluate_xrh import *
from lib.tf_data_tokenize_xrh import *

from lib.ensemble_seq2seq_xrh import *
from lib.seq2seq_xrh import *

import time
import configparser
import json


class MachineTranslation:
    """

    神经机器翻译模型的包装器

    1. 可以包装不同的 seq2seq 模型和 attention 模型

    2. 实现了基于 tf.data 的数据预处理 pipline, 使用 TextVectorization制作词典, 并用 StringLookup 做句子的向量化和反向量化

    3. 在配置中心中维护超参数

    Author: xrh
    Date: 2021-11-20

    """

    def __init__(self,
                 current_config,
                 vocab_source, vocab_target,
                 tokenizer_source=None, tokenizer_target=None,
                 use_pretrain=False,
                 ):
        """
        模型初始化

        :param vocab_source: 源语言的词典对象
        :param vocab_target: 目标语言的词典对象

        :param tokenizer_source:
        :param tokenizer_target:

        :param reverse_source:  是否将源序列反转

        :param use_pretrain: 使用训练好的模型

        """

        self.current_config = current_config

        self.n_h = int(current_config['n_h'])
        self.n_embedding = int(current_config['n_embedding'])

        self.max_seq_length = int(current_config['max_seq_length'])

        self.dropout_rates = json.loads(current_config['dropout_rates'])

        self.reverse_source = bool(int(self.current_config['reverse_source']))

        _null_str = current_config['_null_str']
        _start_str = current_config['_start_str']
        _end_str = current_config['_end_str']
        _unk_str = current_config['_unk_str']

        self.vocab_source = vocab_source
        self.vocab_target = vocab_target

        # 源语言的词表大小
        self.n_vocab_source = self.vocab_source.n_vocab

        # 目标语言的词表大小
        self.n_vocab_target = self.vocab_target.n_vocab


        print('model architecture param:')
        print('n_h:{}, n_embedding:{}, n_vocab_source:{}, n_vocab_target:{}'.format(self.n_h, self.n_embedding, self.n_vocab_source,
                                                                                    self.n_vocab_target))
        print('-------------------------')

        self._null = int(self.vocab_source.map_word_to_id(_null_str))  # 空
        self._start = int(self.vocab_source.map_word_to_id(_start_str))  # 句子的开始
        self._end = int(self.vocab_source.map_word_to_id(_end_str))  # 句子的结束
        self._unk = int(self.vocab_source.map_word_to_id(_unk_str))  # 未登录词

        # vocab_source 和 vocab_target 的标号不同
        self._null_target = int(self.vocab_target.map_word_to_id(_null_str))  # 空
        self._start_target = int(self.vocab_target.map_word_to_id(_start_str))  # 句子的开始
        self._end_target = int(self.vocab_target.map_word_to_id(_end_str))  # 句子的结束
        self._unk_target = int(self.vocab_target.map_word_to_id(_unk_str))  # 未登录词

        self.save_mode = current_config['save_mode']
        self.model_path = current_config['model_path']

        # 构建模型
        self.model_obj = Seq2seq(n_embedding=self.n_embedding, n_h=self.n_h, max_seq_length=self.max_seq_length,
                                  n_vocab_source=self.n_vocab_source, n_vocab_target=self.n_vocab_target,
                                  vocab_target=self.vocab_target,
                                  tokenizer_source=tokenizer_source, tokenizer_target=tokenizer_target,
                                  _start_target=self._start_target, _null_target=self._null_target,
                                  reverse_source=self.reverse_source,
                                  dropout_rates=self.dropout_rates)


        if use_pretrain:  # 载入训练好的模型

            if self.save_mode in ('hdf5', 'weight'):

                self.model_obj.model_train.load_weights(self.model_path)

            elif self.save_mode == 'SavedModel':

                self.model_obj.model_train = tf.saved_model.load(self.model_path)


    def _shuffle_dataset(self, dataset, buffer_size, batch_size):
        """
        数据混洗(shuffle), 分批(batch)和预取(prefetch)

        :param dataset:
        :return:
        """

        # 解开 batch, 数据集的粒度变为行
        dataset = dataset.unbatch()

        # tf.data 的数据混洗,分批和预取,
        # shuffle 后不能进行任何的 map 操作, 因为会改变 batch 中的数据行的组合
        dataset = dataset.shuffle(buffer_size).batch(batch_size)

        final_dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return final_dataset

    def fit_tf_data(self, train_dataset, valid_dataset, valid_source_target_dict, epoch_num,
                    batch_size, buffer_size,
                    ):
        """
        使用内置方法训练模型

        :param train_dataset: 训练数据生成器
        :param valid_dataset: 验证数据生成器
        :param valid_source_target_dict: 验证数据的字典, 用于计算 bleu

        :param epoch_num: 模型训练的 epoch 个数,  一般训练集所有的样本模型都见过一遍才算一个 epoch
        :param batch_size: 选择 min-Batch梯度下降时, 每一次输入模型的样本个数
        :param buffer_size:  shuffle 的窗口大小

        :return:
        """

        train_dataset_prefetch = self._shuffle_dataset(train_dataset, buffer_size, batch_size)
        valid_dataset_prefetch = self._shuffle_dataset(valid_dataset, buffer_size, batch_size)

        checkpoint_models_path = self.current_config['checkpoint_models_path']

        # Callbacks
        # [1] tensorboard
        # 在根目录下运行 tensorboard --logdir ./logs
        tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True,
                                                   write_images=True)

        # [2] 根据 epoch 调整学习率
        def scheduler(epoch, lr):
            if epoch <= 5:
                return lr
            else:
                return lr * 0.5
        dynamic_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

        # [3] 早停: 在验证集上, 损失经过 patience 次的迭代后, 仍然没有下降则暂停训练
        early_stop = EarlyStopping('val_loss', patience=5)

        # [4] 计算在验证集上的 bleu 指标
        model_checkpoint_with_eval = CheckoutCallback(current_config=self.current_config,
                                                      model_obj=self.model_obj,
                                                      vocab_obj=self.vocab_target, valid_source_target_dict=valid_source_target_dict,
                                                      )

        # Final callbacks
        callbacks = [model_checkpoint_with_eval]

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

        optimizer = tf.keras.optimizers.RMSprop(clipnorm=5)

        # optimizer = tf.keras.optimizers.Adam(clipnorm=5)
        # optimizer = tf.keras.optimizers.SGD(learning_rate=1.0, clipnorm=5)

        self.model_obj.model_train.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        history = self.model_obj.model_train.fit(
            x=train_dataset_prefetch,
            epochs=epoch_num,
            validation_data=valid_dataset_prefetch,
            verbose=1,
            callbacks=callbacks
            )

        print('final learning_rate:', round(self.model_obj.model_train.optimizer.lr.numpy(), 5))

        # 将训练好的模型持久化
        # self.model_obj.model_train.save(self.model_path)



    def inference(self, batch_source_dataset, max_seq_length):
        """
        使用训练好的模型进行推理

        :param batch_source_dataset:

        :return:
        """

        # batch_source_dataset shape (N_batch, encoder_length)

        decode_result = self.model_obj.predict(batch_source_dataset, max_seq_length)

        candidates = [sentence.numpy().decode('utf-8').strip() for sentence in decode_result]

        return candidates

class CheckoutCallback(keras.callbacks.Callback):
    """
    回调函数, 实现在每一次 epoch 后 checkout 训练好的模型,
    并且计算在验证集上的 bleu 分数

    """

    def __init__(self, current_config,
                 model_obj, vocab_obj, valid_source_target_dict,
                 ):
        """

        :param current_config: 配置中心
        :param model_obj: 模型对象
        :param vocab_obj: 目标语言的词典对象
        :param valid_source_target_dict: 源序列到目标序列的词典

        """

        keras.callbacks.Callback.__init__(self)

        self.model_obj = model_obj
        self.vocab_obj = vocab_obj

        self.save_mode = current_config['save_mode']
        self.max_seq_length = int(current_config['max_seq_length'])

        self.batch_source_dataset, self.references = self.prepare_data(batch_size=int(current_config['batch_size']),
                                                                           valid_source_target_dict=valid_source_target_dict)

        self.evaluate_obj = Evaluate(
            with_unk=True,
            _null_str=current_config['_null_str'],
            _start_str=current_config['_start_str'],
            _end_str=current_config['_end_str'],
            _unk_str=current_config['_unk_str'])

        self.checkpoint_models_path = current_config['checkpoint_models_path']

    def prepare_data(self, batch_size, valid_source_target_dict):
        """
        返回 图片的 embedding 向量 和 图片对应的 caption

        :param batch_size:
        :param valid_source_target_dict:
        :return:
        """

        source_list = list(valid_source_target_dict.keys())

        print('valid source seq num :{}'.format(len(source_list)))

        references = [valid_source_target_dict[source] for source in source_list]

        source_dataset = tf.data.Dataset.from_tensor_slices(source_list)

        batch_source_dataset = source_dataset.batch(batch_size)

        return batch_source_dataset, references

    def inference_bleu(self):
        """
        使用验证数据集进行推理, 并计算 bleu

        :return:
        """

        # batch_source_dataset shape (N_batch, encoder_length)

        decode_result = self.model_obj.predict(self.batch_source_dataset, self.max_seq_length)

        candidates = [sentence.numpy().decode('utf-8').strip() for sentence in decode_result]

        bleu_score, _ = self.evaluate_obj.evaluate_bleu(self.references, candidates)

        print()
        print('bleu_score:{}'.format(bleu_score))

    def on_epoch_end(self, epoch, logs=None):

        # checkout 模型

        if self.save_mode == 'hdf5':

            # 使用 hdf5 保存整个模型
            fmt = os.path.join(self.checkpoint_models_path, 'model.%02d-%.4f.h5')
            self.model_obj.model_train.save(fmt % (epoch, logs['val_loss']))

        elif self.save_mode == 'SavedModel':

            # 使用 SavedModel 保存整个模型
            fmt = os.path.join(self.checkpoint_models_path, 'model.%02d-%.4f')
            tf.saved_model.save(self.model_obj.model_train, fmt % (epoch, logs['val_loss']))

        elif self.save_mode == 'weight':
            # 'weight' 只保存权重
            fmt = os.path.join(self.checkpoint_models_path, 'model.%02d-%.4f')
            self.model_obj.model_train.save_weights(fmt % (epoch, logs['val_loss']))  # 保存模型的参数

        # 计算 bleu 分数
        self.inference_bleu()


class Test_WMT14_Eng_Ge_Dataset:

    def test_training(self, config_path='lib/config.ini', tag='DEFAULT'):

        # 0. 读取配置文件

        config = configparser.ConfigParser()
        config.read(config_path, 'utf-8')
        current_config = config[tag]

        print('current tag:{}'.format(tag))

        #  配置混合精度
        policy = mixed_precision.Policy(current_config['mixed_precision'])
        mixed_precision.set_global_policy(policy)

        # 1. 数据集的预处理, 运行 tf_data_tokenize_xrh.py 中的 DataPreprocess -> do_mian()
        dataset_obj = WMT14_Eng_Ge_Dataset(base_dir=current_config['base_dir'],
                                           cache_data_folder=current_config['cache_data_folder'], mode='train')


        # 2. 训练模型

        trainer = MachineTranslation(
            current_config=current_config,
            vocab_source=dataset_obj.vocab_source,
            vocab_target=dataset_obj.vocab_target,
            tokenizer_source=dataset_obj.tokenizer_source, tokenizer_target=dataset_obj.tokenizer_target,
            use_pretrain=False
        )
        # use_pretrain=True: 在已有的模型参数基础上, 进行更进一步的训练

        batch_size = int(current_config['batch_size'])
        buffer_size = int(current_config['buffer_size'])
        epoch_num = int(current_config['epoch_num'])

        trainer.fit_tf_data(train_dataset=dataset_obj.train_dataset, valid_dataset=dataset_obj.valid_dataset,
                            valid_source_target_dict=dataset_obj.valid_source_target_dict,
                            epoch_num=epoch_num, batch_size=batch_size, buffer_size=buffer_size)

    def test_evaluating(self, config_path='lib/config.ini', tag='DEFAULT'):

        # 0. 读取配置文件
        config = configparser.ConfigParser()
        config.read(config_path, 'utf-8')
        current_config = config[tag]

        #  配置混合精度
        policy = mixed_precision.Policy(current_config['mixed_precision'])
        mixed_precision.set_global_policy(policy)

        # 1. 数据集的预处理, 运行 tf_data_tokenize_xrh.py 中的 DataPreprocess -> do_mian()
        dataset_obj = WMT14_Eng_Ge_Dataset(base_dir=current_config['base_dir'],
                                           cache_data_folder=current_config['cache_data_folder'], mode='infer')

        batch_size = int(current_config['batch_size'])
        max_seq_length = int(current_config['test_max_seq_length'])

        # 2.模型推理

        infer = MachineTranslation(
            current_config=current_config,
            vocab_source=dataset_obj.vocab_source,
            vocab_target=dataset_obj.vocab_target,
            tokenizer_source=dataset_obj.tokenizer_source, tokenizer_target=dataset_obj.tokenizer_target,
            use_pretrain=True
        )

        source_list = list(dataset_obj.test_source_target_dict.keys())

        print('valid source seq num :{}'.format(len(source_list)))

        references = [dataset_obj.test_source_target_dict[source] for source in source_list]

        source_dataset = tf.data.Dataset.from_tensor_slices(source_list)

        batch_source_dataset = source_dataset.batch(batch_size)

        candidates = infer.inference(batch_source_dataset, max_seq_length)

        print('\ncandidates:')
        for i in range(0, 10):
            print('[{}] {}'.format(i, candidates[i]))

        print('\nreferences:')
        for i in range(0, 10):
            print('[{}] {}'.format(i, references[i]))


        evaluate_obj = Evaluate(
            with_unk=True,
            _null_str=current_config['_null_str'],
            _start_str=current_config['_start_str'],
            _end_str=current_config['_end_str'],
            _unk_str=current_config['_unk_str'])


        bleu_score, _ = evaluate_obj.evaluate_bleu(references, candidates, bleu_N=4)

        print('bleu_score:{}'.format(bleu_score))


if __name__ == '__main__':
    test = Test_WMT14_Eng_Ge_Dataset()

    # TODO: 每次实验前
    #  1. 更改最终模型存放的路径
    #  2. 运行脚本  clean_training_cache_file.bat

    test.test_training(tag='TEST')

    # test.test_evaluating(tag='TEST')
