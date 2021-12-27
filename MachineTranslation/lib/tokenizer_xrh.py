#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf


from tensorflow.keras.layers import TextVectorization, StringLookup

import tensorflow_text as tf_text

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

class SubwordTokenizer(tf.keras.Model):
    """
    使用 tensorflow_text.BertTokenizer 构建的基于 subword 的分词器

    Author: xrh
    Date: 2021-12-15

    ref:
    https://www.tensorflow.org/text/guide/subwords_tokenizer
    https://www.tensorflow.org/text/api_docs/python/text/BertTokenizer

    """

    def __init__(self, fixed_seq_length, reserved_tokens, vocab_list):
        """

        :param fixed_seq_length: 指定的序列长度
        :param reserved_tokens: 保留的控制字符,
                ["[NULL]", "[UNK]", "[START]", "[END]"]
        :param vocab_list: 词典
        """

        super(SubwordTokenizer, self).__init__(self)

        self._reserved_tokens = reserved_tokens
        self.fixed_seq_length = fixed_seq_length
        self.start = tf.argmax(tf.constant(reserved_tokens) == "[START]")
        self.end = tf.argmax(tf.constant(reserved_tokens) == "[END]")

        lookup = tf.lookup.StaticVocabularyTable(
            num_oov_buckets=1,
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=vocab_list,
                values=tf.range(len(vocab_list), dtype=tf.int64)))

        bert_tokenizer_params = dict(lower_case=True)

        self.tokenizer = tf_text.BertTokenizer(lookup, **bert_tokenizer_params)

        self.vocab = tf.Variable(vocab_list)

        ## Create the signatures for export:

        self.tokenize_fixed.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string))

        # Include a tokenize signature for a batch of strings.
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string))

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))

        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

    def add_start_end(self, batch_ragged):
        """
        在句子的首尾添加控制字符

        :param batch_ragged:
        :return:
        """
        N_batch = batch_ragged.bounding_shape()[0]
        starts = tf.fill([N_batch, 1], self.start)
        ends = tf.fill([N_batch, 1], self.end)

        return tf.concat([starts, batch_ragged, ends], axis=1)

    # @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None]), tf.TensorSpec(dtype=tf.int64, shape=None)])
    # TODO: 报错 TypeError: Dimension value must be integer or None
    @tf.function
    def tokenize_fixed(self, strings):
        """
        返回固定序列长度(fixed_seq_length)的 tensor

        :param strings:
        :return:
        """

        enc = self.tokenizer.tokenize(strings)

        # Merge the `word` and `word-piece` axes.
        enc = enc.merge_dims(-2, -1)
        enc = self.add_start_end(enc)

        return enc.to_tensor(shape=[None, self.fixed_seq_length])

    @tf.function
    def tokenize(self, strings):
        """
        返回不规则的 tensor(RaggedTensor)

        :param strings:
        :return:
        """

        enc = self.tokenizer.tokenize(strings)

        # Merge the `word` and `word-piece` axes.
        enc = enc.merge_dims(-2, -1)
        enc = self.add_start_end(enc)

        return enc

    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        return words

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)


class SpaceTokenizer(tf.keras.Model):
    """
    使用 TextVectorization 构建的空格分词器

    Author: xrh
    Date: 2021-12-15

    ref:
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization

    """

    def __init__(self, corpus, max_tokens, fixed_seq_length):
        """

        :param corpus: 待分词的语料
        :param max_tokens: 词典大小
        :param fixed_seq_length: 指定的序列长度
        """

        super(SpaceTokenizer, self).__init__(self)

        self.fixed_seq_length = fixed_seq_length

        self.tokenizer = TextVectorization(
            standardize=None,
            output_sequence_length=None,
            max_tokens=max_tokens)

        self.tokenizer.adapt(corpus)

        self.vocab_list = self.tokenizer.get_vocabulary()

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def tokenize_fixed(self, strings):
        """
        返回序列长度(self.max_seq_length)为固定的 tensor

        :param strings:
        :return:
        """
        tensor = self.tokenizer.call(strings)

        ragged = tf.RaggedTensor.from_tensor(tensor, padding=0)

        return ragged.to_tensor(shape=[None, self.fixed_seq_length])

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def tokenize(self, strings):
        """
        返回不规则的 tensor

        :param strings:
        :return:
        """
        tensor = self.tokenizer.call(strings)

        ragged = tf.RaggedTensor.from_tensor(tensor, padding=0)

        return ragged


class Test:

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
                # sentence = '[START] ' + sentence + ' [END]'
                text_data.append(sentence)

            return text_data

    def test_SpaceTokenizer(self):

        corpus = self.load_corpus_data('../dataset/WMT-14-English-Germa/newstest2012.de')

        # 首尾添加 控制字符
        corpus = ['[START] ' + sentence + ' [END]' for sentence in corpus]

        corpus_dataset = tf.data.Dataset.from_tensor_slices(corpus).batch(2)

        max_tokens = 20000
        fixed_seq_length = 50

        model_tokenizer = SpaceTokenizer(corpus_dataset, max_tokens, fixed_seq_length)

        corpus_vector = corpus_dataset.map(lambda batch_text: model_tokenizer.tokenize(batch_text))

        print(list(corpus_vector)[0])

        corpus_vector_fixed = corpus_dataset.map(lambda batch_text: model_tokenizer.tokenize_fixed(batch_text))

        print(list(corpus_vector_fixed)[0])

    def test_SubwordTokenizer(self):

        corpus = self.load_corpus_data('../dataset/WMT-14-English-Germa/newstest2012.de')

        target_dataset = tf.data.Dataset.from_tensor_slices(corpus).batch(2)

        bert_tokenizer_params = dict(lower_case=True)
        reserved_tokens = ["", "[UNK]", "[START]", "[END]"]

        bert_vocab_args = dict(
            # The target vocabulary size
            vocab_size=5000,
            # Reserved tokens that must be included in the vocabulary
            reserved_tokens=reserved_tokens,
            # Arguments for `text.BertTokenizer`
            bert_tokenizer_params=bert_tokenizer_params,
            # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
            learn_params={},
        )

        target_vocab = bert_vocab.bert_vocab_from_dataset(
            target_dataset.prefetch(buffer_size=tf.data.AUTOTUNE),
            **bert_vocab_args
        )

        fixed_seq_length = 50
        tokenizer = SubwordTokenizer(fixed_seq_length, reserved_tokens, target_vocab)

        for batch_seq in target_dataset.take(1):
            print(batch_seq.numpy())

        token_vector_fixed = tokenizer.tokenize_fixed(batch_seq)
        print(token_vector_fixed)

        token_vector = tokenizer.tokenize(batch_seq)
        print(token_vector)

        sentence = tokenizer.detokenize(token_vector)
        sentence = tf.strings.reduce_join(sentence, separator=' ', axis=-1)
        print(sentence)

        tokens = tokenizer.lookup(token_vector)
        print(tokens)

if __name__ == '__main__':

    test = Test()

    # test.test_SpaceTokenizer()

    test.test_SubwordTokenizer()



