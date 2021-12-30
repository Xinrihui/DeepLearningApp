#!/usr/bin/python
# -*- coding: UTF-8 -*-

import re
from lib.utils.bleu_xrh import *
from nltk.translate.bleu_score import corpus_bleu

class Evaluate:
    """
    评价翻译模型

    Author: xrh
    Date: 2021-9-26

    """
    def __init__(self, with_unk=True,
                       use_nltk=True,
                       _null_str='<NULL>',
                       _start_str='<START>',
                       _end_str='<END>',
                       _unk_str='<UNK>',

                 ):
        """

        :param  with_unk: 是否保留 unk
        :param  use_nltk: 是否使用 nltk 自带的 bleu 包
        :param  _null_str: 空字符
        :param  _start_str: 句子的开始字符
        :param  _end_str: 句子的结束字符
        :param  _unk_str: 未登录字符
        """

        self._null_str = re.escape(_null_str)  # 对文本（字符串）中所有 可能被解释为正则运算符的字符进行转义
        self._start_str = re.escape(_start_str)
        self._end_str = re.escape(_end_str)
        self._unk_str = re.escape(_unk_str)

        self.use_nltk = use_nltk

        if with_unk:  # 不删除 unk

            # 需要删除的控制词
            self.remove_word_re = re.compile(
                r'{}|{}|{}'.format(self._null_str, self._start_str, self._end_str))

        else:  # 删除 unk

            # 需要删除的控制词
            self.remove_word_re = re.compile(
                r'{}|{}|{}|{}'.format(self._null_str, self._start_str, self._end_str, self._unk_str))

    def evaluate_bleu(self, references, candidates, bleu_N=2):
        """
        使用 bleu 对翻译结果进行评价

        :param references: 平行语料库的句子(人工翻译)

        eg. 3个 机器翻译的句子对应 3 组人工翻译的句子, 每一组有 2 个句子
          [
            ['<START> A group of race horses run down a track carrying jockeys . <END>', '<START> A horse race . <END>'],

            ['<START> A man and a woman kissing . <END>', '<START> A man and woman kissing in front of a crowd of people . <END>']

            ['<START> A man racing on a motorbike <END>', '<START> A motorcycle rider drives fast around a curve on a track . <END>']
         ]
        :param candidates: 机器翻译的句子

            candidates:
            [

             'riding horses <NULL> by beach <NULL> by <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>',

             'women in a park at the bottom of a set of a man and a girl with pink hair sitting on a rock near a row of red and white flowers <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> them',

             'racers ride their bikes <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>',

            ]

        :param bleu_N:

              eg. bleu_N=2
                计算 bleu-1 和  bleu-2 的分数

              eg. bleu_N=4
                计算 bleu-1,  bleu-2, bleu-3, bleu-4 的分数

        :return:
        """

        references_arr = []
        candidates_arr = []

        assert len(references) == len(candidates)  # 机器翻译的结果个数和对照语料的组数要相同


        for candidate in candidates:  # 对待评价的候选句子进行清理

            # 删除 控制词
            candidate = self.remove_word_re.sub('', candidate)

            candidate_split = candidate.split()

            candidates_arr.append(candidate_split)

        for reference_list in references:  # 对平行语料的句子进行清理

            group = []

            for reference in reference_list:

                # 删除 控制词
                reference = self.remove_word_re.sub('', reference)

                reference_split = reference.split()

                group.append(reference_split)

            references_arr.append(group)

        if not self.use_nltk:
        # use xrh bleu

            bleu_score_dict_list = {}
            average_bleu_score_dict = {}

            for n in range(1, bleu_N+1):

                bleu_score_dict_list['{}-garm'.format(n)] = BleuScore.compute_bleu_sentences(references_arr, candidates_arr,
                                                                      N=n)

                average_bleu_score_dict['{}-garm'.format(n)] = np.average(bleu_score_dict_list['{}-garm'.format(n)])

        else:
        # use nltk bleu

            average_bleu_score_dict = {}

            for n in range(1, bleu_N + 1):
                average_bleu_score_dict['{}-garm'.format(n)] = corpus_bleu(references_arr, candidates_arr,
                                                           weights=np.array([1 / n] * n))


        return average_bleu_score_dict


    def output_candidate_and_reference(self, candidates,candidate_file, references, reference_dir, ref_corpus_num=1):
        """
        输出 机器翻译的文本 和 对照文本

        1.因为对照文本会有多份, 所以会输出多份对照文本文件, 以后缀数字作为区分

        :param candidates: 待打分的候选译文本列表
        :param candidate_file: 候选译文本输出的路径输出的路径
        :param references: 对照文本列表
        :param reference_dir: 对照文本输出的文件夹
        :param ref_corpus_num: 对照文本的份数

        :return:
        """
        # 处理 candidates
        out_candidates = []
        for candidate in candidates:  # 对待打分的候选句子进行清理

            # 删除 控制词
            candidate = self.remove_word_re.sub('', candidate)
            out_candidates.append(candidate)

        # 输出翻译结果到目标文件夹中
        with open(candidate_file, 'wb') as file:

            data = '\n'.join(out_candidates)
            data = data.encode('utf-8')
            file.write(data)

        # 处理 references
        out_references = {i: [] for i in range(ref_corpus_num)}

        for reference_list in references:

            for i in range(ref_corpus_num):

                # 删除 控制词
                reference = self.remove_word_re.sub('', reference_list[i])
                out_references[i].append(reference)

        for i in range(ref_corpus_num):

            # 输出对照文本
            with open(reference_dir+str(i)+'.txt', 'wb') as file:

                data = '\n'.join(out_references[i])
                data = data.encode('utf-8')
                file.write(data)



