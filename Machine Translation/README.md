
# 机器翻译模型(Machine Translation)

## 1.使用 Keras 实现基于 LSTM + seq2seq + attention 的机器翻译模型

### 1.1 模型设计


    (1) 实现了编码器(encoder): 采用双向的 LSTM, 所有时间步的输出作为 注意力机制(attention)模块的输入

    (2) 采用字符粒度的 wordEmbedding, 输入 LSTM 是经过 one-hot 化的字符

    (3) 实现了注意力机制(attention)模块:

        1. 对 encoder 所有时间步的输出 a 求加权和, 得到 上下文context

        2. 权重通过 a 与 上一个时间步的隐藏状态 s_prev 拼接后求 dense + softmax 得到

    (4) 实现了解码器(decoder):

        1.采用LSTM, 每一个时间步的输入为 attention模块的输出context 与 上一个时间步的输出的最优分类的one-hot向量 的拼接

        2.第一个时间步的 初始隐藏状态s0 和 细胞状态c0 为 encoder 最后一个时间步的隐藏状态和细胞状态

    (5) 实现了基于 beamsearch 的推理(inference)

    (6) 实现了 bleu 评价指标(算法), 并使用它对输出序列(翻译结果)进行评价



### 1.2 实验结果

    1.日期翻译数据集

    |  human date  |   machine date
    |  待翻译的日期     翻译后的标准日期
    | ------------ | ------------ |
    | '9 may 1998' | '1998-05-09' |
    | '10.11.19'   | '2019-11-10' |
    | '9/10/70'    | '1970-09-10' |
    | 'saturday april 28 1990' | '1990-04-28' |
    | 'thursday january 26 1995' | '1995-01-26' |

    2.模型训练

    训练数据集 n=8000

    epoch_num=200, batch_size=2048

    在训练集上
    loss: 0.5548, decoder_output_accuracy: 0.9946

    在验证集上
    val_loss: 0.7749, val_decoder_output_loss: 0.0213,  val_decoder_output_accuracy: 0.9912

    因为输出序列是定长的, 所以输出序列各个时间步的准确率为
    val_decoder_output_1_accuracy: 0.9912 ... val_decoder_output_9_accuracy: 0.9450

    3.模型评价

    测试数据集 n=1000

    1.推理时采用 beamsearch (窗口大小 k = 3), 我们取 bleu 得分最高的作为此样本的预测序列

    2.词元(term)的粒度

    (1) '1978-12-21' 使用分隔符 '-' 切分为 3个 term ['1978','12','21'],
        计算 bleu 时设置 N_gram 的长度上限为 2(仅仅考虑 1-gram, 2-gram)

    (2) '1978-12-21' 切分为 10个 term ['1', '9', '7', '8', '-', '1', '2', '-', '2', '1']

    我们采用 第(2) 种词元的粒度, 考虑下面的例子
    reference = '1978-12-21'
    candidate1 = '1988-12-21'
    candidate2 = '8891-12-21'

    显然, candidate1 比 candidate2效果好, 但是若采用第(1)种词元粒度, 它们两的 score 是相同的;
    测试记录详见: machine_translation_with_attention_xrh.ipynb

    n=200 条测试数据, bleu score( up to 4-gram average)  = 0.965 (96.5)
    推理一条数据耗时 5s

## Ref

1.https://github.com/enggen/Deep-Learning-Coursera

2.论文 Neural machine translation by jointly learning to align and translate







