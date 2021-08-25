
# 机器翻译模型(Machine Translation)

## 1.基于 LSTM + seq2seq + attention 的机器翻译模型

### 1.1 模型设计


    (1) 实现了编码器(encoder): 采用双向的 LSTM, 所有时间步的输出作为 注意力机制(attention)模块的输入

    (2) 实现了注意力机制(attention)模块:

        1. 对 encoder 所有时间步的输出 a 求加权和, 得到 上下文context

        2. 权重通过 a 与 上一个时间步的隐藏状态 s_prev 拼接后求 dense + softmax 得到

    (3) 实现了解码器(decoder):

        1.采用LSTM, 每一个时间步的输入为 attention模块的输出context 与 上一个时间步的输出的最优分类的one-hot向量 的拼接

        2.第一个时间步的 初始隐藏状态s0 和 细胞状态c0 为 encoder 最后一个时间步的隐藏状态和细胞状态

    (4) 实现了基于 beamsearch 的推理(inference)

    (5) 使用 bleu 对输出序列(翻译结果)进行评价



### 1.2 实验结果

    数据集来源: 吴恩达深度学习课程-作业5 机器翻译








