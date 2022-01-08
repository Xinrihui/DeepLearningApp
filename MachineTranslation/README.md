
# 神经机器翻译模型 (Neural Machine Translation, NMT)


![avatar](docs/images/tensorflow_logo.png) 


实现了多种机器翻译模型, 从基础的 seq2seq NMT 发展到 seq2seq with attention NMT 和 transformer NMT

## 项目结构
    .
    ├── config                  # 配置文件
    ├── dataset                 # 数据集
    ├── docs                    # 参考文献
    ├── lib                     # 代码库
        ├── data_generator      # 其中 tf_data_prepare_xrh.py 是生成训练和测试数据集的主程序
        ├── layers              # 包括各种 注意力层
        ├── models              # 包括各种 NMT 模型
        └── utils               # 包括模型评价程序
    ├── logs                    # 记录在不同 NMT 模型在不同数据集下的实验结果 
    ├── models                  # 预训练的模型
    ├── outs                    # 模型的推理结果
    ├── ref                     # 参考项目
    ├── src_bak                 # 项目的历史版本的源码
    ├── tools                   # 工具项目, 包括 Moses 
    └── nmt_solver_xrh.py       # 模型训练和推理的包装器主程序, 可以包装不同的 NMT 模型                    
    

## 1.seq2seq NMT

模型位置: [lib/models/ensemble_seq2seq_xrh.py](lib/models/ensemble_seq2seq_xrh.py)

### 1.1 模型设计

    (1) 编码器和解码器均为 4层 LSTM 的堆叠, 中间使用 dropout 连接

    (2) 在模型训练时, 解码器采用 teacher forcing 模式; 在模型推理时, 解码器采用 autoregressive 模式

    (3) 分词器采用 单词粒度, 词表大小为 50000

    (4) 对源序列反向后再输入编码器, 提升效果


### 1.2 实验结果

1.WMT-14 English-Germa

验证集/测试集  | Bleu1 | Bleu2 | Bleu3 | Bleu4 |
--------------| ------|-------| ------| ------|
newstest2013(dev) | 44.7 | 28  | 18.7 | 12.9 | 
newstest2014(test) | 36 | 20.1  | 12.1 | 7.6 | 

详细实验结果: [logs/nmt/WMT14_lstm_1000_1000.md](logs/nmt/WMT14_lstm_1000_1000.md)

### 1.3 Ref

1. Sequence to Sequence Learning with Neural Networks
2. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation

## 2.seq2seq with attention NMT

### 2.1 Bahdanau Attention

项目位置: [src_bak/attention/v1](src_bak/attention/v1/README.md)


### 2.2 Global Attention

模型位置: [lib/models/attention_seq2seq_xrh.py](lib/models/attention_seq2seq_xrh.py)

#### 2.2.1 模型设计

    (1) 编码器和解码器均为 4层 LSTM 的堆叠, 中间使用 dropout 连接, 并在最上面添加 global attention 层

    (2) 在模型训练时, 解码器采用 teacher forcing 模式; 在模型推理时, 解码器采用 auto-regressive 模式

    (3) 分词粒度为 word level, 词表大小为 50000
    

#### 2.2.2 实验结果

1.WMT-14 English-Germa

验证集/测试集  | Bleu1 | Bleu2 | Bleu3 | Bleu4 |
--------------| ------|-------| ------| ------|
newstest2013(dev) | 51.9 | 36.1  | 26.5 | 20 | 
newstest2014(test) | 46.9 | 31.5  | 22.3 | 16.3 | 

详细实验结果: [logs/nmt/WMT14_lstm_1000_1000.md](logs/nmt/WMT14_lstm_1000_1000.md)

### 2.3 Ref

1. Effective Approaches to Attention-based Neural Machine Translation
2. Neural machine translation by jointly learning to align and translate

## 3.transformer NMT

模型位置: [lib/models/transformer_seq2seq_xrh.py](lib/models/transformer_seq2seq_xrh.py)

### 3.1 模型设计

    (1) 实现了 Scaled Dot-Product Attention 并在其基础上搭建 Multi-Head Attention

    (2) 在模型训练时, 解码器采用 teacher forcing 模式; 在模型推理时, 解码器采用 autoregressive 模式

    (3) 使用 word piece 算法分别对源语料和目标语料进行分词, 分词粒度为 sub-word level

    (4) 实现了带 warmup 的 Adam 优化算法

    (5) 位置编码采用 正弦/余弦编码器
    
    (6) 实现了 Label Smoothing
    
    (7) 实现了动态划分 Batch, 很好地缓解了 GPU OOM
    
### 3.2 实验结果

1.TED-Portuguese-English

验证集/测试集  | Bleu1 | Bleu2 | Bleu3 | Bleu4 |
--------------| ------|-------| ------| ------|
dev           | 58.1 | 43.4 | 33.4 | 26   | 
test          | 57.9 | 43.2 | 33.4 | 26.1 | 

详细实验结果: [logs/nmt/TED_transformer_4_128.md](logs/nmt/TED_transformer_4_128.md)



2.WMT-14 English-Germa

验证集/测试集  | Bleu1 | Bleu2 | Bleu3 | Bleu4 |
--------------| ------|-------| ------| ------|
newstest2013(dev)  |   |    |    |     | 
newstest2014(test) |   |    |    |     | 

详细实验结果: [logs/nmt/WMT14_transformer_6_512.md](logs/nmt/WMT14_transformer_6_512.md)


### 3.3 Ref

1. Attention Is All You Need


## Note

1. 相关数据集下载详见: [dataset/readme.txt](dataset/readme.txt)

2. 软件环境 [Requirements](requirements.txt)

3. 硬件资源('钞'能力)

| CPU  | Mem | GPU | GPU-FP32 (float) |
| ------ | ----- | ----- | ----- |
| i9-12900K | 64GB | RTX3090(24GB)| [35.58TFLOPS](https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622) |








