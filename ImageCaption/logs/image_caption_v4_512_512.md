## 1.基本参数

### 1.1 数据集参数
    
    max_caption_length:37
    
    tokenizer:
    <NULL> id:  0
    <UNK> id:  1
    <START> id:  3
    <END> id:  4
    id:0 word: <NULL>
    id:1 word: <UNK>
    id:2 word: a
    id:3 word: <START>
    id:4 word: <END>
    
    train dataset tuple num: 32360
    
    valid dataset tuple num: 8095

### 1.2 模型参数

    
    词向量维度 n_embedding = 512
    
    隐藏层的维度 n_h = 512

## 2.实验记录

### 2.1   预训练的 CNN 对模型效果的影响

#### 实验 1 使用 inceptionV3

    (1)  采用 soft attention,  解码器的LSTM 和 输出的FC层中间使用 Dropout(0.5) 连接
    
    (2) 数据预处理
    
    过滤掉语料库中的数字和非法字符, 并对所有单词转换为小写
    
    词表大小: n_vocab=8500
    
    经过 CNN 抽取的图片向量维度 n_image_feature = 2048
    
    图片描述句子的长度: max_caption_length=37
    
    
    (3) 优化器参数
    
    epoch_num = 20, EarlyStopping('val_loss', patience=3)
    batch_size = 128
    optimizer='rmsprop'
        
    (4) 训练好的模型位于: models/image_caption_attention_model.h5
    
    (5) 训练过程
    
    训练在 epoch= 时停止,  在 epoch= 的时候达到验证集误差最小
    
    
    (7) 模型评价
    
    1.epoch=11 时的模型
    



