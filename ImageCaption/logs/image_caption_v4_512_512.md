## 1.基本参数

### 1.1 数据集参数
    
    word_index length: 8868
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
    
    采用默认的划分方法将数据集切分为训练集, 验证集和测试集
    train dataset tuple num: 30000
    valid dataset tuple num: 5000
    test dataset tuple num: 5000

### 1.2 模型参数

    词向量维度 n_embedding = 512
    
    隐藏层的维度 n_h = 512

## 2.实验记录

### 2.1 预训练的 CNN 和 输出层 的选择对模型效果的影响

#### 实验 1 使用 inceptionV3, 输出层使用自定义的输出层

    (1)  采用 soft attention,  解码器的LSTM 和 输出层 中间使用 Dropout(0.5) 连接, 输出层使用论文中自定义的输出层
         dropout 正则化 dropout_rates = (0.5,)
    
    (2) 数据预处理
    
    过滤掉语料库中的数字和非法字符, 并对所有单词转换为小写
    
    词表大小: n_vocab = 8868
    
    经过 inceptionV3 抽取的图片向量维度: (train_seq_length=64, n_image_feature = 2048)
    
    编码器长度: train_seq_length=64
    
    图片描述句子的长度: max_caption_length=37
    
    解码器长度: infer_seq_length = 36 
    
    (3) 优化器参数
    
    epoch_num = 15, EarlyStopping('val_loss', patience=10)
    batch_size = 256
    optimizer='rmsprop'
        
    (4) 训练好的模型位于: models/image_caption_attention_model.h5
    
    (5) 训练过程
    
    在每一个 epoch 结束时都对模型进行持久化(checkpoint), 并计算在验证集上的 bleu 得分
    
    
    (6) 模型评价
    
    1.epoch=7 时的模型在验证集上的 bleu 得分最高
    
    bleu_score:{'1-garm': 0.63, '2-garm': 0.41}
    
    在测试集上的结果
    
    candidates:
    a brown dog is playing in the snow <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a boy is jumping into a swimming pool <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a shirtless man in a green shirt and a hat is standing in a crowded street <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a group of people sit on a bench in front of a large building <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a football player in a red jersey and white jersey <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a brown dog is running through the grass <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    two girls pose in a crowd <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a dog is licking its nose <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a black dog is running in the sand <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a basketball player in a white uniform is challenging the player in a white jersey <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    
    references:
    ['<START> the dogs are in the snow in front of a fence   <END>', '<START> the dogs play on the snow   <END>', '<START> two brown dogs playfully fight in the snow   <END>', '<START> two brown dogs wrestle in the snow   <END>', '<START> two dogs playing in the snow   <END>']
    ['<START> a brown and white dog swimming towards some in the pool <END>', '<START> a dog in a swimming pool swims toward sombody we cannot see   <END>', '<START> a dog swims in a pool near a person   <END>', '<START> small dog is paddling through the water in a pool   <END>', '<START> the small brown and white dog is in the pool   <END>']
    ['<START> a man and a woman in festive costumes dancing   <END>', '<START> a man and a woman with feathers on her head dance   <END>', '<START> a man and a woman wearing decorative costumes and dancing in a crowd of onlookers   <END>', '<START> one performer wearing a feathered headdress dancing with another performer in the streets <END>', '<START> two people are dancing with drums on the right and a crowd behind them   <END>']
    ['<START> a couple of people sit outdoors at a table with an umbrella and talk   <END>', '<START> three people are sitting at an outside picnic bench with an umbrella   <END>', '<START> three people sit at an outdoor cafe   <END>', '<START> three people sit at an outdoor table in front of a building painted like the union jack   <END>', '<START> three people sit at a picnic table outside of a building painted like a union jack   <END>']
    ['<START> a man is wearing a sooners red football shirt and helmet   <END>', '<START> a oklahoma sooners football player wearing his jersey number   <END>', '<START> a sooners football player weas the number and black armbands   <END>', '<START> guy in red and white football uniform <END>', '<START> the american footballer is wearing a red and white strip   <END>']
    ['<START> a brown dog running <END>', '<START> a brown dog running over grass   <END>', '<START> a brown dog with its front paws off the ground on a grassy surface near red and purple flowers   <END>', '<START> a dog runs across a grassy lawn near some flowers   <END>', '<START> a yellow dog is playing in a grassy area near flowers   <END>']
    ['<START> a girl with dark brown hair and eyes in a blue scarf is standing next to a girl in a fur edged coat   <END>', '<START> an asian boy and an asian girl are smiling in a crowd of people   <END>', '<START> the girls were in the crowd   <END>', '<START> two dark haired girls are in a crowd   <END>', '<START> two girls are looking past each other in different directions while standing in a crowd   <END>']
    ['<START> a dog with its mouth opened   <END>', '<START> brown and white dog yawning   <END>', '<START> close-up of dog in profile with mouth open   <END>', '<START> dog yawns <END>', "<START> the dog 's mouth is open like he is yawning   <END>"]
    ['<START> a black dog emerges from the water onto the sand   holding a white object in its mouth   <END>', '<START> a black dog emerges from the water with a white ball in its mouth   <END>', '<START> a black dog on a beach carrying a ball in its mouth   <END>', '<START> a black dog walking out of the water with a white ball in his mouth   <END>', '<START> the black dog jumps out of the water with something in its mouth   <END>']
    ['<START> a player from the white and green highschool team dribbles down court defended by a player from the other team   <END>', '<START> four basketball players in action   <END>', '<START> four men playing basketball   two from each team   <END>', '<START> two boys in green and white uniforms play basketball with two boys in blue and white uniforms   <END>', '<START> young men playing basketball in a competition   <END>']
    
    bleu_score:{'1-garm': 0.6358068429961851, '2-garm': 0.4157117863012699}

#### 实验 2 使用 inceptionV3, 输出层使用 FC 层

    (1)  采用 soft attention,  解码器的LSTM 和 输出层 中间使用 Dropout(0.5) 连接, 输出层使用 FC 层
         dropout 正则化 dropout_rates = (0.5,) 
         
    (2) 数据预处理
    
    过滤掉语料库中的数字和非法字符, 并对所有单词转换为小写
    
    词表大小: n_vocab = 8868
    
    经过 inceptionV3 抽取的图片向量维度: (train_seq_length=64, n_image_feature = 2048)
    
    编码器长度: train_seq_length=64
    
    图片描述句子的长度: max_caption_length=37
    
    解码器长度: infer_seq_length = 36 
    
    (3) 优化器参数
    
    epoch_num = 15, EarlyStopping('val_loss', patience=10)
    batch_size = 256
    optimizer='rmsprop'
        
    (4) 训练好的模型位于: models/image_caption_attention_model.h5
    
    (5) 训练过程
    
    在每一个 epoch 结束时都对模型进行持久化(checkpoint), 并计算在验证集上的 bleu 得分
    
    
    (6) 模型评价
    
    1.epoch=5 时的模型在验证集上的 bleu 得分最高
    
    bleu_score:{'1-garm': 0.63, '2-garm': 0.41}
    
    在测试集上的结果
    
    candidates:
    a brown dog is running on a large field <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a girl is playing with a pool <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a woman in a red shirt and a white shirt is standing in a city street <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a man is sitting on a bench with a white umbrella <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a football player in a red jersey is running <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a brown dog is running through the grass <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a woman with a woman in a black shirt and a woman <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a dog is running with a ball <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a black dog is running through the water <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a basketball player is playing basketball <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    
    references:
    ['<START> the dogs are in the snow in front of a fence   <END>', '<START> the dogs play on the snow   <END>', '<START> two brown dogs playfully fight in the snow   <END>', '<START> two brown dogs wrestle in the snow   <END>', '<START> two dogs playing in the snow   <END>']
    ['<START> a brown and white dog swimming towards some in the pool <END>', '<START> a dog in a swimming pool swims toward sombody we cannot see   <END>', '<START> a dog swims in a pool near a person   <END>', '<START> small dog is paddling through the water in a pool   <END>', '<START> the small brown and white dog is in the pool   <END>']
    ['<START> a man and a woman in festive costumes dancing   <END>', '<START> a man and a woman with feathers on her head dance   <END>', '<START> a man and a woman wearing decorative costumes and dancing in a crowd of onlookers   <END>', '<START> one performer wearing a feathered headdress dancing with another performer in the streets <END>', '<START> two people are dancing with drums on the right and a crowd behind them   <END>']
    ['<START> a couple of people sit outdoors at a table with an umbrella and talk   <END>', '<START> three people are sitting at an outside picnic bench with an umbrella   <END>', '<START> three people sit at an outdoor cafe   <END>', '<START> three people sit at an outdoor table in front of a building painted like the union jack   <END>', '<START> three people sit at a picnic table outside of a building painted like a union jack   <END>']
    ['<START> a man is wearing a sooners red football shirt and helmet   <END>', '<START> a oklahoma sooners football player wearing his jersey number   <END>', '<START> a sooners football player weas the number and black armbands   <END>', '<START> guy in red and white football uniform <END>', '<START> the american footballer is wearing a red and white strip   <END>']
    ['<START> a brown dog running <END>', '<START> a brown dog running over grass   <END>', '<START> a brown dog with its front paws off the ground on a grassy surface near red and purple flowers   <END>', '<START> a dog runs across a grassy lawn near some flowers   <END>', '<START> a yellow dog is playing in a grassy area near flowers   <END>']
    ['<START> a girl with dark brown hair and eyes in a blue scarf is standing next to a girl in a fur edged coat   <END>', '<START> an asian boy and an asian girl are smiling in a crowd of people   <END>', '<START> the girls were in the crowd   <END>', '<START> two dark haired girls are in a crowd   <END>', '<START> two girls are looking past each other in different directions while standing in a crowd   <END>']
    ['<START> a dog with its mouth opened   <END>', '<START> brown and white dog yawning   <END>', '<START> close-up of dog in profile with mouth open   <END>', '<START> dog yawns <END>', "<START> the dog 's mouth is open like he is yawning   <END>"]
    ['<START> a black dog emerges from the water onto the sand   holding a white object in its mouth   <END>', '<START> a black dog emerges from the water with a white ball in its mouth   <END>', '<START> a black dog on a beach carrying a ball in its mouth   <END>', '<START> a black dog walking out of the water with a white ball in his mouth   <END>', '<START> the black dog jumps out of the water with something in its mouth   <END>']
    ['<START> a player from the white and green highschool team dribbles down court defended by a player from the other team   <END>', '<START> four basketball players in action   <END>', '<START> four men playing basketball   two from each team   <END>', '<START> two boys in green and white uniforms play basketball with two boys in blue and white uniforms   <END>', '<START> young men playing basketball in a competition   <END>']
    
    bleu_score:{'1-garm': 0.6407726223939458, '2-garm': 0.42138132356762925}

对比 实验1 实验2 可以看出, 输出层使用 FC层, 结果更稳定

#### 实验 3 使用 VGG-19, 输出层使用 FC 层

    (1)  采用 soft attention,  解码器的LSTM 和 输出层 中间使用 Dropout(0.5) 连接, 输出层使用 FC 层
         dropout 正则化 dropout_rates = (0.5,)
         
    (2) 数据预处理
    
    过滤掉语料库中的数字和非法字符, 并对所有单词转换为小写
    
    词表大小: n_vocab = 8868
    
    经过 VGG-19 抽取的图片向量维度: (train_seq_length=196, n_image_feature = 512)
    
    编码器长度: train_seq_length=196
    
    图片描述句子的长度: max_caption_length=37
    
    解码器长度: infer_seq_length = 36 
    
    (3) 优化器参数
    
    epoch_num = 15, EarlyStopping('val_loss', patience=10)
    batch_size = 256
    optimizer='rmsprop'
        
    (4) 训练好的模型位于: models/image_caption_attention_model.h5
    
    (5) 训练过程
    
    在每一个 epoch 结束时都对模型进行持久化(checkpoint), 并计算在验证集上的 bleu 得分
    
    
    (6) 模型评价
    
    1.epoch=5 时的模型在验证集上的 bleu 得分最高
    
    在测试集上的结果
    
    candidates:
    a brown dog is running through the snow <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a white dog is jumping into the air <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a man in a pink shirt is dancing in a parade <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a man in a red dress is standing in front of a red umbrella <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a football player in red is holding a football <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a brown dog is running through the grass <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a woman in a black and white shirt is smiling <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a brown and white dog is running with a stick <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a black dog is running through the snow <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a group of men are playing basketball <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    
    references:
    ['<START> the dogs are in the snow in front of a fence   <END>', '<START> the dogs play on the snow   <END>', '<START> two brown dogs playfully fight in the snow   <END>', '<START> two brown dogs wrestle in the snow   <END>', '<START> two dogs playing in the snow   <END>']
    ['<START> a brown and white dog swimming towards some in the pool <END>', '<START> a dog in a swimming pool swims toward sombody we cannot see   <END>', '<START> a dog swims in a pool near a person   <END>', '<START> small dog is paddling through the water in a pool   <END>', '<START> the small brown and white dog is in the pool   <END>']
    ['<START> a man and a woman in festive costumes dancing   <END>', '<START> a man and a woman with feathers on her head dance   <END>', '<START> a man and a woman wearing decorative costumes and dancing in a crowd of onlookers   <END>', '<START> one performer wearing a feathered headdress dancing with another performer in the streets <END>', '<START> two people are dancing with drums on the right and a crowd behind them   <END>']
    ['<START> a couple of people sit outdoors at a table with an umbrella and talk   <END>', '<START> three people are sitting at an outside picnic bench with an umbrella   <END>', '<START> three people sit at an outdoor cafe   <END>', '<START> three people sit at an outdoor table in front of a building painted like the union jack   <END>', '<START> three people sit at a picnic table outside of a building painted like a union jack   <END>']
    ['<START> a man is wearing a sooners red football shirt and helmet   <END>', '<START> a oklahoma sooners football player wearing his jersey number   <END>', '<START> a sooners football player weas the number and black armbands   <END>', '<START> guy in red and white football uniform <END>', '<START> the american footballer is wearing a red and white strip   <END>']
    ['<START> a brown dog running <END>', '<START> a brown dog running over grass   <END>', '<START> a brown dog with its front paws off the ground on a grassy surface near red and purple flowers   <END>', '<START> a dog runs across a grassy lawn near some flowers   <END>', '<START> a yellow dog is playing in a grassy area near flowers   <END>']
    ['<START> a girl with dark brown hair and eyes in a blue scarf is standing next to a girl in a fur edged coat   <END>', '<START> an asian boy and an asian girl are smiling in a crowd of people   <END>', '<START> the girls were in the crowd   <END>', '<START> two dark haired girls are in a crowd   <END>', '<START> two girls are looking past each other in different directions while standing in a crowd   <END>']
    ['<START> a dog with its mouth opened   <END>', '<START> brown and white dog yawning   <END>', '<START> close-up of dog in profile with mouth open   <END>', '<START> dog yawns <END>', "<START> the dog 's mouth is open like he is yawning   <END>"]
    ['<START> a black dog emerges from the water onto the sand   holding a white object in its mouth   <END>', '<START> a black dog emerges from the water with a white ball in its mouth   <END>', '<START> a black dog on a beach carrying a ball in its mouth   <END>', '<START> a black dog walking out of the water with a white ball in his mouth   <END>', '<START> the black dog jumps out of the water with something in its mouth   <END>']
    ['<START> a player from the white and green highschool team dribbles down court defended by a player from the other team   <END>', '<START> four basketball players in action   <END>', '<START> four men playing basketball   two from each team   <END>', '<START> two boys in green and white uniforms play basketball with two boys in blue and white uniforms   <END>', '<START> young men playing basketball in a competition   <END>']

    bleu_score:{'1-garm': 0.6362045291267041, '2-garm': 0.42239847687566007}
    
对比 实验2 实验3 可以看出, 预训练的 CNN 使用 inceptionV3, 模型训练速度更快, 在测试集上的效果更好


### 2.2 训练时 batch_size 的大小 对模型效果的影响

#### 实验 4 使用 inceptionV3,  batch_size=512

    (1)  采用 soft attention,  解码器的LSTM 和 输出层 中间使用 Dropout(0.5) 连接, 输出层使用 FC 层
         dropout 正则化 dropout_rates = (0.5,)
         
    (2) 数据预处理
    
    过滤掉语料库中的数字和非法字符, 并对所有单词转换为小写
    
    词表大小: n_vocab = 8868
    
    经过 inceptionV3 抽取的图片向量维度: (train_seq_length=64, n_image_feature = 2048)
    
    编码器长度: train_seq_length=64
    
    图片描述句子的长度: max_caption_length=37
    
    解码器长度: infer_seq_length = 36 
    
    (3) 优化器参数
    
    epoch_num = 15, EarlyStopping('val_loss', patience=10)
    batch_size = 512
    optimizer='rmsprop'
        
    (4) 训练好的模型位于: models/image_caption_attention_model.h5
    
    (5) 训练过程
    
    在每一个 epoch 结束时都对模型进行持久化(checkpoint), 并计算在验证集上的 bleu 得分
    
    
    (6) 模型评价
    
    1.epoch=9 时的模型在验证集上的 bleu 得分最高
    
    bleu_score:{'1-garm': 0.6241007073742428, '2-garm': 0.4135677174187707}
    
    在测试集上的结果
    
    candidates:
    a brown dog is running through the snow <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a small child is swimming in a pool <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a man in a blue shirt and white shorts is walking in the street <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a group of people are sitting on a bench <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a football player in a red uniform <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a brown dog is running through the grass <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a group of people are standing in front of a crowd <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a brown dog is running in the air <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a black dog is running through the water <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a basketball player is being tackled by the player in the air <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    
    references:
    ['<START> the dogs are in the snow in front of a fence   <END>', '<START> the dogs play on the snow   <END>', '<START> two brown dogs playfully fight in the snow   <END>', '<START> two brown dogs wrestle in the snow   <END>', '<START> two dogs playing in the snow   <END>']
    ['<START> a brown and white dog swimming towards some in the pool <END>', '<START> a dog in a swimming pool swims toward sombody we cannot see   <END>', '<START> a dog swims in a pool near a person   <END>', '<START> small dog is paddling through the water in a pool   <END>', '<START> the small brown and white dog is in the pool   <END>']
    ['<START> a man and a woman in festive costumes dancing   <END>', '<START> a man and a woman with feathers on her head dance   <END>', '<START> a man and a woman wearing decorative costumes and dancing in a crowd of onlookers   <END>', '<START> one performer wearing a feathered headdress dancing with another performer in the streets <END>', '<START> two people are dancing with drums on the right and a crowd behind them   <END>']
    ['<START> a couple of people sit outdoors at a table with an umbrella and talk   <END>', '<START> three people are sitting at an outside picnic bench with an umbrella   <END>', '<START> three people sit at an outdoor cafe   <END>', '<START> three people sit at an outdoor table in front of a building painted like the union jack   <END>', '<START> three people sit at a picnic table outside of a building painted like a union jack   <END>']
    ['<START> a man is wearing a sooners red football shirt and helmet   <END>', '<START> a oklahoma sooners football player wearing his jersey number   <END>', '<START> a sooners football player weas the number and black armbands   <END>', '<START> guy in red and white football uniform <END>', '<START> the american footballer is wearing a red and white strip   <END>']
    ['<START> a brown dog running <END>', '<START> a brown dog running over grass   <END>', '<START> a brown dog with its front paws off the ground on a grassy surface near red and purple flowers   <END>', '<START> a dog runs across a grassy lawn near some flowers   <END>', '<START> a yellow dog is playing in a grassy area near flowers   <END>']
    ['<START> a girl with dark brown hair and eyes in a blue scarf is standing next to a girl in a fur edged coat   <END>', '<START> an asian boy and an asian girl are smiling in a crowd of people   <END>', '<START> the girls were in the crowd   <END>', '<START> two dark haired girls are in a crowd   <END>', '<START> two girls are looking past each other in different directions while standing in a crowd   <END>']
    ['<START> a dog with its mouth opened   <END>', '<START> brown and white dog yawning   <END>', '<START> close-up of dog in profile with mouth open   <END>', '<START> dog yawns <END>', "<START> the dog 's mouth is open like he is yawning   <END>"]
    ['<START> a black dog emerges from the water onto the sand   holding a white object in its mouth   <END>', '<START> a black dog emerges from the water with a white ball in its mouth   <END>', '<START> a black dog on a beach carrying a ball in its mouth   <END>', '<START> a black dog walking out of the water with a white ball in his mouth   <END>', '<START> the black dog jumps out of the water with something in its mouth   <END>']
    ['<START> a player from the white and green highschool team dribbles down court defended by a player from the other team   <END>', '<START> four basketball players in action   <END>', '<START> four men playing basketball   two from each team   <END>', '<START> two boys in green and white uniforms play basketball with two boys in blue and white uniforms   <END>', '<START> young men playing basketball in a competition   <END>']

    bleu_score:{'1-garm': 0.6364854466640457, '2-garm': 0.426242182747637}


#### 实验 5 使用 VGG-19, batch_size=512

    (1)  采用 soft attention,  解码器的LSTM 和 输出层 中间使用 Dropout(0.5) 连接, 输出层使用 FC 层
         dropout 正则化 dropout_rates = (0.5,)
         
    (2) 数据预处理
    
    过滤掉语料库中的数字和非法字符, 并对所有单词转换为小写
    
    词表大小: n_vocab = 8868
    
    经过 VGG-19 抽取的图片向量维度: (train_seq_length=196, n_image_feature = 512)
    
    编码器长度: train_seq_length=196
    
    图片描述句子的长度: max_caption_length=37
    
    解码器长度: infer_seq_length = 36 
    
    (3) 优化器参数
    
    epoch_num = 15, EarlyStopping('val_loss', patience=10)
    batch_size = 256
    optimizer='rmsprop'
        
    (4) 训练好的模型位于: models/image_caption_attention_model.h5
    
    (5) 训练过程
    
    在每一个 epoch 结束时都对模型进行持久化(checkpoint), 并计算在验证集上的 bleu 得分
    
    
    (6) 模型评价
    
    1.epoch=8 时的模型在验证集上的 bleu 得分最高
    
    在测试集上的结果

    candidates:
    a brown dog is running through the snow <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a dog is jumping into the air <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a man in a blue shirt is sitting in a crowd of a crowd <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a woman in a red shirt is sitting on a bench <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a football player in a red uniform is wearing a red jersey <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a brown dog is running through the grass <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a woman and a woman are posing for a picture <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a brown and white dog is playing with a ball <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a black dog is running through the snow <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a group of men are playing basketball <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    
    references:
    ['<START> the dogs are in the snow in front of a fence   <END>', '<START> the dogs play on the snow   <END>', '<START> two brown dogs playfully fight in the snow   <END>', '<START> two brown dogs wrestle in the snow   <END>', '<START> two dogs playing in the snow   <END>']
    ['<START> a brown and white dog swimming towards some in the pool <END>', '<START> a dog in a swimming pool swims toward sombody we cannot see   <END>', '<START> a dog swims in a pool near a person   <END>', '<START> small dog is paddling through the water in a pool   <END>', '<START> the small brown and white dog is in the pool   <END>']
    ['<START> a man and a woman in festive costumes dancing   <END>', '<START> a man and a woman with feathers on her head dance   <END>', '<START> a man and a woman wearing decorative costumes and dancing in a crowd of onlookers   <END>', '<START> one performer wearing a feathered headdress dancing with another performer in the streets <END>', '<START> two people are dancing with drums on the right and a crowd behind them   <END>']
    ['<START> a couple of people sit outdoors at a table with an umbrella and talk   <END>', '<START> three people are sitting at an outside picnic bench with an umbrella   <END>', '<START> three people sit at an outdoor cafe   <END>', '<START> three people sit at an outdoor table in front of a building painted like the union jack   <END>', '<START> three people sit at a picnic table outside of a building painted like a union jack   <END>']
    ['<START> a man is wearing a sooners red football shirt and helmet   <END>', '<START> a oklahoma sooners football player wearing his jersey number   <END>', '<START> a sooners football player weas the number and black armbands   <END>', '<START> guy in red and white football uniform <END>', '<START> the american footballer is wearing a red and white strip   <END>']
    ['<START> a brown dog running <END>', '<START> a brown dog running over grass   <END>', '<START> a brown dog with its front paws off the ground on a grassy surface near red and purple flowers   <END>', '<START> a dog runs across a grassy lawn near some flowers   <END>', '<START> a yellow dog is playing in a grassy area near flowers   <END>']
    ['<START> a girl with dark brown hair and eyes in a blue scarf is standing next to a girl in a fur edged coat   <END>', '<START> an asian boy and an asian girl are smiling in a crowd of people   <END>', '<START> the girls were in the crowd   <END>', '<START> two dark haired girls are in a crowd   <END>', '<START> two girls are looking past each other in different directions while standing in a crowd   <END>']
    ['<START> a dog with its mouth opened   <END>', '<START> brown and white dog yawning   <END>', '<START> close-up of dog in profile with mouth open   <END>', '<START> dog yawns <END>', "<START> the dog 's mouth is open like he is yawning   <END>"]
    ['<START> a black dog emerges from the water onto the sand   holding a white object in its mouth   <END>', '<START> a black dog emerges from the water with a white ball in its mouth   <END>', '<START> a black dog on a beach carrying a ball in its mouth   <END>', '<START> a black dog walking out of the water with a white ball in his mouth   <END>', '<START> the black dog jumps out of the water with something in its mouth   <END>']
    ['<START> a player from the white and green highschool team dribbles down court defended by a player from the other team   <END>', '<START> four basketball players in action   <END>', '<START> four men playing basketball   two from each team   <END>', '<START> two boys in green and white uniforms play basketball with two boys in blue and white uniforms   <END>', '<START> young men playing basketball in a competition   <END>']
    
    bleu_score:{'1-garm': 0.636908181133452, '2-garm': 0.4254961799467181}


### 2.3 正则化参数 对模型效果的影响

#### 实验 6 使用 inceptionV3,  dropout_rates = (0.8,)

    (1)  采用 soft attention,  解码器的LSTM 和 输出层 中间使用 Dropout(0.5) 连接, 输出层使用 FC 层
         dropout 正则化 dropout_rates = (0.8,)
         
    (2) 数据预处理
    
    过滤掉语料库中的数字和非法字符, 并对所有单词转换为小写
    
    词表大小: n_vocab = 8868
    
    经过 inceptionV3 抽取的图片向量维度: (train_seq_length=64, n_image_feature = 2048)
    
    编码器长度: train_seq_length=64
    
    图片描述句子的长度: max_caption_length=37
    
    解码器长度: infer_seq_length = 36 
    
    (3) 优化器参数
    
    epoch_num = 15, EarlyStopping('val_loss', patience=10)
    batch_size = 256
    optimizer='rmsprop'
        
    (4) 训练好的模型位于: models/image_caption_attention_model.h5
    
    (5) 训练过程
    
    在每一个 epoch 结束时都对模型进行持久化(checkpoint), 并计算在验证集上的 bleu 得分
    
    
    (6) 模型评价
    
    1.epoch=8 时的模型在验证集上的 bleu 得分最高
    
    bleu_score:{'1-garm': 0.6326867102457525, '2-garm': 0.4090306553547222}
    
    在测试集上的结果
    
    candidates:
    a brown dog is running through the snow <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a boy is jumping into a pool <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a man in a blue shirt is standing in front of a building <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a man is sitting on a bench <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a football player in a red uniform is running on the field <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a dog is running through a field <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a woman and a woman are standing in front of a crowd <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a dog is running in the air <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a black dog is running through the water <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    a basketball player is playing basketball <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END> <END>
    
    references:
    ['<START> the dogs are in the snow in front of a fence   <END>', '<START> the dogs play on the snow   <END>', '<START> two brown dogs playfully fight in the snow   <END>', '<START> two brown dogs wrestle in the snow   <END>', '<START> two dogs playing in the snow   <END>']
    ['<START> a brown and white dog swimming towards some in the pool <END>', '<START> a dog in a swimming pool swims toward sombody we cannot see   <END>', '<START> a dog swims in a pool near a person   <END>', '<START> small dog is paddling through the water in a pool   <END>', '<START> the small brown and white dog is in the pool   <END>']
    ['<START> a man and a woman in festive costumes dancing   <END>', '<START> a man and a woman with feathers on her head dance   <END>', '<START> a man and a woman wearing decorative costumes and dancing in a crowd of onlookers   <END>', '<START> one performer wearing a feathered headdress dancing with another performer in the streets <END>', '<START> two people are dancing with drums on the right and a crowd behind them   <END>']
    ['<START> a couple of people sit outdoors at a table with an umbrella and talk   <END>', '<START> three people are sitting at an outside picnic bench with an umbrella   <END>', '<START> three people sit at an outdoor cafe   <END>', '<START> three people sit at an outdoor table in front of a building painted like the union jack   <END>', '<START> three people sit at a picnic table outside of a building painted like a union jack   <END>']
    ['<START> a man is wearing a sooners red football shirt and helmet   <END>', '<START> a oklahoma sooners football player wearing his jersey number   <END>', '<START> a sooners football player weas the number and black armbands   <END>', '<START> guy in red and white football uniform <END>', '<START> the american footballer is wearing a red and white strip   <END>']
    ['<START> a brown dog running <END>', '<START> a brown dog running over grass   <END>', '<START> a brown dog with its front paws off the ground on a grassy surface near red and purple flowers   <END>', '<START> a dog runs across a grassy lawn near some flowers   <END>', '<START> a yellow dog is playing in a grassy area near flowers   <END>']
    ['<START> a girl with dark brown hair and eyes in a blue scarf is standing next to a girl in a fur edged coat   <END>', '<START> an asian boy and an asian girl are smiling in a crowd of people   <END>', '<START> the girls were in the crowd   <END>', '<START> two dark haired girls are in a crowd   <END>', '<START> two girls are looking past each other in different directions while standing in a crowd   <END>']
    ['<START> a dog with its mouth opened   <END>', '<START> brown and white dog yawning   <END>', '<START> close-up of dog in profile with mouth open   <END>', '<START> dog yawns <END>', "<START> the dog 's mouth is open like he is yawning   <END>"]
    ['<START> a black dog emerges from the water onto the sand   holding a white object in its mouth   <END>', '<START> a black dog emerges from the water with a white ball in its mouth   <END>', '<START> a black dog on a beach carrying a ball in its mouth   <END>', '<START> a black dog walking out of the water with a white ball in his mouth   <END>', '<START> the black dog jumps out of the water with something in its mouth   <END>']
    ['<START> a player from the white and green highschool team dribbles down court defended by a player from the other team   <END>', '<START> four basketball players in action   <END>', '<START> four men playing basketball   two from each team   <END>', '<START> two boys in green and white uniforms play basketball with two boys in blue and white uniforms   <END>', '<START> young men playing basketball in a competition   <END>']
    
    bleu_score:{'1-garm': 0.6519457443227565, '2-garm': 0.43191868743300027}










