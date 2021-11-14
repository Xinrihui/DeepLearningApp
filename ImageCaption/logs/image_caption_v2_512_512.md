## 1.基本参数

### 1.1 数据集参数
    
    Flicker8k 数据集


### 1.2 模型参数

    词向量维度 n_embedding = 512
    隐藏层的维度 n_h = 512


## 2.实验记录

### 2.1 图片嵌入层的激活函数对模型效果的影响

#### 实验1

    (1) 图片嵌入层使用 relu 激活函数
    
    (2) 采用 1层 LSTM
    
    (3) 数据预处理
    freq_threshold=0 在语料库中出现次数小于0次的 不计入词表
    
    词表大小: n_vocab=9199
    
    经过 CNN 抽取的图片向量维度 n_image_feature = 2048
    
    图片描述句子的长度: max_caption_length=40
    
    (4) 优化器参数
    epoch_num = 50 ( 迭代的次数不够会出现推理的句子都是一样的现象 )
    batch_size = 64
    decay=0.01
    
    (5) 训练好的模型位于: image_caption_ensemble_1lstm_hid_512_emb_512_len_40.h5
    
    (6) 训练过程
    
    Epoch 1/50
    505/505 [==============================] - ETA: 0s - loss: 1.3706 - accuracy: 0.1149
    Epoch 00001: val_loss improved from inf to 1.15378, saving model to models/model.01-1.1538.hdf5
    505/505 [==============================] - 129s 255ms/step - loss: 1.3706 - accuracy: 0.1149 - val_loss: 1.1538 - val_accuracy: 0.1230
    Epoch 2/50
    505/505 [==============================] - ETA: 0s - loss: 1.0553 - accuracy: 0.1291
    Epoch 00002: val_loss improved from 1.15378 to 1.08588, saving model to models/model.02-1.0859.hdf5
    505/505 [==============================] - 128s 253ms/step - loss: 1.0553 - accuracy: 0.1291 - val_loss: 1.0859 - val_accuracy: 0.1297
    Epoch 3/50
    505/505 [==============================] - ETA: 0s - loss: 0.9920 - accuracy: 0.1338
    Epoch 00003: val_loss improved from 1.08588 to 1.06401, saving model to models/model.03-1.0640.hdf5
    505/505 [==============================] - 127s 252ms/step - loss: 0.9920 - accuracy: 0.1338 - val_loss: 1.0640 - val_accuracy: 0.1337
    Epoch 4/50
    505/505 [==============================] - ETA: 0s - loss: 0.9490 - accuracy: 0.1384
    Epoch 00004: val_loss improved from 1.06401 to 1.03515, saving model to models/model.04-1.0351.hdf5
    505/505 [==============================] - 128s 253ms/step - loss: 0.9490 - accuracy: 0.1384 - val_loss: 1.0351 - val_accuracy: 0.1345
    Epoch 5/50
    505/505 [==============================] - ETA: 0s - loss: 0.9211 - accuracy: 0.1400
    Epoch 00005: val_loss improved from 1.03515 to 1.02830, saving model to models/model.05-1.0283.hdf5
    505/505 [==============================] - 129s 256ms/step - loss: 0.9211 - accuracy: 0.1400 - val_loss: 1.0283 - val_accuracy: 0.1365
    Epoch 6/50
    505/505 [==============================] - ETA: 0s - loss: 0.9046 - accuracy: 0.1426
    Epoch 00006: val_loss improved from 1.02830 to 1.02343, saving model to models/model.06-1.0234.hdf5
    505/505 [==============================] - 130s 258ms/step - loss: 0.9046 - accuracy: 0.1426 - val_loss: 1.0234 - val_accuracy: 0.1370
    Epoch 7/50
    505/505 [==============================] - ETA: 0s - loss: 0.8853 - accuracy: 0.1434
    Epoch 00007: val_loss improved from 1.02343 to 1.01753, saving model to models/model.07-1.0175.hdf5
    505/505 [==============================] - 136s 270ms/step - loss: 0.8853 - accuracy: 0.1434 - val_loss: 1.0175 - val_accuracy: 0.1396
    Epoch 8/50
    505/505 [==============================] - ETA: 0s - loss: 0.8741 - accuracy: 0.1444
    Epoch 00008: val_loss improved from 1.01753 to 1.00052, saving model to models/model.08-1.0005.hdf5
    505/505 [==============================] - 131s 259ms/step - loss: 0.8741 - accuracy: 0.1444 - val_loss: 1.0005 - val_accuracy: 0.1385
    Epoch 9/50
    505/505 [==============================] - ETA: 0s - loss: 0.8640 - accuracy: 0.1453
    Epoch 00009: val_loss improved from 1.00052 to 0.99704, saving model to models/model.09-0.9970.hdf5
    505/505 [==============================] - 128s 253ms/step - loss: 0.8640 - accuracy: 0.1453 - val_loss: 0.9970 - val_accuracy: 0.1394
    Epoch 10/50
    505/505 [==============================] - ETA: 0s - loss: 0.8564 - accuracy: 0.1467
    Epoch 00010: val_loss did not improve from 0.99704
    505/505 [==============================] - 128s 253ms/step - loss: 0.8564 - accuracy: 0.1467 - val_loss: 0.9989 - val_accuracy: 0.1402
    ...
    Epoch 49/50
    505/505 [==============================] - ETA: 0s - loss: 0.7350 - accuracy: 0.1611
    Epoch 00049: val_loss did not improve from 0.96107
    505/505 [==============================] - 128s 253ms/step - loss: 0.7350 - accuracy: 0.1611 - val_loss: 0.9721 - val_accuracy: 0.1438
    Epoch 50/50
    505/505 [==============================] - ETA: 0s - loss: 0.7338 - accuracy: 0.1615
    Epoch 00050: val_loss improved from 0.96107 to 0.95781, saving model to models/model.50-0.9578.hdf5
    505/505 [==============================] - 127s 251ms/step - loss: 0.7338 - accuracy: 0.1615 - val_loss: 0.9578 - val_accuracy: 0.1446
    
    可以看出 val_loss 也随着 train_loss 在不断下降
    
    (7) 模型评价
    candidates:
    A man in a black shirt and a black hat is standing in front of a brick wall <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
    A black dog is running through a field of grass <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
    A little girl in a pink shirt is playing with a toy <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
    A man in a red shirt and a woman in a black shirt and a man in a white shirt and a man in a white shirt and a man in a white shirt and a black shirt and
    A man in a red shirt is standing on a rock in front of a waterfall <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
    A man in a red shirt and black shorts is playing a game <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL>
    A group of people are riding on a roller coaster <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
    A young boy wearing a blue shirt is playing with a toy <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
    A little girl in a pink dress is playing with a toy <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
    A man in a red shirt and black shorts is jumping on a skateboard <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
    
    references:
    ['<START> A boy holds a red bucket up to a pony   <END>', '<START> A man feeding a horse wearing a blue strapped blanket   <END>', '<START> A man holds feed for a horse in a blue blanket   <END>', '<START> A young boy is feeding a horse some hay from an orange bucket   <END>', '<START> Grey horse wearing blue cover eating from a orange bucket held by a person in a green shirt   <END>']
    ['<START> A dog bites a cat whiel they lay on a bed together   <END>', '<START> A white dog biting an orange cat on the bed   pillows behind   <END>', '<START> A white dog biting at an orange cat   <END>', '<START> A white dog in a red collar biting an orange tabby cat on a bed   <END>', '<START> White dog biting orange cat   <END>']
    ['<START> Children with painted red faces being sprayed with water   on grass   <END>', '<START> The two children are being sprayed by water <END>', '<START> Two children playing in a spray of water   <END>', '<START> Two children standing in the grass being sprayed by a hose   <END>', '<START> Two indian children are being squirted by a jet of water   <END>']
    ['<START> The middle eastern woman wearing the pink headscarf is walking beside a woman in a purple headscarf   <END>', '<START> Two Muslim woman wearing their head scarves and frowning faces   <END>', '<START> Two women are wearing lavender scarves an their heads   <END>', '<START> Two women dressed with scarves over their heads look angrily at the photographer   <END>', '<START> Two women in head wraps   <END>']
    ['<START> A girl in a red jacket   surrounded by people   <END>', '<START> A woman in a puffy red jacket poses for a picture at an ice skating rink   <END>', '<START> A woman in a red coat is smiling   while people in the background are walking around in winter clothing   <END>', '<START> A woman wearing a red coat smiles down at the camera   <END>', '<START> The woman in a red jacket is smiling at the camera   <END>']
    ['<START> a wrestler throws another wrestler to the ground   <END>', '<START> Two men engage in a professional wrestling match   <END>', '<START> two men in a fight in a ring <END>', '<START> Two men in midair fighting in a professional wrestling ring   <END>', '<START> Two wrestlers jump in a ring while an official watches   <END>']
    ['<START> A group of hockey players slide along the ice during a game   <END>', '<START> A group of men playing hockey   <END>', '<START> A hockey game is being played   <END>', '<START> A hockey goalie lays on the ice and other players skate past him   <END>', '<START> two hockey player teams playing a game on the ice <END>']
    ['<START> A girl in a purple tutu dances in the yard   <END>', '<START> A little girl in a puffy purple skirt dances in a park   <END>', '<START> A young girl twirls her fluffy purple skirt   <END>', '<START> Little girl is spinning around on the grass in a flowing purple skirt   <END>', '<START> The little girl has a purple dress on   <END>']
    ['<START> A baby and a toddler are smiling whilst playing in a nursery   <END>', '<START> A baby in a bouncy seat and a standing boy surrounded by toys   <END>', '<START> A baby in a walker and an older child nearby   <END>', '<START> A young boy and his baby brother get excited about picture taking   <END>', '<START> Two little boys are smiling and laughing while one is standing and one is in a bouncy seat   <END>']
    ['<START> A bicyclist does tricks on a lime green bike   <END>', '<START> A man dressed in black rides a green bike   <END>', '<START> A man performs a stoppie trick on his BMX green bicycle with no hands   <END>', '<START> Man on green bicycle performing a trick on one wheel   <END>', '<START> The man is doing a trick on his green bike   <END>']
    
    bleu_score:{'1-garm': 0.5336553991036749, '2-garm': 0.3116329882120794}

#### 实验2

    (1) 图片嵌入层使用 sigmoid 激活函数
    
    (2) 采用 1层 LSTM
    
    (3) 数据预处理
    freq_threshold=0 在语料库中出现次数小于0次的 不计入词表
    
    词表大小: n_vocab=9199
    
    经过 CNN 抽取的图片向量维度 n_image_feature = 2048
    
    图片描述句子的长度: max_caption_length=40
    
    (4) 优化器参数
    epoch_num = 50 ( 迭代的次数不够会出现推理的句子都是一样的现象 )
    batch_size = 64
    decay=0.01
    
    (5) 训练好的模型位于:
    
    (6) 模型评价


### 2.2 将图片向量降维后对模型效果的影响

#### 实验3

    (1) 图片嵌入层使用 relu 激活函数
    
    (2) 采用1层 LSTM
    
    (3) 数据预处理
    freq_threshold=0 在语料库中出现次数小于0次的 不计入词表
    
    词表大小: n_vocab=9199
    
    图片经过 CNN 抽取后再经过 PCA 降维, 最终图片向量维度 n_image_feature = 512
    
    图片描述句子的长度: max_caption_length=40
    
    
    (4) 优化器参数
    epoch_num = 50
    batch_size = 64
    decay=0.01
    
    (5) 训练好的模型位于: models/image_caption_ensemble_1lstm_hid_512_emb_512_thres_0_len_40_pca.h5
    
    (6) 训练过程
    
    训练集上的损失不断下降, 但是验证集上的损失在很早的 epcho 时就达到了最低, 说明模型早早过拟合了
    
    (7) 模型评价
    
    epcho=50 时模型的效果
    
    candidates:
    Children are playing in a pool <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
    A boy is climbing a large rock wall <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
    A young boy in a black shirt and black shorts is running in the water <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL>
    A man in a blue shirt is sitting in a shopping cart <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
    A man in a white shirt is playing with a dog <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL>
    A person is running with a ball in front of a crowd <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
    A young girl is playing a game of a group of people <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
    A man in a red shirt is riding a bike on a motorcycle <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL>
    A man with a white shirt and a backpack is standing on a sidewalk <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
    A little boy is sitting on a red and white bench <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL>
    
    references:
    ['<START> A boy holds a red bucket up to a pony   <END>', '<START> A man feeding a horse wearing a blue strapped blanket   <END>', '<START> A man holds feed for a horse in a blue blanket   <END>', '<START> A young boy is feeding a horse some hay from an orange bucket   <END>', '<START> Grey horse wearing blue cover eating from a orange bucket held by a person in a green shirt   <END>']
    ['<START> A dog bites a cat whiel they lay on a bed together   <END>', '<START> A white dog biting an orange cat on the bed   pillows behind   <END>', '<START> A white dog biting at an orange cat   <END>', '<START> A white dog in a red collar biting an orange tabby cat on a bed   <END>', '<START> White dog biting orange cat   <END>']
    ['<START> Children with painted red faces being sprayed with water   on grass   <END>', '<START> The two children are being sprayed by water <END>', '<START> Two children playing in a spray of water   <END>', '<START> Two children standing in the grass being sprayed by a hose   <END>', '<START> Two indian children are being squirted by a jet of water   <END>']
    ['<START> The middle eastern woman wearing the pink headscarf is walking beside a woman in a purple headscarf   <END>', '<START> Two Muslim woman wearing their head scarves and frowning faces   <END>', '<START> Two women are wearing lavender scarves an their heads   <END>', '<START> Two women dressed with scarves over their heads look angrily at the photographer   <END>', '<START> Two women in head wraps   <END>']
    ['<START> A girl in a red jacket   surrounded by people   <END>', '<START> A woman in a puffy red jacket poses for a picture at an ice skating rink   <END>', '<START> A woman in a red coat is smiling   while people in the background are walking around in winter clothing   <END>', '<START> A woman wearing a red coat smiles down at the camera   <END>', '<START> The woman in a red jacket is smiling at the camera   <END>']
    ['<START> a wrestler throws another wrestler to the ground   <END>', '<START> Two men engage in a professional wrestling match   <END>', '<START> two men in a fight in a ring <END>', '<START> Two men in midair fighting in a professional wrestling ring   <END>', '<START> Two wrestlers jump in a ring while an official watches   <END>']
    ['<START> A group of hockey players slide along the ice during a game   <END>', '<START> A group of men playing hockey   <END>', '<START> A hockey game is being played   <END>', '<START> A hockey goalie lays on the ice and other players skate past him   <END>', '<START> two hockey player teams playing a game on the ice <END>']
    ['<START> A girl in a purple tutu dances in the yard   <END>', '<START> A little girl in a puffy purple skirt dances in a park   <END>', '<START> A young girl twirls her fluffy purple skirt   <END>', '<START> Little girl is spinning around on the grass in a flowing purple skirt   <END>', '<START> The little girl has a purple dress on   <END>']
    ['<START> A baby and a toddler are smiling whilst playing in a nursery   <END>', '<START> A baby in a bouncy seat and a standing boy surrounded by toys   <END>', '<START> A baby in a walker and an older child nearby   <END>', '<START> A young boy and his baby brother get excited about picture taking   <END>', '<START> Two little boys are smiling and laughing while one is standing and one is in a bouncy seat   <END>']
    ['<START> A bicyclist does tricks on a lime green bike   <END>', '<START> A man dressed in black rides a green bike   <END>', '<START> A man performs a stoppie trick on his BMX green bicycle with no hands   <END>', '<START> Man on green bicycle performing a trick on one wheel   <END>', '<START> The man is doing a trick on his green bike   <END>']
    
    bleu_score:{'1-garm': 0.3958408122192601, '2-garm': 0.15163522631560875}


### 2.3 使用 集成模型 对模型效果的影响

#### 实验4

    (1) 图片嵌入层使用 relu 激活函数
    
    (2) 采用 2层 LSTM, 每一层LSTM 的初始隐状态h0 都为图片向量, 中间使用 Dropout(0.5) 连接
    
    (3) 数据预处理
    freq_threshold=0 在语料库中出现次数小于0次的 不计入词表
    
    词表大小: n_vocab=9199
    
    经过 CNN 抽取的图片向量维度 n_image_feature = 2048
    
    图片描述句子的长度: max_caption_length=40
    
    
    (4) 优化器参数
    
    epoch_num = 500, EarlyStopping('val_loss', patience=20)
    batch_size = 64
    decay=0.01
    
    (5) 训练好的模型位于: models/image_caption_ensemble_2lstm_hid_512_emb_512_thres_0_len_40.h5
    
    (6) 训练过程
    
    验证集上的损失基本一直在下降, 训练在 epoch=127 时停止
    
    Epoch 127/500
    505/505 [==============================] - ETA: 0s - loss: 1.2164 - accuracy: 0.1389
    Epoch 00127: val_loss did not improve from 1.26718
    505/505 [==============================] - 139s 274ms/step - loss: 1.2164 - accuracy: 0.1389 - val_loss: 1.2817 - val_accuracy: 0.1345
    
    最低验证集损失出现在 epoch=107
    
    Epoch 107/500
    505/505 [==============================] - ETA: 0s - loss: 1.2218 - accuracy: 0.1378
    Epoch 00107: val_loss improved from 1.26969 to 1.26718, saving model to models/cache/model.107-1.2672.hdf5
    505/505 [==============================] - 143s 283ms/step - loss: 1.2218 - accuracy: 0.1378 - val_loss: 1.2672 - val_accuracy: 0.1346
    
    (7) 模型评价
    
    1.epoch=127 时的模型
    
    candidates:
    A man in a black shirt and a black shirt is sitting on a bench <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL>
    A dog is running through the grass <END> <NULL> <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END>
    A boy in a red shirt is playing with a ball <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL>
    A man in a white shirt and a woman in a black shirt and a black shirt and a woman in a red shirt <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL>
    A man in a black shirt and a black shirt is sitting on a bench <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL>
    A man in a white shirt and a black shirt is standing on a bench <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL>
    A group of people are playing in a field <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END>
    A boy in a red shirt is jumping down a swing <END> <NULL> <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL>
    A young girl in a blue shirt is standing on a bench <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL>
    A man in a red shirt is riding a bike on a ramp <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL>
    
    references:
    ['<START> A boy holds a red bucket up to a pony   <END>', '<START> A man feeding a horse wearing a blue strapped blanket   <END>', '<START> A man holds feed for a horse in a blue blanket   <END>', '<START> A young boy is feeding a horse some hay from an orange bucket   <END>', '<START> Grey horse wearing blue cover eating from a orange bucket held by a person in a green shirt   <END>']
    ['<START> A dog bites a cat whiel they lay on a bed together   <END>', '<START> A white dog biting an orange cat on the bed   pillows behind   <END>', '<START> A white dog biting at an orange cat   <END>', '<START> A white dog in a red collar biting an orange tabby cat on a bed   <END>', '<START> White dog biting orange cat   <END>']
    ['<START> Children with painted red faces being sprayed with water   on grass   <END>', '<START> The two children are being sprayed by water <END>', '<START> Two children playing in a spray of water   <END>', '<START> Two children standing in the grass being sprayed by a hose   <END>', '<START> Two indian children are being squirted by a jet of water   <END>']
    ['<START> The middle eastern woman wearing the pink headscarf is walking beside a woman in a purple headscarf   <END>', '<START> Two Muslim woman wearing their head scarves and frowning faces   <END>', '<START> Two women are wearing lavender scarves an their heads   <END>', '<START> Two women dressed with scarves over their heads look angrily at the photographer   <END>', '<START> Two women in head wraps   <END>']
    ['<START> A girl in a red jacket   surrounded by people   <END>', '<START> A woman in a puffy red jacket poses for a picture at an ice skating rink   <END>', '<START> A woman in a red coat is smiling   while people in the background are walking around in winter clothing   <END>', '<START> A woman wearing a red coat smiles down at the camera   <END>', '<START> The woman in a red jacket is smiling at the camera   <END>']
    ['<START> a wrestler throws another wrestler to the ground   <END>', '<START> Two men engage in a professional wrestling match   <END>', '<START> two men in a fight in a ring <END>', '<START> Two men in midair fighting in a professional wrestling ring   <END>', '<START> Two wrestlers jump in a ring while an official watches   <END>']
    ['<START> A group of hockey players slide along the ice during a game   <END>', '<START> A group of men playing hockey   <END>', '<START> A hockey game is being played   <END>', '<START> A hockey goalie lays on the ice and other players skate past him   <END>', '<START> two hockey player teams playing a game on the ice <END>']
    ['<START> A girl in a purple tutu dances in the yard   <END>', '<START> A little girl in a puffy purple skirt dances in a park   <END>', '<START> A young girl twirls her fluffy purple skirt   <END>', '<START> Little girl is spinning around on the grass in a flowing purple skirt   <END>', '<START> The little girl has a purple dress on   <END>']
    ['<START> A baby and a toddler are smiling whilst playing in a nursery   <END>', '<START> A baby in a bouncy seat and a standing boy surrounded by toys   <END>', '<START> A baby in a walker and an older child nearby   <END>', '<START> A young boy and his baby brother get excited about picture taking   <END>', '<START> Two little boys are smiling and laughing while one is standing and one is in a bouncy seat   <END>']
    ['<START> A bicyclist does tricks on a lime green bike   <END>', '<START> A man dressed in black rides a green bike   <END>', '<START> A man performs a stoppie trick on his BMX green bicycle with no hands   <END>', '<START> Man on green bicycle performing a trick on one wheel   <END>', '<START> The man is doing a trick on his green bike   <END>']
    
    bleu_score:{'1-garm': 0.5149163391046844, '2-garm': 0.3030671107074098}
    
    
    2.epoch=107 时的模型
    
    candidates:
    A man in a black shirt is sitting on a bench <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END>
    A dog is running through the grass <END> <NULL> <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END>
    A boy in a red shirt is playing in the water <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL>
    A man in a white shirt and a woman in a black shirt and a black shirt is standing in front of a building <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <END> <NULL>
    A man in a black shirt and a black shirt is sitting on a bench <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END>
    A man in a white shirt and a black shirt is standing on a bench <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL>
    A group of people are playing in a field <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END>
    A boy in a blue shirt is jumping down a swing <END> <NULL> <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL>
    A boy in a blue shirt is standing on a swing <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END>
    A man in a red shirt is riding a bike on a ramp <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL>
    references:
    ['<START> A boy holds a red bucket up to a pony   <END>', '<START> A man feeding a horse wearing a blue strapped blanket   <END>', '<START> A man holds feed for a horse in a blue blanket   <END>', '<START> A young boy is feeding a horse some hay from an orange bucket   <END>', '<START> Grey horse wearing blue cover eating from a orange bucket held by a person in a green shirt   <END>']
    ['<START> A dog bites a cat whiel they lay on a bed together   <END>', '<START> A white dog biting an orange cat on the bed   pillows behind   <END>', '<START> A white dog biting at an orange cat   <END>', '<START> A white dog in a red collar biting an orange tabby cat on a bed   <END>', '<START> White dog biting orange cat   <END>']
    ['<START> Children with painted red faces being sprayed with water   on grass   <END>', '<START> The two children are being sprayed by water <END>', '<START> Two children playing in a spray of water   <END>', '<START> Two children standing in the grass being sprayed by a hose   <END>', '<START> Two indian children are being squirted by a jet of water   <END>']
    ['<START> The middle eastern woman wearing the pink headscarf is walking beside a woman in a purple headscarf   <END>', '<START> Two Muslim woman wearing their head scarves and frowning faces   <END>', '<START> Two women are wearing lavender scarves an their heads   <END>', '<START> Two women dressed with scarves over their heads look angrily at the photographer   <END>', '<START> Two women in head wraps   <END>']
    ['<START> A girl in a red jacket   surrounded by people   <END>', '<START> A woman in a puffy red jacket poses for a picture at an ice skating rink   <END>', '<START> A woman in a red coat is smiling   while people in the background are walking around in winter clothing   <END>', '<START> A woman wearing a red coat smiles down at the camera   <END>', '<START> The woman in a red jacket is smiling at the camera   <END>']
    ['<START> a wrestler throws another wrestler to the ground   <END>', '<START> Two men engage in a professional wrestling match   <END>', '<START> two men in a fight in a ring <END>', '<START> Two men in midair fighting in a professional wrestling ring   <END>', '<START> Two wrestlers jump in a ring while an official watches   <END>']
    ['<START> A group of hockey players slide along the ice during a game   <END>', '<START> A group of men playing hockey   <END>', '<START> A hockey game is being played   <END>', '<START> A hockey goalie lays on the ice and other players skate past him   <END>', '<START> two hockey player teams playing a game on the ice <END>']
    ['<START> A girl in a purple tutu dances in the yard   <END>', '<START> A little girl in a puffy purple skirt dances in a park   <END>', '<START> A young girl twirls her fluffy purple skirt   <END>', '<START> Little girl is spinning around on the grass in a flowing purple skirt   <END>', '<START> The little girl has a purple dress on   <END>']
    ['<START> A baby and a toddler are smiling whilst playing in a nursery   <END>', '<START> A baby in a bouncy seat and a standing boy surrounded by toys   <END>', '<START> A baby in a walker and an older child nearby   <END>', '<START> A young boy and his baby brother get excited about picture taking   <END>', '<START> Two little boys are smiling and laughing while one is standing and one is in a bouncy seat   <END>']
    ['<START> A bicyclist does tricks on a lime green bike   <END>', '<START> A man dressed in black rides a green bike   <END>', '<START> A man performs a stoppie trick on his BMX green bicycle with no hands   <END>', '<START> Man on green bicycle performing a trick on one wheel   <END>', '<START> The man is doing a trick on his green bike   <END>']
    
    
    bleu_score:{'1-garm': 0.5022415999429367, '2-garm': 0.2912250491276374}
    
    
    3.epoch=53 时的模型
    
    candidates:
    A man in a red shirt is sitting on a bench <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <NULL> <END> <NULL> <NULL> <NULL> <END> <NULL> <NULL>
    A dog is running through the grass <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL>
    A man in a red shirt is playing in the water <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL>
    A man in a red shirt and a black shirt is standing on a bench <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <NULL> <END> <NULL> <NULL>
    A man in a red shirt is standing on a bench <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <NULL> <END> <NULL> <NULL> <NULL> <END> <NULL> <NULL> <NULL> <END> <NULL>
    A man in a red shirt and a black shirt is standing on a bench <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <NULL> <END> <NULL> <NULL> <NULL> <END> <NULL>
    A man in a red shirt is standing on a beach <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <NULL> <END> <NULL> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <NULL> <END> <NULL> <NULL> <NULL> <END> <NULL> <NULL> <NULL>
    A man in a red shirt is standing on a bench <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <NULL> <END> <NULL> <NULL> <NULL> <END> <NULL> <NULL> <NULL> <END> <NULL> <NULL> <NULL> <END> <NULL> <NULL> <NULL>
    A man in a red shirt is standing on a bench <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <NULL> <END> <NULL> <NULL> <NULL> <END> <NULL> <NULL> <NULL> <END> <NULL>
    A man in a red shirt is riding a bike on a rock <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL>
    references:
    ['<START> A boy holds a red bucket up to a pony   <END>', '<START> A man feeding a horse wearing a blue strapped blanket   <END>', '<START> A man holds feed for a horse in a blue blanket   <END>', '<START> A young boy is feeding a horse some hay from an orange bucket   <END>', '<START> Grey horse wearing blue cover eating from a orange bucket held by a person in a green shirt   <END>']
    ['<START> A dog bites a cat whiel they lay on a bed together   <END>', '<START> A white dog biting an orange cat on the bed   pillows behind   <END>', '<START> A white dog biting at an orange cat   <END>', '<START> A white dog in a red collar biting an orange tabby cat on a bed   <END>', '<START> White dog biting orange cat   <END>']
    ['<START> Children with painted red faces being sprayed with water   on grass   <END>', '<START> The two children are being sprayed by water <END>', '<START> Two children playing in a spray of water   <END>', '<START> Two children standing in the grass being sprayed by a hose   <END>', '<START> Two indian children are being squirted by a jet of water   <END>']
    ['<START> The middle eastern woman wearing the pink headscarf is walking beside a woman in a purple headscarf   <END>', '<START> Two Muslim woman wearing their head scarves and frowning faces   <END>', '<START> Two women are wearing lavender scarves an their heads   <END>', '<START> Two women dressed with scarves over their heads look angrily at the photographer   <END>', '<START> Two women in head wraps   <END>']
    ['<START> A girl in a red jacket   surrounded by people   <END>', '<START> A woman in a puffy red jacket poses for a picture at an ice skating rink   <END>', '<START> A woman in a red coat is smiling   while people in the background are walking around in winter clothing   <END>', '<START> A woman wearing a red coat smiles down at the camera   <END>', '<START> The woman in a red jacket is smiling at the camera   <END>']
    ['<START> a wrestler throws another wrestler to the ground   <END>', '<START> Two men engage in a professional wrestling match   <END>', '<START> two men in a fight in a ring <END>', '<START> Two men in midair fighting in a professional wrestling ring   <END>', '<START> Two wrestlers jump in a ring while an official watches   <END>']
    ['<START> A group of hockey players slide along the ice during a game   <END>', '<START> A group of men playing hockey   <END>', '<START> A hockey game is being played   <END>', '<START> A hockey goalie lays on the ice and other players skate past him   <END>', '<START> two hockey player teams playing a game on the ice <END>']
    ['<START> A girl in a purple tutu dances in the yard   <END>', '<START> A little girl in a puffy purple skirt dances in a park   <END>', '<START> A young girl twirls her fluffy purple skirt   <END>', '<START> Little girl is spinning around on the grass in a flowing purple skirt   <END>', '<START> The little girl has a purple dress on   <END>']
    ['<START> A baby and a toddler are smiling whilst playing in a nursery   <END>', '<START> A baby in a bouncy seat and a standing boy surrounded by toys   <END>', '<START> A baby in a walker and an older child nearby   <END>', '<START> A young boy and his baby brother get excited about picture taking   <END>', '<START> Two little boys are smiling and laughing while one is standing and one is in a bouncy seat   <END>']
    ['<START> A bicyclist does tricks on a lime green bike   <END>', '<START> A man dressed in black rides a green bike   <END>', '<START> A man performs a stoppie trick on his BMX green bicycle with no hands   <END>', '<START> Man on green bicycle performing a trick on one wheel   <END>', '<START> The man is doing a trick on his green bike   <END>']
    
    bleu_score:{'1-garm': 0.497161617807078, '2-garm': 0.27769731228298195}


#### 实验4-1

    在实验4的预训练模型的基础上继续训练
    
    (1) 优化器参数
    
    epoch_num = 200, EarlyStopping('val_loss', patience=30)
    batch_size = 64
    decay=0.01
    
    (2) 训练过程
    
    验证集上的损失在 epoch=47 时就不再下降了, 训练在 epoch=77 时停止
    
    Epoch 77/200
    505/505 [==============================] - ETA: 0s - loss: 0.7171 - accuracy: 0.1656
    Epoch 00077: val_loss did not improve from 0.95456
    505/505 [==============================] - 142s 282ms/step - loss: 0.7171 - accuracy: 0.1656 - val_loss: 0.9720 - val_accuracy: 0.1445
    
    最低验证集损失出现在 epoch=47
    Epoch 47/200
    505/505 [==============================] - ETA: 0s - loss: 0.7363 - accuracy: 0.1626
    Epoch 00047: val_loss improved from 0.95832 to 0.95456, saving model to models/cache/model.47-0.9546.hdf5
    505/505 [==============================] - 137s 271ms/step - loss: 0.7363 - accuracy: 0.1626 - val_loss: 0.9546 - val_accuracy: 0.1442
    
    
    (3) 模型评价

    1.epoch=77 时的模型
    
    candidates:
    A man in a white shirt and jeans is standing in a crowd of people <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL>
    A white dog is running through a grassy area <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL>
    A young boy is playing in a fountain <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
    A man and woman are standing in front of a brick wall <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
    A woman in a red jacket is standing in front of a waterfall <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL>
    A man in a white shirt and a baseball uniform is being watched by a man in a white shirt <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
    A group of people are riding a raft on a track <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL>
    A woman in a red dress is holding a baby in a blue dress <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
    A little girl in a pink dress is eating a snack on a red carpeted floor <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
    A man in a blue shirt and jeans is riding a skateboard on a street <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL>
    
    references:
    ['<START> A boy holds a red bucket up to a pony   <END>', '<START> A man feeding a horse wearing a blue strapped blanket   <END>', '<START> A man holds feed for a horse in a blue blanket   <END>', '<START> A young boy is feeding a horse some hay from an orange bucket   <END>', '<START> Grey horse wearing blue cover eating from a orange bucket held by a person in a green shirt   <END>']
    ['<START> A dog bites a cat whiel they lay on a bed together   <END>', '<START> A white dog biting an orange cat on the bed   pillows behind   <END>', '<START> A white dog biting at an orange cat   <END>', '<START> A white dog in a red collar biting an orange tabby cat on a bed   <END>', '<START> White dog biting orange cat   <END>']
    ['<START> Children with painted red faces being sprayed with water   on grass   <END>', '<START> The two children are being sprayed by water <END>', '<START> Two children playing in a spray of water   <END>', '<START> Two children standing in the grass being sprayed by a hose   <END>', '<START> Two indian children are being squirted by a jet of water   <END>']
    ['<START> The middle eastern woman wearing the pink headscarf is walking beside a woman in a purple headscarf   <END>', '<START> Two Muslim woman wearing their head scarves and frowning faces   <END>', '<START> Two women are wearing lavender scarves an their heads   <END>', '<START> Two women dressed with scarves over their heads look angrily at the photographer   <END>', '<START> Two women in head wraps   <END>']
    ['<START> A girl in a red jacket   surrounded by people   <END>', '<START> A woman in a puffy red jacket poses for a picture at an ice skating rink   <END>', '<START> A woman in a red coat is smiling   while people in the background are walking around in winter clothing   <END>', '<START> A woman wearing a red coat smiles down at the camera   <END>', '<START> The woman in a red jacket is smiling at the camera   <END>']
    ['<START> a wrestler throws another wrestler to the ground   <END>', '<START> Two men engage in a professional wrestling match   <END>', '<START> two men in a fight in a ring <END>', '<START> Two men in midair fighting in a professional wrestling ring   <END>', '<START> Two wrestlers jump in a ring while an official watches   <END>']
    ['<START> A group of hockey players slide along the ice during a game   <END>', '<START> A group of men playing hockey   <END>', '<START> A hockey game is being played   <END>', '<START> A hockey goalie lays on the ice and other players skate past him   <END>', '<START> two hockey player teams playing a game on the ice <END>']
    ['<START> A girl in a purple tutu dances in the yard   <END>', '<START> A little girl in a puffy purple skirt dances in a park   <END>', '<START> A young girl twirls her fluffy purple skirt   <END>', '<START> Little girl is spinning around on the grass in a flowing purple skirt   <END>', '<START> The little girl has a purple dress on   <END>']
    ['<START> A baby and a toddler are smiling whilst playing in a nursery   <END>', '<START> A baby in a bouncy seat and a standing boy surrounded by toys   <END>', '<START> A baby in a walker and an older child nearby   <END>', '<START> A young boy and his baby brother get excited about picture taking   <END>', '<START> Two little boys are smiling and laughing while one is standing and one is in a bouncy seat   <END>']
    ['<START> A bicyclist does tricks on a lime green bike   <END>', '<START> A man dressed in black rides a green bike   <END>', '<START> A man performs a stoppie trick on his BMX green bicycle with no hands   <END>', '<START> Man on green bicycle performing a trick on one wheel   <END>', '<START> The man is doing a trick on his green bike   <END>']
    
    bleu_score:{'1-garm': 0.5546340744817153, '2-garm': 0.3404716669311104}

    2.epoch=47 时的模型
    
    candidates:
    A man in a black shirt and a woman in a black shirt are standing in front of a white building <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL>
    A small white dog is running through a grassy area <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
    A young boy is playing with a toy in a backyard <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL>
    A man and a woman are standing in front of a brick wall <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL>
    A woman in a red jacket is standing in front of a large waterfall <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
    A man in a white shirt and white shorts is playing a game of other people <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
    A group of people are riding a raft on a track <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL>
    A young girl in a pink dress is holding a camera <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL>
    A little girl in a pink dress is holding a baby in a red dress <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL>
    A man in a blue shirt and jeans is riding a skateboard on a street <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL>
    
    references:
    ['<START> A boy holds a red bucket up to a pony   <END>', '<START> A man feeding a horse wearing a blue strapped blanket   <END>', '<START> A man holds feed for a horse in a blue blanket   <END>', '<START> A young boy is feeding a horse some hay from an orange bucket   <END>', '<START> Grey horse wearing blue cover eating from a orange bucket held by a person in a green shirt   <END>']
    ['<START> A dog bites a cat whiel they lay on a bed together   <END>', '<START> A white dog biting an orange cat on the bed   pillows behind   <END>', '<START> A white dog biting at an orange cat   <END>', '<START> A white dog in a red collar biting an orange tabby cat on a bed   <END>', '<START> White dog biting orange cat   <END>']
    ['<START> Children with painted red faces being sprayed with water   on grass   <END>', '<START> The two children are being sprayed by water <END>', '<START> Two children playing in a spray of water   <END>', '<START> Two children standing in the grass being sprayed by a hose   <END>', '<START> Two indian children are being squirted by a jet of water   <END>']
    ['<START> The middle eastern woman wearing the pink headscarf is walking beside a woman in a purple headscarf   <END>', '<START> Two Muslim woman wearing their head scarves and frowning faces   <END>', '<START> Two women are wearing lavender scarves an their heads   <END>', '<START> Two women dressed with scarves over their heads look angrily at the photographer   <END>', '<START> Two women in head wraps   <END>']
    ['<START> A girl in a red jacket   surrounded by people   <END>', '<START> A woman in a puffy red jacket poses for a picture at an ice skating rink   <END>', '<START> A woman in a red coat is smiling   while people in the background are walking around in winter clothing   <END>', '<START> A woman wearing a red coat smiles down at the camera   <END>', '<START> The woman in a red jacket is smiling at the camera   <END>']
    ['<START> a wrestler throws another wrestler to the ground   <END>', '<START> Two men engage in a professional wrestling match   <END>', '<START> two men in a fight in a ring <END>', '<START> Two men in midair fighting in a professional wrestling ring   <END>', '<START> Two wrestlers jump in a ring while an official watches   <END>']
    ['<START> A group of hockey players slide along the ice during a game   <END>', '<START> A group of men playing hockey   <END>', '<START> A hockey game is being played   <END>', '<START> A hockey goalie lays on the ice and other players skate past him   <END>', '<START> two hockey player teams playing a game on the ice <END>']
    ['<START> A girl in a purple tutu dances in the yard   <END>', '<START> A little girl in a puffy purple skirt dances in a park   <END>', '<START> A young girl twirls her fluffy purple skirt   <END>', '<START> Little girl is spinning around on the grass in a flowing purple skirt   <END>', '<START> The little girl has a purple dress on   <END>']
    ['<START> A baby and a toddler are smiling whilst playing in a nursery   <END>', '<START> A baby in a bouncy seat and a standing boy surrounded by toys   <END>', '<START> A baby in a walker and an older child nearby   <END>', '<START> A young boy and his baby brother get excited about picture taking   <END>', '<START> Two little boys are smiling and laughing while one is standing and one is in a bouncy seat   <END>']
    ['<START> A bicyclist does tricks on a lime green bike   <END>', '<START> A man dressed in black rides a green bike   <END>', '<START> A man performs a stoppie trick on his BMX green bicycle with no hands   <END>', '<START> Man on green bicycle performing a trick on one wheel   <END>', '<START> The man is doing a trick on his green bike   <END>']
    
    bleu_score:{'1-garm': 0.5518695301547729, '2-garm': 0.33546981581599294}



