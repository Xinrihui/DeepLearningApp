## 1.基本参数

词向量维度 n_embedding = 512
隐藏层的维度 n_h = 512

## 2.实验记录

### 2.1 每一层 LSTM 初始隐状态 h0 对模型效果的影响

#### 实验 1 

    (1) 采用 2层 LSTM 堆叠, 每一层LSTM 的初始隐状态 h0 都为图片向量, 中间使用 Dropout(0.5) 连接
    
    (2) 数据预处理
    
    过滤掉语料库中的数字和非法字符, 并对所有单词转换为小写
    
    freq_threshold=0 在语料库中出现次数小于0次的 不计入词表
    
    词表大小: n_vocab=8445
    
    经过 CNN 抽取的图片向量维度 n_image_feature = 2048
    
    图片描述句子的长度: max_caption_length=37
    
    
    (3) 优化器参数
    
    epoch_num = 100, EarlyStopping('val_loss', patience=20)
    batch_size = 128
    optimizer='rmsprop'
        
    (4) 训练好的模型位于: models/image_caption_ensemble_2lstm_hid_512_emb_512_thres_0_len_37.h5
    
    (5) 训练过程
    
    训练在 epoch=31 时停止,  在 epoch=11 的时候达到验证集误差最小
    
    
    (7) 模型评价
    
    1.epoch=11 时的模型
    
      candidates:
        a boy in a red shirt and a woman in a white shirt and blue jeans is riding a horse <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
        a white dog with a red collar is chewing on a white dog <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL>
        a boy in a blue shirt is holding a stick <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
        a woman wearing a red jacket and sunglasses <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
        a woman wearing a red jacket and sunglasses is standing in front of a mountain <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL>
        a man in a red shirt and white shorts is performing a trick in a parade <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
        a hockey player in a red uniform is sliding down the ice <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
        a girl in a pink dress is running through a parking lot <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
        a little boy in a red shirt is holding a red cup <END> <NULL> <END> <NULL> <NULL> <END> <NULL> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
        a man wearing a blue shirt and jeans is riding a unicycle <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>
        
       references:
        ['<START> a boy holds a red bucket up to a pony   <END>', '<START> a man feeding a horse wearing a blue strapped blanket   <END>', '<START> a man holds feed for a horse in a blue blanket   <END>', '<START> a young boy is feeding a horse some hay from an orange bucket   <END>', '<START> grey horse wearing blue cover eating from a orange bucket held by a person in a green shirt   <END>']
        ['<START> a dog bites a cat whiel they lay on a bed together   <END>', '<START> a white dog biting an orange cat on the bed   pillows behind   <END>', '<START> a white dog biting at an orange cat   <END>', '<START> a white dog in a red collar biting an orange tabby cat on a bed   <END>', '<START> white dog biting orange cat   <END>']
        ['<START> children with painted red faces being sprayed with water   on grass   <END>', '<START> the two children are being sprayed by water <END>', '<START> two children playing in a spray of water   <END>', '<START> two children standing in the grass being sprayed by a hose   <END>', '<START> two indian children are being squirted by a jet of water   <END>']
        ['<START> the middle eastern woman wearing the pink headscarf is walking beside a woman in a purple headscarf   <END>', '<START> two muslim woman wearing their head scarves and frowning faces   <END>', '<START> two women are wearing lavender scarves an their heads   <END>', '<START> two women dressed with scarves over their heads look angrily at the photographer   <END>', '<START> two women in head wraps   <END>']
        ['<START> a girl in a red jacket   surrounded by people   <END>', '<START> a woman in a puffy red jacket poses for a picture at an ice skating rink   <END>', '<START> a woman in a red coat is smiling   while people in the background are walking around in winter clothing   <END>', '<START> a woman wearing a red coat smiles down at the camera   <END>', '<START> the woman in a red jacket is smiling at the camera   <END>']
        ['<START> a wrestler throws another wrestler to the ground   <END>', '<START> two men engage in a professional wrestling match   <END>', '<START> two men in a fight in a ring <END>', '<START> two men in midair fighting in a professional wrestling ring   <END>', '<START> two wrestlers jump in a ring while an official watches   <END>']
        ['<START> a group of hockey players slide along the ice during a game   <END>', '<START> a group of men playing hockey   <END>', '<START> a hockey game is being played   <END>', '<START> a hockey goalie lays on the ice and other players skate past him   <END>', '<START> two hockey player teams playing a game on the ice <END>']
        ['<START> a girl in a purple tutu dances in the yard   <END>', '<START> a little girl in a puffy purple skirt dances in a park   <END>', '<START> a young girl twirls her fluffy purple skirt   <END>', '<START> little girl is spinning around on the grass in a flowing purple skirt   <END>', '<START> the little girl has a purple dress on   <END>']
        ['<START> a baby and a toddler are smiling whilst playing in a nursery   <END>', '<START> a baby in a bouncy seat and a standing boy surrounded by toys   <END>', '<START> a baby in a walker and an older child nearby   <END>', '<START> a young boy and his baby brother get excited about picture taking   <END>', '<START> two little boys are smiling and laughing while one is standing and one is in a bouncy seat   <END>']
        ['<START> a bicyclist does tricks on a lime green bike   <END>', '<START> a man dressed in black rides a green bike   <END>', '<START> a man performs a stoppie trick on his bmx green bicycle with no hands   <END>', '<START> man on green bicycle performing a trick on one wheel   <END>', '<START> the man is doing a trick on his green bike   <END>']
        
        bleu_score:{'1-garm': 0.5833985855661277, '2-garm': 0.37858463488248173}

#### 实验 2 

    (1) 采用 2层 LSTM 堆叠, 第1层的 LSTM 的初始隐状态 h0 为图片向量, 其余层的 LSTM 的初始隐状态 h0 为0向量, 
    中间使用 Dropout(0.5) 连接
    
    (2) 数据预处理
    
    过滤掉语料库中的数字和非法字符, 并对所有单词转换为小写
    
    freq_threshold=0 在语料库中出现次数小于0次的 不计入词表
    
    词表大小: n_vocab=8445
    
    经过 CNN 抽取的图片向量维度 n_image_feature = 2048
    
    图片描述句子的长度: max_caption_length=37
    
    
    (3) 优化器参数
    
    epoch_num = 100, EarlyStopping('val_loss', patience=20)
    batch_size = 128
    optimizer='rmsprop'
        
    (4) 训练好的模型位于: models/image_caption_ensemble_2lstm_hid_512_emb_512_thres_0_len_37.h5
    
    (5) 训练过程
    
    训练在 epoch=30 时停止,  在 epoch=10 的时候达到验证集误差最小
    
    
    (7) 模型评价
    
    1.epoch=10 时的模型
    
    candidates:
    a man and woman are sitting in a field of flowers <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a white dog is jumping over a bed <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a boy in a green shirt is playing with a toy <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a woman with a mohawk and a man in a blue shirt <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a man in a hat and sunglasses is standing in front of a mountain <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a man in a green shirt is jumping in the air <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a hockey player is in the air <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a young girl is playing with a toy outside <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a little girl is sitting on a wooden bench <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a skateboarder is performing a trick on a ramp <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    
    references:
    ['<START> a boy holds a red bucket up to a pony   <END>', '<START> a man feeding a horse wearing a blue strapped blanket   <END>', '<START> a man holds feed for a horse in a blue blanket   <END>', '<START> a young boy is feeding a horse some hay from an orange bucket   <END>', '<START> grey horse wearing blue cover eating from a orange bucket held by a person in a green shirt   <END>']
    ['<START> a dog bites a cat whiel they lay on a bed together   <END>', '<START> a white dog biting an orange cat on the bed   pillows behind   <END>', '<START> a white dog biting at an orange cat   <END>', '<START> a white dog in a red collar biting an orange tabby cat on a bed   <END>', '<START> white dog biting orange cat   <END>']
    ['<START> children with painted red faces being sprayed with water   on grass   <END>', '<START> the two children are being sprayed by water <END>', '<START> two children playing in a spray of water   <END>', '<START> two children standing in the grass being sprayed by a hose   <END>', '<START> two indian children are being squirted by a jet of water   <END>']
    ['<START> the middle eastern woman wearing the pink headscarf is walking beside a woman in a purple headscarf   <END>', '<START> two muslim woman wearing their head scarves and frowning faces   <END>', '<START> two women are wearing lavender scarves an their heads   <END>', '<START> two women dressed with scarves over their heads look angrily at the photographer   <END>', '<START> two women in head wraps   <END>']
    ['<START> a girl in a red jacket   surrounded by people   <END>', '<START> a woman in a puffy red jacket poses for a picture at an ice skating rink   <END>', '<START> a woman in a red coat is smiling   while people in the background are walking around in winter clothing   <END>', '<START> a woman wearing a red coat smiles down at the camera   <END>', '<START> the woman in a red jacket is smiling at the camera   <END>']
    ['<START> a wrestler throws another wrestler to the ground   <END>', '<START> two men engage in a professional wrestling match   <END>', '<START> two men in a fight in a ring <END>', '<START> two men in midair fighting in a professional wrestling ring   <END>', '<START> two wrestlers jump in a ring while an official watches   <END>']
    ['<START> a group of hockey players slide along the ice during a game   <END>', '<START> a group of men playing hockey   <END>', '<START> a hockey game is being played   <END>', '<START> a hockey goalie lays on the ice and other players skate past him   <END>', '<START> two hockey player teams playing a game on the ice <END>']
    ['<START> a girl in a purple tutu dances in the yard   <END>', '<START> a little girl in a puffy purple skirt dances in a park   <END>', '<START> a young girl twirls her fluffy purple skirt   <END>', '<START> little girl is spinning around on the grass in a flowing purple skirt   <END>', '<START> the little girl has a purple dress on   <END>']
    ['<START> a baby and a toddler are smiling whilst playing in a nursery   <END>', '<START> a baby in a bouncy seat and a standing boy surrounded by toys   <END>', '<START> a baby in a walker and an older child nearby   <END>', '<START> a young boy and his baby brother get excited about picture taking   <END>', '<START> two little boys are smiling and laughing while one is standing and one is in a bouncy seat   <END>']
    ['<START> a bicyclist does tricks on a lime green bike   <END>', '<START> a man dressed in black rides a green bike   <END>', '<START> a man performs a stoppie trick on his bmx green bicycle with no hands   <END>', '<START> man on green bicycle performing a trick on one wheel   <END>', '<START> the man is doing a trick on his green bike   <END>']
    
    bleu_score:{'1-garm': 0.5811244586229712, '2-garm': 0.3687733408712277}
    
    2.epoch=30 时的模型
    
    candidates:
    a woman in a white dress is standing next to a young girl in a stuffed animal <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a white dog is running down a blue and white dog <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a man in a field throws a stick <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a man and woman wearing sunglasses pose for a picture <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a woman wearing a black coat and sunglasses sits on a rock in front of a mountain <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a man in black shorts and a white top is running in a parade <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    two hockey players are on the ice <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a girl in a pink dress is running through a parking lot <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    two children are playing on a bed <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a man is doing a trick on a skateboard in a city street <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    
    references:
    ['<START> a boy holds a red bucket up to a pony   <END>', '<START> a man feeding a horse wearing a blue strapped blanket   <END>', '<START> a man holds feed for a horse in a blue blanket   <END>', '<START> a young boy is feeding a horse some hay from an orange bucket   <END>', '<START> grey horse wearing blue cover eating from a orange bucket held by a person in a green shirt   <END>']
    ['<START> a dog bites a cat whiel they lay on a bed together   <END>', '<START> a white dog biting an orange cat on the bed   pillows behind   <END>', '<START> a white dog biting at an orange cat   <END>', '<START> a white dog in a red collar biting an orange tabby cat on a bed   <END>', '<START> white dog biting orange cat   <END>']
    ['<START> children with painted red faces being sprayed with water   on grass   <END>', '<START> the two children are being sprayed by water <END>', '<START> two children playing in a spray of water   <END>', '<START> two children standing in the grass being sprayed by a hose   <END>', '<START> two indian children are being squirted by a jet of water   <END>']
    ['<START> the middle eastern woman wearing the pink headscarf is walking beside a woman in a purple headscarf   <END>', '<START> two muslim woman wearing their head scarves and frowning faces   <END>', '<START> two women are wearing lavender scarves an their heads   <END>', '<START> two women dressed with scarves over their heads look angrily at the photographer   <END>', '<START> two women in head wraps   <END>']
    ['<START> a girl in a red jacket   surrounded by people   <END>', '<START> a woman in a puffy red jacket poses for a picture at an ice skating rink   <END>', '<START> a woman in a red coat is smiling   while people in the background are walking around in winter clothing   <END>', '<START> a woman wearing a red coat smiles down at the camera   <END>', '<START> the woman in a red jacket is smiling at the camera   <END>']
    ['<START> a wrestler throws another wrestler to the ground   <END>', '<START> two men engage in a professional wrestling match   <END>', '<START> two men in a fight in a ring <END>', '<START> two men in midair fighting in a professional wrestling ring   <END>', '<START> two wrestlers jump in a ring while an official watches   <END>']
    ['<START> a group of hockey players slide along the ice during a game   <END>', '<START> a group of men playing hockey   <END>', '<START> a hockey game is being played   <END>', '<START> a hockey goalie lays on the ice and other players skate past him   <END>', '<START> two hockey player teams playing a game on the ice <END>']
    ['<START> a girl in a purple tutu dances in the yard   <END>', '<START> a little girl in a puffy purple skirt dances in a park   <END>', '<START> a young girl twirls her fluffy purple skirt   <END>', '<START> little girl is spinning around on the grass in a flowing purple skirt   <END>', '<START> the little girl has a purple dress on   <END>']
    ['<START> a baby and a toddler are smiling whilst playing in a nursery   <END>', '<START> a baby in a bouncy seat and a standing boy surrounded by toys   <END>', '<START> a baby in a walker and an older child nearby   <END>', '<START> a young boy and his baby brother get excited about picture taking   <END>', '<START> two little boys are smiling and laughing while one is standing and one is in a bouncy seat   <END>']
    ['<START> a bicyclist does tricks on a lime green bike   <END>', '<START> a man dressed in black rides a green bike   <END>', '<START> a man performs a stoppie trick on his bmx green bicycle with no hands   <END>', '<START> man on green bicycle performing a trick on one wheel   <END>', '<START> the man is doing a trick on his green bike   <END>']
    
    bleu_score:{'1-garm': 0.5600676154195787, '2-garm': 0.3462389335254294}

### 2.2 采用多层 LSTM堆叠后, 中间 dropout 参数的调优

#### 实验 3 

    (1) 采用 3层 LSTM 堆叠, 每层的 LSTM 的初始隐状态 h0 为图片向量, 中间使用 Dropout 连接, 
    神经元失活参数 rate 从底层到顶层分别为 0.1, 0.2, 0.4
    
    (2) 数据预处理
    
    过滤掉语料库中的数字和非法字符, 并对所有单词转换为小写
    
    freq_threshold=0 在语料库中出现次数小于0次的 不计入词表
    
    词表大小: n_vocab=8445
    
    经过 CNN 抽取的图片向量维度 n_image_feature = 2048
    
    图片描述句子的长度: max_caption_length=37
    
    
    (3) 优化器参数
    
    epoch_num = 100, EarlyStopping('val_loss', patience=20)
    batch_size = 128
    optimizer='rmsprop'
        
    (4) 训练好的模型位于: models/image_caption_ensemble_3lstm_hid_512_emb_512_thres_0_len_37.hdf5
    
    (5) 训练过程
    
    训练在 epoch=31 时停止,  在 epoch=11 的时候达到验证集误差最小
    
    
    (7) 模型评价
        
     1.epoch=11 时的模型
     
    candidates:
    a girl is playing with a large child on a playground <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a white dog is walking through a grassy area <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a child in a blue shirt is playing in a field of water <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a woman with a mohawk and a hat <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a woman in a black jacket and a black jacket is standing in front of a large building <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a man in a white shirt is doing a handstand on a trampoline <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a hockey player is being pulled by a goal <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a girl in a pink dress is dancing in a parade <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a little boy is playing with a toy <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a man is riding a bicycle on a skateboard <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    
    references:
    ['<START> a boy holds a red bucket up to a pony   <END>', '<START> a man feeding a horse wearing a blue strapped blanket   <END>', '<START> a man holds feed for a horse in a blue blanket   <END>', '<START> a young boy is feeding a horse some hay from an orange bucket   <END>', '<START> grey horse wearing blue cover eating from a orange bucket held by a person in a green shirt   <END>']
    ['<START> a dog bites a cat whiel they lay on a bed together   <END>', '<START> a white dog biting an orange cat on the bed   pillows behind   <END>', '<START> a white dog biting at an orange cat   <END>', '<START> a white dog in a red collar biting an orange tabby cat on a bed   <END>', '<START> white dog biting orange cat   <END>']
    ['<START> children with painted red faces being sprayed with water   on grass   <END>', '<START> the two children are being sprayed by water <END>', '<START> two children playing in a spray of water   <END>', '<START> two children standing in the grass being sprayed by a hose   <END>', '<START> two indian children are being squirted by a jet of water   <END>']
    ['<START> the middle eastern woman wearing the pink headscarf is walking beside a woman in a purple headscarf   <END>', '<START> two muslim woman wearing their head scarves and frowning faces   <END>', '<START> two women are wearing lavender scarves an their heads   <END>', '<START> two women dressed with scarves over their heads look angrily at the photographer   <END>', '<START> two women in head wraps   <END>']
    ['<START> a girl in a red jacket   surrounded by people   <END>', '<START> a woman in a puffy red jacket poses for a picture at an ice skating rink   <END>', '<START> a woman in a red coat is smiling   while people in the background are walking around in winter clothing   <END>', '<START> a woman wearing a red coat smiles down at the camera   <END>', '<START> the woman in a red jacket is smiling at the camera   <END>']
    ['<START> a wrestler throws another wrestler to the ground   <END>', '<START> two men engage in a professional wrestling match   <END>', '<START> two men in a fight in a ring <END>', '<START> two men in midair fighting in a professional wrestling ring   <END>', '<START> two wrestlers jump in a ring while an official watches   <END>']
    ['<START> a group of hockey players slide along the ice during a game   <END>', '<START> a group of men playing hockey   <END>', '<START> a hockey game is being played   <END>', '<START> a hockey goalie lays on the ice and other players skate past him   <END>', '<START> two hockey player teams playing a game on the ice <END>']
    ['<START> a girl in a purple tutu dances in the yard   <END>', '<START> a little girl in a puffy purple skirt dances in a park   <END>', '<START> a young girl twirls her fluffy purple skirt   <END>', '<START> little girl is spinning around on the grass in a flowing purple skirt   <END>', '<START> the little girl has a purple dress on   <END>']
    ['<START> a baby and a toddler are smiling whilst playing in a nursery   <END>', '<START> a baby in a bouncy seat and a standing boy surrounded by toys   <END>', '<START> a baby in a walker and an older child nearby   <END>', '<START> a young boy and his baby brother get excited about picture taking   <END>', '<START> two little boys are smiling and laughing while one is standing and one is in a bouncy seat   <END>']
    ['<START> a bicyclist does tricks on a lime green bike   <END>', '<START> a man dressed in black rides a green bike   <END>', '<START> a man performs a stoppie trick on his bmx green bicycle with no hands   <END>', '<START> man on green bicycle performing a trick on one wheel   <END>', '<START> the man is doing a trick on his green bike   <END>']

    bleu_score:{'1-garm': 0.6090063975010919, '2-garm': 0.40223084466464126}
 
 
 #### 实验 4 

    (1) 采用 3层 LSTM 堆叠, 每层的 LSTM 的初始隐状态 h0 为图片向量, 中间使用 Dropout 连接, 
    神经元失活参数 rate 从底层到顶层分别为 0.1, 0.2, 0.4
    
    (2) 数据预处理
    
    过滤掉语料库中的数字和非法字符, 并对所有单词转换为小写
    
    freq_threshold=0 在语料库中出现次数小于 0次的 不计入词表, 
    构建词表时不对标点符号做出过滤, 因为此步骤在语料的预处理阶段做了, 
    对词表加入停用词字典
    
    词表大小: n_vocab=8861
    
    经过 CNN 抽取的图片向量维度 n_image_feature = 2048
    
    图片描述句子的长度: max_caption_length=37
    
    
    (3) 优化器参数
    
    epoch_num = 100, EarlyStopping('val_loss', patience=20)
    batch_size = 128
    optimizer='rmsprop'
        
    (4) 训练好的模型位于: models/image_caption_ensemble_3lstm_hid_512_emb_512_thres_0_len_37.hdf5
    
    (5) 训练过程
    
    训练在 epoch=33 时停止,  在 epoch=13 的时候达到验证集误差最小
    
    
    (7) 模型评价
        
     1.epoch=30 时的模型
 
        candidates:
        a woman and a woman are playing with a small dog <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
        a white dog is running through a grassy area <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
        a child is holding a stick in his mouth <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
        a woman wearing a black hat and a black hat is standing in front of a crowd <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
        a man in a red jacket and sunglasses is standing in front of a snow covered mountain <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
        a man in a white shirt is dancing <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
        a hockey player in a red uniform is skiing with a hockey ball <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
        a girl in a pink dress and pink pants is running <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
        a baby is sitting on a red chair <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
        a man is doing a trick on a bike in a park <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
        
        references:
        ['<START> a boy holds a red bucket up to a pony   <END>', '<START> a man feeding a horse wearing a blue strapped blanket   <END>', '<START> a man holds feed for a horse in a blue blanket   <END>', '<START> a young boy is feeding a horse some hay from an orange bucket   <END>', '<START> grey horse wearing blue cover eating from a orange bucket held by a person in a green shirt   <END>']
        ['<START> a dog bites a cat whiel they lay on a bed together   <END>', '<START> a white dog biting an orange cat on the bed   pillows behind   <END>', '<START> a white dog biting at an orange cat   <END>', '<START> a white dog in a red collar biting an orange tabby cat on a bed   <END>', '<START> white dog biting orange cat   <END>']
        ['<START> children with painted red faces being sprayed with water   on grass   <END>', '<START> the two children are being sprayed by water <END>', '<START> two children playing in a spray of water   <END>', '<START> two children standing in the grass being sprayed by a hose   <END>', '<START> two indian children are being squirted by a jet of water   <END>']
        ['<START> the middle eastern woman wearing the pink headscarf is walking beside a woman in a purple headscarf   <END>', '<START> two muslim woman wearing their head scarves and frowning faces   <END>', '<START> two women are wearing lavender scarves an their heads   <END>', '<START> two women dressed with scarves over their heads look angrily at the photographer   <END>', '<START> two women in head wraps   <END>']
        ['<START> a girl in a red jacket   surrounded by people   <END>', '<START> a woman in a puffy red jacket poses for a picture at an ice skating rink   <END>', '<START> a woman in a red coat is smiling   while people in the background are walking around in winter clothing   <END>', '<START> a woman wearing a red coat smiles down at the camera   <END>', '<START> the woman in a red jacket is smiling at the camera   <END>']
        ['<START> a wrestler throws another wrestler to the ground   <END>', '<START> two men engage in a professional wrestling match   <END>', '<START> two men in a fight in a ring <END>', '<START> two men in midair fighting in a professional wrestling ring   <END>', '<START> two wrestlers jump in a ring while an official watches   <END>']
        ['<START> a group of hockey players slide along the ice during a game   <END>', '<START> a group of men playing hockey   <END>', '<START> a hockey game is being played   <END>', '<START> a hockey goalie lays on the ice and other players skate past him   <END>', '<START> two hockey player teams playing a game on the ice <END>']
        ['<START> a girl in a purple tutu dances in the yard   <END>', '<START> a little girl in a puffy purple skirt dances in a park   <END>', '<START> a young girl twirls her fluffy purple skirt   <END>', '<START> little girl is spinning around on the grass in a flowing purple skirt   <END>', '<START> the little girl has a purple dress on   <END>']
        ['<START> a baby and a toddler are smiling whilst playing in a nursery   <END>', '<START> a baby in a bouncy seat and a standing boy surrounded by toys   <END>', '<START> a baby in a walker and an older child nearby   <END>', '<START> a young boy and his baby brother get excited about picture taking   <END>', '<START> two little boys are smiling and laughing while one is standing and one is in a bouncy seat   <END>']
        ['<START> a bicyclist does tricks on a lime green bike   <END>', '<START> a man dressed in black rides a green bike   <END>', '<START> a man performs a stoppie trick on his bmx green bicycle with no hands   <END>', '<START> man on green bicycle performing a trick on one wheel   <END>', '<START> the man is doing a trick on his green bike   <END>']

        bleu_score:{'1-garm': 0.6112034448070367, '2-garm': 0.40344569250796414}

 #### 实验 5 
 
    (1)使用 Keras Turning  进行参数调优, 3层 dropout , 每一层的可选参数为 [0.1, 0.2, 0.4]
    
    dropout_rate1 = hp.Choice('dropout_rate1',
                  values=[0.1, 0.2, 0.4])
    
    dropout_rate2 = hp.Choice('dropout_rate2',
                  values=[0.1, 0.2, 0.4])
    
    dropout_rate3 = hp.Choice('dropout_rate3',
                  values=[0.1, 0.2, 0.4])
    
    (2) 
    tuner（调参器）: kt.Hyperband
    优化目标:  objective='val_accuracy', 
                
    (3) 最佳超参数
    
    Trial summary
    Hyperparameters:
    dropout_rate1: 0.4
    dropout_rate2: 0.1
    dropout_rate3: 0.4
    tuner/epochs: 15
    tuner/initial_epoch: 0
    tuner/bracket: 0
    tuner/round: 0
    Score: 0.8052317500114441
    
    Trial summary
    Hyperparameters:
    dropout_rate1: 0.4
    dropout_rate2: 0.4
    dropout_rate3: 0.4
    tuner/epochs: 15
    tuner/initial_epoch: 0
    tuner/bracket: 0
    tuner/round: 0
    Score: 0.8051593899726868
    
    Trial summary
    Hyperparameters:
    dropout_rate1: 0.1
    dropout_rate2: 0.1
    dropout_rate3: 0.4
    tuner/epochs: 15
    tuner/initial_epoch: 0
    tuner/bracket: 0
    tuner/round: 0
    Score: 0.8049389123916626
    
    (3) 模型评价
    
    1.dropout_rates = (0.4, 0.1, 0.4)
    
    candidates:
    a man and a woman are sitting on a bench <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    two dogs are playing with a toy <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a boy is playing with a water bottle <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a woman in a black jacket and a man in a black jacket <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a woman in a red jacket and a red jacket is standing in front of a mountain <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a man in a black shirt is standing in a parade <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a hockey player is being tackled by a player in a red uniform <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a girl in a pink dress is playing with a red and white toy <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a baby is sitting on a red chair with a child on her face <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a man in a blue shirt is riding a bicycle on a sidewalk <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    
    references:
    ['<START> a boy holds a red bucket up to a pony   <END>', '<START> a man feeding a horse wearing a blue strapped blanket   <END>', '<START> a man holds feed for a horse in a blue blanket   <END>', '<START> a young boy is feeding a horse some hay from an orange bucket   <END>', '<START> grey horse wearing blue cover eating from a orange bucket held by a person in a green shirt   <END>']
    ['<START> a dog bites a cat whiel they lay on a bed together   <END>', '<START> a white dog biting an orange cat on the bed   pillows behind   <END>', '<START> a white dog biting at an orange cat   <END>', '<START> a white dog in a red collar biting an orange tabby cat on a bed   <END>', '<START> white dog biting orange cat   <END>']
    ['<START> children with painted red faces being sprayed with water   on grass   <END>', '<START> the two children are being sprayed by water <END>', '<START> two children playing in a spray of water   <END>', '<START> two children standing in the grass being sprayed by a hose   <END>', '<START> two indian children are being squirted by a jet of water   <END>']
    ['<START> the middle eastern woman wearing the pink headscarf is walking beside a woman in a purple headscarf   <END>', '<START> two muslim woman wearing their head scarves and frowning faces   <END>', '<START> two women are wearing lavender scarves an their heads   <END>', '<START> two women dressed with scarves over their heads look angrily at the photographer   <END>', '<START> two women in head wraps   <END>']
    ['<START> a girl in a red jacket   surrounded by people   <END>', '<START> a woman in a puffy red jacket poses for a picture at an ice skating rink   <END>', '<START> a woman in a red coat is smiling   while people in the background are walking around in winter clothing   <END>', '<START> a woman wearing a red coat smiles down at the camera   <END>', '<START> the woman in a red jacket is smiling at the camera   <END>']
    ['<START> a wrestler throws another wrestler to the ground   <END>', '<START> two men engage in a professional wrestling match   <END>', '<START> two men in a fight in a ring <END>', '<START> two men in midair fighting in a professional wrestling ring   <END>', '<START> two wrestlers jump in a ring while an official watches   <END>']
    ['<START> a group of hockey players slide along the ice during a game   <END>', '<START> a group of men playing hockey   <END>', '<START> a hockey game is being played   <END>', '<START> a hockey goalie lays on the ice and other players skate past him   <END>', '<START> two hockey player teams playing a game on the ice <END>']
    ['<START> a girl in a purple tutu dances in the yard   <END>', '<START> a little girl in a puffy purple skirt dances in a park   <END>', '<START> a young girl twirls her fluffy purple skirt   <END>', '<START> little girl is spinning around on the grass in a flowing purple skirt   <END>', '<START> the little girl has a purple dress on   <END>']
    ['<START> a baby and a toddler are smiling whilst playing in a nursery   <END>', '<START> a baby in a bouncy seat and a standing boy surrounded by toys   <END>', '<START> a baby in a walker and an older child nearby   <END>', '<START> a young boy and his baby brother get excited about picture taking   <END>', '<START> two little boys are smiling and laughing while one is standing and one is in a bouncy seat   <END>']
    ['<START> a bicyclist does tricks on a lime green bike   <END>', '<START> a man dressed in black rides a green bike   <END>', '<START> a man performs a stoppie trick on his bmx green bicycle with no hands   <END>', '<START> man on green bicycle performing a trick on one wheel   <END>', '<START> the man is doing a trick on his green bike   <END>']
    
    bleu_score:{'1-garm': 0.6037632782455581, '2-garm': 0.3954779265024883}
    
 #### 实验 6 
 
    (1)使用 Keras Turning 进行参数调优, 3层 dropout , 每一层的可选参数为 [0.1, 0.2, 0.4]
    
    dropout_rate1 = hp.Choice('dropout_rate1',
                  values=[0.1, 0.2, 0.4])
    
    dropout_rate2 = hp.Choice('dropout_rate2',
                  values=[0.1, 0.2, 0.4])
    
    dropout_rate3 = hp.Choice('dropout_rate3',
                  values=[0.1, 0.2, 0.4])
    
    (2) 
    tuner（调参器）: kt.Hyperband
    优化目标 objective='val_loss', 
                
    (3) 最佳超参数
    Showing 3 best trials
    Objective(name='val_loss', direction='min')
    
    Trial summary
    Hyperparameters:
    dropout_rate1: 0.2
    dropout_rate2: 0.1
    dropout_rate3: 0.2
    tuner/epochs: 15
    tuner/initial_epoch: 5
    tuner/bracket: 1
    tuner/round: 1
    tuner/trial_id: c8e6194a4bc419ff066fd5fffe6c3808
    Score: 1.0103245973587036
    
    Trial summary
    Hyperparameters:
    dropout_rate1: 0.2
    dropout_rate2: 0.4
    dropout_rate3: 0.2
    tuner/epochs: 15
    tuner/initial_epoch: 0
    tuner/bracket: 0
    tuner/round: 0
    Score: 1.0103501081466675
    
    Trial summary
    Hyperparameters:
    dropout_rate1: 0.4
    dropout_rate2: 0.1
    dropout_rate3: 0.2
    tuner/epochs: 15
    tuner/initial_epoch: 5
    tuner/bracket: 1
    tuner/round: 1
    tuner/trial_id: 983ffa9a99a9f95d824e31c103ec5a30
    Score: 1.0106135606765747

    
    (3) 模型评价
    
    1.dropout_rates = (0.2, 0.1, 0.2)
    
    candidates:
    a young boy in a red shirt is playing with a brown and white dog <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    two white dogs are playing in the snow <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a young boy is playing with a red toy <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a man in a red jacket and a woman in a black jacket with a red scarf <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a woman in a red jacket and a black jacket is standing in a snow covered field <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a group of people are playing a game <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a hockey player in a red uniform is getting a break from the goal <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a girl in a pink dress is running through a park <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    two little boys are playing with a toy car <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    a boy in a red shirt is riding a bicycle down a sidewalk <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    
    references:
    ['<START> a boy holds a red bucket up to a pony   <END>', '<START> a man feeding a horse wearing a blue strapped blanket   <END>', '<START> a man holds feed for a horse in a blue blanket   <END>', '<START> a young boy is feeding a horse some hay from an orange bucket   <END>', '<START> grey horse wearing blue cover eating from a orange bucket held by a person in a green shirt   <END>']
    ['<START> a dog bites a cat whiel they lay on a bed together   <END>', '<START> a white dog biting an orange cat on the bed   pillows behind   <END>', '<START> a white dog biting at an orange cat   <END>', '<START> a white dog in a red collar biting an orange tabby cat on a bed   <END>', '<START> white dog biting orange cat   <END>']
    ['<START> children with painted red faces being sprayed with water   on grass   <END>', '<START> the two children are being sprayed by water <END>', '<START> two children playing in a spray of water   <END>', '<START> two children standing in the grass being sprayed by a hose   <END>', '<START> two indian children are being squirted by a jet of water   <END>']
    ['<START> the middle eastern woman wearing the pink headscarf is walking beside a woman in a purple headscarf   <END>', '<START> two muslim woman wearing their head scarves and frowning faces   <END>', '<START> two women are wearing lavender scarves an their heads   <END>', '<START> two women dressed with scarves over their heads look angrily at the photographer   <END>', '<START> two women in head wraps   <END>']
    ['<START> a girl in a red jacket   surrounded by people   <END>', '<START> a woman in a puffy red jacket poses for a picture at an ice skating rink   <END>', '<START> a woman in a red coat is smiling   while people in the background are walking around in winter clothing   <END>', '<START> a woman wearing a red coat smiles down at the camera   <END>', '<START> the woman in a red jacket is smiling at the camera   <END>']
    ['<START> a wrestler throws another wrestler to the ground   <END>', '<START> two men engage in a professional wrestling match   <END>', '<START> two men in a fight in a ring <END>', '<START> two men in midair fighting in a professional wrestling ring   <END>', '<START> two wrestlers jump in a ring while an official watches   <END>']
    ['<START> a group of hockey players slide along the ice during a game   <END>', '<START> a group of men playing hockey   <END>', '<START> a hockey game is being played   <END>', '<START> a hockey goalie lays on the ice and other players skate past him   <END>', '<START> two hockey player teams playing a game on the ice <END>']
    ['<START> a girl in a purple tutu dances in the yard   <END>', '<START> a little girl in a puffy purple skirt dances in a park   <END>', '<START> a young girl twirls her fluffy purple skirt   <END>', '<START> little girl is spinning around on the grass in a flowing purple skirt   <END>', '<START> the little girl has a purple dress on   <END>']
    ['<START> a baby and a toddler are smiling whilst playing in a nursery   <END>', '<START> a baby in a bouncy seat and a standing boy surrounded by toys   <END>', '<START> a baby in a walker and an older child nearby   <END>', '<START> a young boy and his baby brother get excited about picture taking   <END>', '<START> two little boys are smiling and laughing while one is standing and one is in a bouncy seat   <END>']
    ['<START> a bicyclist does tricks on a lime green bike   <END>', '<START> a man dressed in black rides a green bike   <END>', '<START> a man performs a stoppie trick on his bmx green bicycle with no hands   <END>', '<START> man on green bicycle performing a trick on one wheel   <END>', '<START> the man is doing a trick on his green bike   <END>']
    
    bleu_score:{'1-garm': 0.5893799525927057, '2-garm': 0.3807984116941174}

