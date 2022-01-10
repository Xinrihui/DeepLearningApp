## 1.基本参数

### 1.1 数据集参数
    
    Flicker8k 数据集


### 1.2 模型参数

    词向量维度 n_embedding = 512
    隐藏层的维度 n_h = 512

## 2.实验记录

### 2.1 词表大小对模型效果的影响

#### 实验1

(1) 图片嵌入层使用 relu 激活函数

(2) 数据预处理
freq_threshold=0 在语料库中出现次数小于0次的 不计入词表

词表大小: n_vocab=9199

经过 CNN 抽取的图片向量维度 n_image_feature = 2048

图片描述句子的长度: max_caption_length=40

(3) 优化器参数
epoch_num = 50 ( 迭代的次数不够会出现推理的句子都是一样的现象 )
batch_size = 64
decay=0.01

(4) 训练好的模型位于: models/image_caption_naive_lstm_hid_512_emb_512_thres_0_len_40.h5

(5) 模型评价

test image num:1619

candidates:  [
'A man and a woman are playing a guitar and a little girl in a pink dress <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK>',
 'A dog is running through a field of grass <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL>',
 'A little girl in a pink shirt is playing with a red ball <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END>',
 'A man in a black shirt and black pants is standing in front of a crowd of people <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL>',
 'A man in a blue shirt is standing in front of a brick wall <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK>',
 'A group of people are riding on a roller coaster <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END>',
 'A football player is being tackled by the player in the field <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL>',
 'A little girl in a pink dress is playing with a red ball <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END>',
 'A little girl in a pink dress is playing with a red ball <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END>',
 'A man in a red shirt is riding a bike on a dirt road <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK> <END> <NULL> <UNK>']

reference:  [
['<START> A boy holds a red bucket up to a pony . <END>', '<START> A man feeding a horse wearing a blue strapped blanket . <END>', '<START> A man holds feed for a horse in a blue blanket . <END>', '<START> A young boy is feeding a horse some hay from an orange bucket . <END>', '<START> Grey horse wearing blue cover eating from a orange bucket held by a person in a green shirt . <END>'],
['<START> A dog bites a cat whiel they lay on a bed together . <END>', '<START> A white dog biting an orange cat on the bed , pillows behind . <END>', '<START> A white dog biting at an orange cat . <END>', '<START> A white dog in a red collar biting an orange tabby cat on a bed . <END>', '<START> White dog biting orange cat . <END>'],
['<START> Children with painted red faces being sprayed with water , on grass . <END>', '<START> The two children are being sprayed by water <END>', '<START> Two children playing in a spray of water . <END>', '<START> Two children standing in the grass being sprayed by a hose . <END>', '<START> Two indian children are being squirted by a jet of water . <END>'],
['<START> The middle eastern woman wearing the pink headscarf is walking beside a woman in a purple headscarf . <END>', '<START> Two Muslim woman wearing their head scarves and frowning faces . <END>', '<START> Two women are wearing lavender scarves an their heads . <END>', '<START> Two women dressed with scarves over their heads look angrily at the photographer . <END>', '<START> Two women in head wraps . <END>'],
['<START> A girl in a red jacket , surrounded by people . <END>', '<START> A woman in a puffy red jacket poses for a picture at an ice skating rink . <END>', '<START> A woman in a red coat is smiling , while people in the background are walking around in winter clothing . <END>', '<START> A woman wearing a red coat smiles down at the camera . <END>', '<START> The woman in a red jacket is smiling at the camera . <END>'],
['<START> a wrestler throws another wrestler to the ground . <END>', '<START> Two men engage in a professional wrestling match . <END>', '<START> two men in a fight in a ring <END>', '<START> Two men in midair fighting in a professional wrestling ring . <END>', '<START> Two wrestlers jump in a ring while an official watches . <END>'],
['<START> A group of hockey players slide along the ice during a game . <END>', '<START> A group of men playing hockey . <END>', '<START> A hockey game is being played . <END>', '<START> A hockey goalie lays on the ice and other players skate past him . <END>', '<START> two hockey player teams playing a game on the ice <END>'],
['<START> A girl in a purple tutu dances in the yard . <END>', '<START> A little girl in a puffy purple skirt dances in a park . <END>', '<START> A young girl twirls her fluffy purple skirt . <END>', '<START> Little girl is spinning around on the grass in a flowing purple skirt . <END>', '<START> The little girl has a purple dress on . <END>'],
['<START> A baby and a toddler are smiling whilst playing in a nursery . <END>', '<START> A baby in a bouncy seat and a standing boy surrounded by toys . <END>', '<START> A baby in a walker and an older child nearby . <END>', '<START> A young boy and his baby brother get excited about picture taking . <END>', '<START> Two little boys are smiling and laughing while one is standing and one is in a bouncy seat . <END>'],
['<START> A bicyclist does tricks on a lime green bike . <END>', '<START> A man dressed in black rides a green bike . <END>', '<START> A man performs a stoppie trick on his BMX green bicycle with no hands . <END>', '<START> Man on green bicycle performing a trick on one wheel . <END>', '<START> The man is doing a trick on his green bike . <END>']]


推理序列的长度: 39
bleu_score:{'1-garm': 0.5245789698114178, '2-garm': 0.3040563058902678}


#### 实验2

(1) 图片嵌入层使用 relu 激活函数

(2) 数据预处理
freq_threshold=3 在语料库中出现次数小于3次的 不计入词表
词表大小 : 4217

词表大小: n_vocab=9199

经过 CNN 抽取的图片向量维度 n_image_feature = 2048

图片描述句子的长度: max_caption_length=40

(3) 优化器参数
epoch_num = 50 ( 迭代的次数不够会出现推理的句子都是一样的现象 )
batch_size = 64
decay=0.01

(4) 训练过程

Epoch 1/50
505/505 [==============================] - 57s 113ms/step - loss: 1.1706 - accuracy: 0.1437
Epoch 2/50
505/505 [==============================] - 57s 114ms/step - loss: 0.9856 - accuracy: 0.1552
...
Epoch 49/50
505/505 [==============================] - 57s 113ms/step - loss: 0.7688 - accuracy: 0.1774
Epoch 50/50
505/505 [==============================] - 57s 113ms/step - loss: 0.7699 - accuracy: 0.1782

(5) 训练好的模型位于: models/image_caption_naive_lstm_hid_512_emb_512_thres_3_len_40.h5

(6) 模型评价

candidates:
['A man in a red shirt is standing on a rock in front of a crowd of people . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL>',
'A dog is running through the grass . <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> . <END> <NULL> <END> <NULL> . <END> <NULL> <END> <NULL> . <END> <NULL> <END> <NULL> <END> <NULL> . <END>',
'A man in a blue shirt is standing in front of a large rock formation . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> <END> <NULL> . <END> <NULL> <END> <NULL> . <END>',
'A man in a black shirt and jeans is standing in front of a large rock formation . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> <END> <NULL> . <END>',
'A man in a blue shirt is standing in front of a large rock formation . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> <END> <NULL> <END> <NULL> . <END>',
 'A man in a red shirt is standing on a rock in front of a crowd of people . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL>',
 'A man in a red shirt is standing on a rock in front of a crowd of people . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL>',
 'A man in a blue shirt is standing on a rock in front of a crowd of people . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL>',
 'A man in a red shirt is standing on a rock in front of a crowd of people . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL>',
 'A man in a blue shirt is standing on a rock in front of a crowd . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> <END> <NULL> . <END> <NULL>']
reference:  [
['<START> A boy holds a red bucket up to a pony . <END>', '<START> A man feeding a horse wearing a blue strapped blanket . <END>', '<START> A man holds feed for a horse in a blue blanket . <END>', '<START> A young boy is feeding a horse some hay from an orange bucket . <END>', '<START> Grey horse wearing blue cover eating from a orange bucket held by a person in a green shirt . <END>'],
['<START> A dog bites a cat whiel they lay on a bed together . <END>', '<START> A white dog biting an orange cat on the bed , pillows behind . <END>', '<START> A white dog biting at an orange cat . <END>', '<START> A white dog in a red collar biting an orange tabby cat on a bed . <END>', '<START> White dog biting orange cat . <END>'],
['<START> Children with painted red faces being sprayed with water , on grass . <END>', '<START> The two children are being sprayed by water <END>', '<START> Two children playing in a spray of water . <END>', '<START> Two children standing in the grass being sprayed by a hose . <END>', '<START> Two indian children are being squirted by a jet of water . <END>'],
['<START> The middle eastern woman wearing the pink headscarf is walking beside a woman in a purple headscarf . <END>', '<START> Two Muslim woman wearing their head scarves and frowning faces . <END>', '<START> Two women are wearing lavender scarves an their heads . <END>', '<START> Two women dressed with scarves over their heads look angrily at the photographer . <END>', '<START> Two women in head wraps . <END>'],
['<START> A girl in a red jacket , surrounded by people . <END>', '<START> A woman in a puffy red jacket poses for a picture at an ice skating rink . <END>', '<START> A woman in a red coat is smiling , while people in the background are walking around in winter clothing . <END>', '<START> A woman wearing a red coat smiles down at the camera . <END>', '<START> The woman in a red jacket is smiling at the camera . <END>'],
['<START> a wrestler throws another wrestler to the ground . <END>', '<START> Two men engage in a professional wrestling match . <END>', '<START> two men in a fight in a ring <END>', '<START> Two men in midair fighting in a professional wrestling ring . <END>', '<START> Two wrestlers jump in a ring while an official watches . <END>'],
['<START> A group of hockey players slide along the ice during a game . <END>', '<START> A group of men playing hockey . <END>', '<START> A hockey game is being played . <END>', '<START> A hockey goalie lays on the ice and other players skate past him . <END>', '<START> two hockey player teams playing a game on the ice <END>'],
['<START> A girl in a purple tutu dances in the yard . <END>', '<START> A little girl in a puffy purple skirt dances in a park . <END>', '<START> A young girl twirls her fluffy purple skirt . <END>', '<START> Little girl is spinning around on the grass in a flowing purple skirt . <END>', '<START> The little girl has a purple dress on . <END>'],
['<START> A baby and a toddler are smiling whilst playing in a nursery . <END>', '<START> A baby in a bouncy seat and a standing boy surrounded by toys . <END>', '<START> A baby in a walker and an older child nearby . <END>', '<START> A young boy and his baby brother get excited about picture taking . <END>', '<START> Two little boys are smiling and laughing while one is standing and one is in a bouncy seat . <END>'],
['<START> A bicyclist does tricks on a lime green bike . <END>', '<START> A man dressed in black rides a green bike . <END>', '<START> A man performs a stoppie trick on his BMX green bicycle with no hands . <END>', '<START> Man on green bicycle performing a trick on one wheel . <END>', '<START> The man is doing a trick on his green bike . <END>']]

bleu_score:{'1-garm': 0.45003001097494794, '2-garm': 0.25489965047512314}


把词表缩小后模型的效果下降的厉害, 由此可见机器学习大部分的问题并不出在模型, 而是在数据上


### 2.2 图片描述句子的长度即输入序列的长度 对模型效果的影响

#### 实验3

(1) 图片嵌入层使用 relu 激活函数

(2) 数据预处理
freq_threshold=0 在语料库中出现次数小于0次的 不计入词表

词表大小: n_vocab=9199

经过 CNN 抽取的图片向量维度 n_image_feature = 2048

图片描述句子的长度: max_caption_length=30

(3) 优化器参数
epoch_num = 50 ( 迭代的次数不够会出现推理的句子都是一样的现象 )
batch_size = 64
decay=0.01

(4) 训练好的模型位于: models/image_caption_naive_lstm_hid_512_emb_512_thres_0_len_30.h5

(5) 训练过程

Epoch 1/50
505/505 [==============================] - 86s 171ms/step - loss: 1.9730 - accuracy: 0.1786
Epoch 2/50
505/505 [==============================] - 87s 171ms/step - loss: 1.4595 - accuracy: 0.2015
...
Epoch 49/50
505/505 [==============================] - 87s 172ms/step - loss: 1.0004 - accuracy: 0.2464
Epoch 50/50
505/505 [==============================] - 89s 176ms/step - loss: 1.0027 - accuracy: 0.2470

(6) 模型评价

candidates:
['A man and a woman are playing with a dog . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> .',
'A dog is running through a field of grass and grass . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL>',
'A little girl in a pink shirt is jumping into a pool . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END>',
'A man in a red shirt and a woman in a white shirt and a black hat , a white and white , and a black , one in',
'A man in a black jacket is standing on a rock wall . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END>',
'A group of people are riding horses in a race . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> .',
'A group of people are playing in a field of a race . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END>',
 'A young girl in a pink dress is jumping on a swing . <END> <NULL> . <END> <NULL> . <END> <NULL> <END> <NULL> . <END> <NULL> . <END> <NULL>',
 'A little girl in a pink dress is playing with a hula hoop . <END> <NULL> . <END> <NULL> . <END> <NULL> <END> <NULL> <END> <NULL> . <END> <NULL>',
 'A man in a red shirt is riding a bike on a street . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> .']

reference:  [
['<START> A boy holds a red bucket up to a pony . <END>', '<START> A man feeding a horse wearing a blue strapped blanket . <END>', '<START> A man holds feed for a horse in a blue blanket . <END>', '<START> A young boy is feeding a horse some hay from an orange bucket . <END>', '<START> Grey horse wearing blue cover eating from a orange bucket held by a person in a green shirt . <END>'],
['<START> A dog bites a cat whiel they lay on a bed together . <END>', '<START> A white dog biting an orange cat on the bed , pillows behind . <END>', '<START> A white dog biting at an orange cat . <END>', '<START> A white dog in a red collar biting an orange tabby cat on a bed . <END>', '<START> White dog biting orange cat . <END>'],
['<START> Children with painted red faces being sprayed with water , on grass . <END>', '<START> The two children are being sprayed by water <END>', '<START> Two children playing in a spray of water . <END>', '<START> Two children standing in the grass being sprayed by a hose . <END>', '<START> Two indian children are being squirted by a jet of water . <END>'],
['<START> The middle eastern woman wearing the pink headscarf is walking beside a woman in a purple headscarf . <END>', '<START> Two Muslim woman wearing their head scarves and frowning faces . <END>', '<START> Two women are wearing lavender scarves an their heads . <END>', '<START> Two women dressed with scarves over their heads look angrily at the photographer . <END>', '<START> Two women in head wraps . <END>'],
['<START> A girl in a red jacket , surrounded by people . <END>', '<START> A woman in a puffy red jacket poses for a picture at an ice skating rink . <END>', '<START> A woman in a red coat is smiling , while people in the background are walking around in winter clothing . <END>', '<START> A woman wearing a red coat smiles down at the camera . <END>', '<START> The woman in a red jacket is smiling at the camera . <END>'],
['<START> a wrestler throws another wrestler to the ground . <END>', '<START> Two men engage in a professional wrestling match . <END>', '<START> two men in a fight in a ring <END>', '<START> Two men in midair fighting in a professional wrestling ring . <END>', '<START> Two wrestlers jump in a ring while an official watches . <END>'],
['<START> A group of hockey players slide along the ice during a game . <END>', '<START> A group of men playing hockey . <END>', '<START> A hockey game is being played . <END>', '<START> A hockey goalie lays on the ice and other players skate past him . <END>', '<START> two hockey player teams playing a game on the ice <END>'],
['<START> A girl in a purple tutu dances in the yard . <END>', '<START> A little girl in a puffy purple skirt dances in a park . <END>', '<START> A young girl twirls her fluffy purple skirt . <END>', '<START> Little girl is spinning around on the grass in a flowing purple skirt . <END>', '<START> The little girl has a purple dress on . <END>'],
['<START> A baby and a toddler are smiling whilst playing in a nursery . <END>', '<START> A baby in a bouncy seat and a standing boy surrounded by toys . <END>', '<START> A baby in a walker and an older child nearby . <END>', '<START> A young boy and his baby brother get excited about picture taking . <END>', '<START> Two little boys are smiling and laughing while one is standing and one is in a bouncy seat . <END>'],
['<START> A bicyclist does tricks on a lime green bike . <END>', '<START> A man dressed in black rides a green bike . <END>', '<START> A man performs a stoppie trick on his BMX green bicycle with no hands . <END>', '<START> Man on green bicycle performing a trick on one wheel . <END>', '<START> The man is doing a trick on his green bike . <END>']]

bleu_score:{'1-garm': 0.5262316969574249, '2-garm': 0.3089612022877798}

再进行 50 epoch 的训练:

epoch_num = 50
batch_size = 64
decay=0.01

candidates:
['A man and a woman are walking towards a potato chip truck . <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> . <END> <NULL> .',
'A white dog is biting a black dog on the floor . <END> <NULL> . <END> <NULL> . <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>',
'A young boy is playing in a yard with a green ball . <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL>',
'Two women in black are standing in front of a crowd of people . <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END>',
'A man in a black shirt and jeans is standing in front of a rock wall . <END> <NULL> <END> <NULL> <END> <NULL> . <END> <NULL> <END> <NULL> .',
'A group of men are playing rugby . <END> <NULL> <END> <NULL> <END> <NULL> . <END> <NULL> <END> <NULL> <END> <NULL> . <END> <NULL> <END> <NULL> . <END> <NULL>',
'A hockey player in a red uniform is guarding the goal . <END> <NULL> . <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> . <END> <NULL> <END> <NULL> <END>',
'A girl in a pink dress is running through a field of grass . <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> . <END> <NULL> <END> <NULL>',
'A little girl in a pink dress is holding a camera . <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> <END> <NULL> . <END> <NULL> <END> <NULL>',
'A man in a red shirt is riding a unicycle . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> . <END> <NULL> .']

reference:  [
['<START> A boy holds a red bucket up to a pony . <END>', '<START> A man feeding a horse wearing a blue strapped blanket . <END>', '<START> A man holds feed for a horse in a blue blanket . <END>', '<START> A young boy is feeding a horse some hay from an orange bucket . <END>', '<START> Grey horse wearing blue cover eating from a orange bucket held by a person in a green shirt . <END>'],
['<START> A dog bites a cat whiel they lay on a bed together . <END>', '<START> A white dog biting an orange cat on the bed , pillows behind . <END>', '<START> A white dog biting at an orange cat . <END>', '<START> A white dog in a red collar biting an orange tabby cat on a bed . <END>', '<START> White dog biting orange cat . <END>'],
['<START> Children with painted red faces being sprayed with water , on grass . <END>', '<START> The two children are being sprayed by water <END>', '<START> Two children playing in a spray of water . <END>', '<START> Two children standing in the grass being sprayed by a hose . <END>', '<START> Two indian children are being squirted by a jet of water . <END>'],
['<START> The middle eastern woman wearing the pink headscarf is walking beside a woman in a purple headscarf . <END>', '<START> Two Muslim woman wearing their head scarves and frowning faces . <END>', '<START> Two women are wearing lavender scarves an their heads . <END>', '<START> Two women dressed with scarves over their heads look angrily at the photographer . <END>', '<START> Two women in head wraps . <END>'],
['<START> A girl in a red jacket , surrounded by people . <END>', '<START> A woman in a puffy red jacket poses for a picture at an ice skating rink . <END>', '<START> A woman in a red coat is smiling , while people in the background are walking around in winter clothing . <END>', '<START> A woman wearing a red coat smiles down at the camera . <END>', '<START> The woman in a red jacket is smiling at the camera . <END>'],
['<START> a wrestler throws another wrestler to the ground . <END>', '<START> Two men engage in a professional wrestling match . <END>', '<START> two men in a fight in a ring <END>', '<START> Two men in midair fighting in a professional wrestling ring . <END>', '<START> Two wrestlers jump in a ring while an official watches . <END>'],
['<START> A group of hockey players slide along the ice during a game . <END>', '<START> A group of men playing hockey . <END>', '<START> A hockey game is being played . <END>', '<START> A hockey goalie lays on the ice and other players skate past him . <END>', '<START> two hockey player teams playing a game on the ice <END>'],
['<START> A girl in a purple tutu dances in the yard . <END>', '<START> A little girl in a puffy purple skirt dances in a park . <END>', '<START> A young girl twirls her fluffy purple skirt . <END>', '<START> Little girl is spinning around on the grass in a flowing purple skirt . <END>', '<START> The little girl has a purple dress on . <END>'],
['<START> A baby and a toddler are smiling whilst playing in a nursery . <END>', '<START> A baby in a bouncy seat and a standing boy surrounded by toys . <END>', '<START> A baby in a walker and an older child nearby . <END>', '<START> A young boy and his baby brother get excited about picture taking . <END>', '<START> Two little boys are smiling and laughing while one is standing and one is in a bouncy seat . <END>'],
['<START> A bicyclist does tricks on a lime green bike . <END>', '<START> A man dressed in black rides a green bike . <END>', '<START> A man performs a stoppie trick on his BMX green bicycle with no hands . <END>', '<START> Man on green bicycle performing a trick on one wheel . <END>', '<START> The man is doing a trick on his green bike . <END>']]

bleu_score:{'1-garm': 0.5373025248915398, '2-garm': 0.3228842877393273}

### 2.3 在 LSTM 加入 mask 对模型效果的影响

#### 实验4

(1) 图片嵌入层使用 relu 激活函数

(2) 数据预处理
freq_threshold=0 在语料库中出现次数小于0次的 不计入词表

词表大小: n_vocab=9199

经过 CNN 抽取的图片向量维度 n_image_feature = 2048

图片描述句子的长度: max_caption_length=40

(3) 优化器参数
epoch_num = 50 ( 迭代的次数不够会出现推理的句子都是一样的现象 )
batch_size = 64
decay=0.01

(4) 训练好的模型位于: models/image_caption_naive_lstm_hid_512_emb_512_thres_0_len_40_no_mask.h5

(5) 训练过程

Epoch 1/50
505/505 [==============================] - 112s 222ms/step - loss: 1.2784 - accuracy: 0.7819
Epoch 2/50
505/505 [==============================] - 110s 219ms/step - loss: 1.0380 - accuracy: 0.8014
...
Epoch 49/50
505/505 [==============================] - 115s 229ms/step - loss: 0.7187 - accuracy: 0.8366
Epoch 50/50
505/505 [==============================] - 114s 225ms/step - loss: 0.7137 - accuracy: 0.8375

(6) 模型评价

candidates:  ['A man and a woman are standing in front of a large waterfall <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>', 'A dog is running through a field of grass <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>', 'A little girl in a pink dress is standing in a field of grass <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>', 'A man and a woman are sitting on a bench <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>', 'A man in a red jacket is standing on a rocky shore <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>', 'A group of people are standing in a field of grass <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>', 'A group of people are playing soccer on a field <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>', 'A young boy wearing a blue shirt is playing with a toy <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>', 'A little girl in a pink dress is playing with a hula hoop <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>', 'A man in a red shirt is riding a bicycle down a paved road <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>']

reference:  [
['<START> A boy holds a red bucket up to a pony . <END>', '<START> A man feeding a horse wearing a blue strapped blanket . <END>', '<START> A man holds feed for a horse in a blue blanket . <END>', '<START> A young boy is feeding a horse some hay from an orange bucket . <END>', '<START> Grey horse wearing blue cover eating from a orange bucket held by a person in a green shirt . <END>'],
['<START> A dog bites a cat whiel they lay on a bed together . <END>', '<START> A white dog biting an orange cat on the bed , pillows behind . <END>', '<START> A white dog biting at an orange cat . <END>', '<START> A white dog in a red collar biting an orange tabby cat on a bed . <END>', '<START> White dog biting orange cat . <END>'],
['<START> Children with painted red faces being sprayed with water , on grass . <END>', '<START> The two children are being sprayed by water <END>', '<START> Two children playing in a spray of water . <END>', '<START> Two children standing in the grass being sprayed by a hose . <END>', '<START> Two indian children are being squirted by a jet of water . <END>'],
['<START> The middle eastern woman wearing the pink headscarf is walking beside a woman in a purple headscarf . <END>', '<START> Two Muslim woman wearing their head scarves and frowning faces . <END>', '<START> Two women are wearing lavender scarves an their heads . <END>', '<START> Two women dressed with scarves over their heads look angrily at the photographer . <END>', '<START> Two women in head wraps . <END>'],
['<START> A girl in a red jacket , surrounded by people . <END>', '<START> A woman in a puffy red jacket poses for a picture at an ice skating rink . <END>', '<START> A woman in a red coat is smiling , while people in the background are walking around in winter clothing . <END>', '<START> A woman wearing a red coat smiles down at the camera . <END>', '<START> The woman in a red jacket is smiling at the camera . <END>'],
['<START> a wrestler throws another wrestler to the ground . <END>', '<START> Two men engage in a professional wrestling match . <END>', '<START> two men in a fight in a ring <END>', '<START> Two men in midair fighting in a professional wrestling ring . <END>', '<START> Two wrestlers jump in a ring while an official watches . <END>'],
['<START> A group of hockey players slide along the ice during a game . <END>', '<START> A group of men playing hockey . <END>', '<START> A hockey game is being played . <END>', '<START> A hockey goalie lays on the ice and other players skate past him . <END>', '<START> two hockey player teams playing a game on the ice <END>'],
['<START> A girl in a purple tutu dances in the yard . <END>', '<START> A little girl in a puffy purple skirt dances in a park . <END>', '<START> A young girl twirls her fluffy purple skirt . <END>', '<START> Little girl is spinning around on the grass in a flowing purple skirt . <END>', '<START> The little girl has a purple dress on . <END>'],
['<START> A baby and a toddler are smiling whilst playing in a nursery . <END>', '<START> A baby in a bouncy seat and a standing boy surrounded by toys . <END>', '<START> A baby in a walker and an older child nearby . <END>', '<START> A young boy and his baby brother get excited about picture taking . <END>', '<START> Two little boys are smiling and laughing while one is standing and one is in a bouncy seat . <END>'],
['<START> A bicyclist does tricks on a lime green bike . <END>', '<START> A man dressed in black rides a green bike . <END>', '<START> A man performs a stoppie trick on his BMX green bicycle with no hands . <END>', '<START> Man on green bicycle performing a trick on one wheel . <END>', '<START> The man is doing a trick on his green bike . <END>']]

bleu_score:{'1-garm': 0.5288176821352424, '2-garm': 0.30357563757453204}

