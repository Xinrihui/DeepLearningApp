## 1.基本参数

### 1.1 数据集参数
    
    anki 数据集

### 1.2 模型参数
    
    基于词粒度(word level)
    
    词向量维度 n_embedding = 50
    
    隐藏层的维度 n_h = 50

## 2.实验记录

### 2.1 验证两种 seq2seq keras 建模方法的效果

#### 实验 1 - 解码采用一体化的模型方式

(prev_version/machine_translation_seq2seq_xrh.py)


    (1) 数据集
    
    英文-法文:
    anki/fra-eng/fra.txt
    
    sample num in whole dataset:  50000 (源句子-目标句子 对的数目)
    source_id num:  31502 (源句子的数目, 一个源句子会对应多个目标句子)
    
    训练数据比例 train_size=0.9
    
    N_train = 44978 (源句子-目标句子 对的数目)
    N_valid = 5022
    
    (2) 数据预处理
    
    freq_threshold=0 在语料库中出现次数小于0次的 不计入词表
    
    源语言词表大小: n_vocab_source=5911
    目标语言词表大小: n_vocab_target=10691
    
    源句子的长度: max_source_length = 9
    目标句子的长度: max_target_length = 16
    
    
    (3) 优化器参数
    
    epoch_num = 20
    batch_size = 128
    optimizer='rmsprop'
    
    (4) 训练好的模型位于: models/machine_translation_seq2seq_hid_50_emb_50.h5
    
    (5) 模型评价
    
    test source_id num:3151
    
    batch_sources:
    <START> i bet you're right  <END>
    <START> you're wise  <END>
    <START> i'm proud of you  <END>
    <START> we're lost  <END>
    <START> you can't fool us  <END>
    <START> i've found a good job  <END>
    <START> come again  <END>
    <START> he just got home  <END>
    <START> he's my husband  <END>
    <START> who's on duty today  <END>
    
    candidates:
    je sais que tu es en sécurité <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> (我知道你很安全)
    vous êtes <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> (您是)
    je suis heureux de vous <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> (我为你感到高兴)
    ils sont perdu <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> (他们迷路了)
    tu ne peux pas y aller <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> (你不能去那里)
    il <UNK> une bonne idée <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> (这是个好主意)
    allez ici <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> (过来这里)
    il <UNK> a besoin <UNK> <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> (他 需要 思考)
    <UNK> mon faute <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> (我的错)
    je suis en train de tom <UNK> <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> (我在训练汤姆)
    
    references:
    ['<START> je parie que vous avez raison  <END>', '<START> je parie que tu as raison  <END>']
    ['<START> vous êtes raisonnables  <END>']
    ['<START> je suis fier de toi  <END>']
    ['<START> nous sommes perdus  <END>', '<START> nous sommes perdues  <END>']
    ['<START> vous ne pouvez pas nous tromper  <END>', '<START> tu ne peux pas nous tromper  <END>']
    ["<START> j'ai trouvé un bon travail  <END>", "<START> j'ai trouvé un bon emploi  <END>"]
    ['<START> revenez nous voir  <END>', '<START> reviens nous voir  <END>', '<START> à la prochaine  <END>']
    ['<START> il vient juste de rentrer à la maison  <END>', '<START> il vient juste de rentrer chez lui  <END>', '<START> il vient de rentrer à la maison  <END>']
    ['<START> il est mon époux  <END>']
    ["<START> qui est de service aujourd'hui\u202f  <END>", "<START> qui est de garde aujourd'hui\u202f  <END>"]
    
    bleu_score:{'1-garm': 0.31442389638993895, '2-garm': 0.14175733740343172}


#### 实验 2 - 解码采用分步骤的方式
 
 (prev_version/machine_translation_seq2seq_ref_xrh.py)

    (1) 数据集
    
    英文-法文:
    anki/fra-eng/fra.txt
    
    sample num in whole dataset:  50000 (源句子-目标句子 对的数目)
    source_id num:  31502 (源句子的数目, 一个源句子会对应多个目标句子)
    
    训练数据比例 train_size=0.9
    
    N_train = 44978 (源句子-目标句子 对的数目)
    N_valid = 5022
    
    (2) 数据预处理
    
    freq_threshold=0 在语料库中出现次数小于0次的 不计入词表
    
    源语言词表大小: n_vocab_source=5911
    目标语言词表大小: n_vocab_target=10691
    
    源句子的长度: max_source_length = 9
    目标句子的长度: max_target_length = 16
    
    
    (3) 优化器参数
    
    epoch_num = 20
    batch_size = 128
    optimizer='rmsprop'
    
    (4) 训练好的模型位于: models/ref/machine_translation_seq2seq_hid_50_emb_50.h5
    
    (5) 模型评价
    
    test source_id num:3151
    
    batch_sources:
    <START> i bet you're right  <END>
    <START> you're wise  <END>
    <START> i'm proud of you  <END>
    <START> we're lost  <END>
    <START> you can't fool us  <END>
    <START> i've found a good job  <END>
    <START> come again  <END>
    <START> he just got home  <END>
    <START> he's my husband  <END>
    <START> who's on duty today  <END>
    
    candidates:
    je pense que tu es en train de faire une <END> <NULL> <NULL> <NULL> <NULL>  (我觉得你在做一个)
    vous êtes fort <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> (你很坚强)
    je suis en sécurité <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> (我很安全)
    <UNK> été <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    vous nous ne sommes pas <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    <UNK> un bon garçon <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    va toi à la maison <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    il me faut le français <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    <UNK> mon travail <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    <UNK> <UNK> en train de <UNK> <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    
    references:
    ['<START> je parie que vous avez raison  <END>', '<START> je parie que tu as raison  <END>']
    ['<START> vous êtes raisonnables  <END>']
    ['<START> je suis fier de toi  <END>']
    ['<START> nous sommes perdus  <END>', '<START> nous sommes perdues  <END>']
    ['<START> vous ne pouvez pas nous tromper  <END>', '<START> tu ne peux pas nous tromper  <END>']
    ["<START> j'ai trouvé un bon travail  <END>", "<START> j'ai trouvé un bon emploi  <END>"]
    ['<START> revenez nous voir  <END>', '<START> reviens nous voir  <END>', '<START> à la prochaine  <END>']
    ['<START> il vient juste de rentrer à la maison  <END>', '<START> il vient juste de rentrer chez lui  <END>', '<START> il vient de rentrer à la maison  <END>']
    ['<START> il est mon époux  <END>']
    ["<START> qui est de service aujourd'hui\u202f  <END>", "<START> qui est de garde aujourd'hui\u202f  <END>"]
    
    bleu_score:{'1-garm': 0.301879351474718, '2-garm': 0.13710124249264424}


### 2.2 验证增加 epoch_num 的效果

#### 实验 3 epoch_num = 50

(prev_version/machine_translation_seq2seq_xrh.py)

    (1) 数据集
    
    英文-法文:
    anki/fra-eng/fra.txt
    
    sample num in whole dataset:  50000 (源句子-目标句子 对的数目)
    source_id num:  31502 (源句子的数目, 一个源句子会对应多个目标句子)
    
    训练数据比例 train_size=0.9
    
    N_train = 44978 (源句子-目标句子 对的数目)
    N_valid = 5022
    
    (2) 数据预处理
    
    删除语料库中的数字, 符号( 保留 ' 和 - ), 和不可见字符
    
    freq_threshold=0 在语料库中出现次数小于0次的 不计入词表
    
    源语言词表大小: n_vocab_source=5910
    目标语言词表大小: n_vocab_target=10688
    
    源句子的长度: max_source_length = 9
    目标句子的长度: max_target_length = 16
    
    
    (3) 优化器参数
    
    epoch_num = 50
    batch_size = 128
    optimizer='rmsprop'
    
    (4) 训练好的模型位于: models/machine_translation_seq2seq_hid_50_emb_50.h5
    
    (5) 训练过程
    
    验证集上的损失在 epoch=48 时就不再下降了, 训练在 epoch=50 时停止
    
    Epoch 48/50
    351/351 [==============================] - ETA: 0s - loss: 0.6771 - accuracy: 0.8870
    Epoch 00048: val_loss improved from 0.94200 to 0.94173, saving model to models/cache/model.48-0.9417.hdf5
    351/351 [==============================] - 35s 99ms/step - loss: 0.6771 - accuracy: 0.8870 - val_loss: 0.9417 - val_accuracy: 0.8503
    
    Epoch 50/50
    351/351 [==============================] - ETA: 0s - loss: 0.6713 - accuracy: 0.8883
    Epoch 00050: val_loss did not improve from 0.94173
    351/351 [==============================] - 35s 99ms/step - loss: 0.6713 - accuracy: 0.8883 - val_loss: 0.9447 - val_accuracy: 0.8500
    
    (6) 模型评价
    
    1.epoch=50 时的模型
    
    batch_sources:
    <START> i bet you're right  <END>
    <START> you're wise  <END>
    <START> i'm proud of you  <END>
    <START> we're lost  <END>
    <START> you can't fool us  <END>
    <START> i've found a good job  <END>
    <START> come again  <END>
    <START> he just got home  <END>
    <START> he's my husband  <END>
    <START> who's on duty today  <END>
    
    candidates:
    je veux que <UNK> raison <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    tu es au moment <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    je suis fier de vous <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    nous sommes perdu <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    vous ne pouvez pas <UNK> <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    on se <UNK> un bon dit <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    reviens <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    il vient de la maison <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    <UNK> mon suis tom <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    je suis <UNK> de <UNK> <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    
    references:
    ['<START> je parie que vous avez raison  <END>', '<START> je parie que tu as raison  <END>']
    ['<START> vous êtes raisonnables  <END>']
    ['<START> je suis fier de toi  <END>']
    ['<START> nous sommes perdus  <END>', '<START> nous sommes perdues  <END>']
    ['<START> vous ne pouvez pas nous tromper  <END>', '<START> tu ne peux pas nous tromper  <END>']
    ["<START> j'ai trouvé un bon travail  <END>", "<START> j'ai trouvé un bon emploi  <END>"]
    ['<START> revenez nous voir  <END>', '<START> reviens nous voir  <END>', '<START> à la prochaine  <END>']
    ['<START> il vient juste de rentrer à la maison  <END>', '<START> il vient juste de rentrer chez lui  <END>', '<START> il vient de rentrer à la maison  <END>']
    ['<START> il est mon époux  <END>']
    ["<START> qui est de service aujourd'hui   <END>", "<START> qui est de garde aujourd'hui   <END>"]
    
    
    bleu_score:{'1-garm': 0.35544368181626196, '2-garm': 0.17612294838200665}

#### 实验 3-1 在 实验3 基础上继续训练模型

(machine_translation_seq2seq_xrh.py)

    (1) 优化器参数
    
    epoch_num = 50
    batch_size = 128
    optimizer='rmsprop'
    
    (2) 训练过程
    
    验证集上的损失在 epoch=11 时就不再下降了, 训练在 epoch=41 时停止
    
    Epoch 11/50
    351/351 [==============================] - ETA: 0s - loss: 0.6368 - accuracy: 0.8947
    Epoch 00011: val_loss improved from 0.94336 to 0.94176, saving model to models/cache/model.11-0.9418.hdf5
    351/351 [==============================] - 35s 99ms/step - loss: 0.6368 - accuracy: 0.8947 - val_loss: 0.9418 - val_accuracy: 0.8500
    
    Epoch 41/50
    351/351 [==============================] - ETA: 0s - loss: 0.5843 - accuracy: 0.9048
    Epoch 00041: val_loss did not improve from 0.94176
    351/351 [==============================] - 35s 99ms/step - loss: 0.5843 - accuracy: 0.9048 - val_loss: 0.9737 - val_accuracy: 0.8459
    
    虽然验证集的损失没有下降, 但是这并不一定代表训练是没有效果的, 因为数据集中, 源句子对应的目标目标句子不多, 而一句话往往会有多种翻译,
    这导致可能模型明明翻译的很好的一句话, 而对其 bleu 评分却不高; 因此, 源句子的对照句子越多, 这个数据集的质量越好.
    
    (3) 模型评价
    
    1.epoch=41 时的模型
    
    batch_sources:
    <START> i bet you're right  <END>
    <START> you're wise  <END>
    <START> i'm proud of you  <END>
    <START> we're lost  <END>
    <START> you can't fool us  <END>
    <START> i've found a good job  <END>
    <START> come again  <END>
    <START> he just got home  <END>
    <START> he's my husband  <END>
    <START> who's on duty today  <END>
    
    candidates:
    je parie que tu as raison <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> (我打赌你是对的)
    vous êtes <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> (您是)
    je suis fier de vous <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> (我为你感到骄傲)
    je suis perdu <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> (我迷路了)
    vous ne pouvez pas y <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> (你不能在那里)
    <UNK> trouvé un bon boulot <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> (找到一份好工作)
    <UNK> à nouveau <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> (再次)
    il vient de la maison <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> (他从家里来的)
    <UNK> mon type <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> (我的类型)
    <UNK> un il est à <UNK> <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> (它在)
    
    references:
    ['<START> je parie que vous avez raison  <END>', '<START> je parie que tu as raison  <END>']
    ['<START> vous êtes raisonnables  <END>']
    ['<START> je suis fier de toi  <END>']
    ['<START> nous sommes perdus  <END>', '<START> nous sommes perdues  <END>']
    ['<START> vous ne pouvez pas nous tromper  <END>', '<START> tu ne peux pas nous tromper  <END>']
    ["<START> j'ai trouvé un bon travail  <END>", "<START> j'ai trouvé un bon emploi  <END>"]
    ['<START> revenez nous voir  <END>', '<START> reviens nous voir  <END>', '<START> à la prochaine  <END>']
    ['<START> il vient juste de rentrer à la maison  <END>', '<START> il vient juste de rentrer chez lui  <END>', '<START> il vient de rentrer à la maison  <END>']
    ['<START> il est mon époux  <END>']
    ["<START> qui est de service aujourd'hui   <END>", "<START> qui est de garde aujourd'hui   <END>"]
    
    bleu_score:{'1-garm': 0.36580329557955715, '2-garm': 0.19272407626617513}
    
    2.epoch=11 时的模型
    
    batch_sources:
    <START> i bet you're right  <END>
    <START> you're wise  <END>
    <START> i'm proud of you  <END>
    <START> we're lost  <END>
    <START> you can't fool us  <END>
    <START> i've found a good job  <END>
    <START> come again  <END>
    <START> he just got home  <END>
    <START> he's my husband  <END>
    <START> who's on duty today  <END>
    
    candidates:
    je parie que tu as raison <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    tu es ceci <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    je suis fier de vous <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    je suis perdu <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    vous ne pouvez pas y y aller <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    <UNK> trouvé un bon boulot <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    reviens <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    il vient de la maison <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    <UNK> mon qui est tom <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    il est à <UNK> il <END> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>
    
    references:
    ['<START> je parie que vous avez raison  <END>', '<START> je parie que tu as raison  <END>']
    ['<START> vous êtes raisonnables  <END>']
    ['<START> je suis fier de toi  <END>']
    ['<START> nous sommes perdus  <END>', '<START> nous sommes perdues  <END>']
    ['<START> vous ne pouvez pas nous tromper  <END>', '<START> tu ne peux pas nous tromper  <END>']
    ["<START> j'ai trouvé un bon travail  <END>", "<START> j'ai trouvé un bon emploi  <END>"]
    ['<START> revenez nous voir  <END>', '<START> reviens nous voir  <END>', '<START> à la prochaine  <END>']
    ['<START> il vient juste de rentrer à la maison  <END>', '<START> il vient juste de rentrer chez lui  <END>', '<START> il vient de rentrer à la maison  <END>']
    ['<START> il est mon époux  <END>']
    ["<START> qui est de service aujourd'hui   <END>", "<START> qui est de garde aujourd'hui   <END>"]
    
    bleu_score:{'1-garm': 0.36153853023100885, '2-garm': 0.18639428157156804}


