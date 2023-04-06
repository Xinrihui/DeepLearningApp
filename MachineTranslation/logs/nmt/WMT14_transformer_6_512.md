## 1.基本参数

### 1.1 数据集参数
    
    WMT-14-English-Germa 数据集
    (预处理后的 数据集地址 https://nlp.stanford.edu/projects/nmt/ )
    
### 1.2 模型参数

配置文件: config/transformer_seq2seq.ini (tag='DEFAULT') 
          
    # 源句子的最大编码位置
    maximum_position_source = 1000
    
    # 目标句子的最大编码位置
    maximum_position_target = 1000
    
    # 堆叠的编码器(解码器)的层数
    num_layers = 6
    
    # 模型整体的隐藏层的维度
    d_model = 512
    
    #并行注意力层的个数(头数)
    num_heads = 8
    
    #Position-wise Feed-Forward 的中间层的维度
    dff = 2048

## 2.实验记录

### 2.1 验证 warmup_steps 的效果

#### 实验 1  warmup_steps=32000
    
    (0) 模型
       
    (1) 数据集 
    
    训练数据:    
    N_train = 4343134 ( 源句子-目标句子 pair 的数目, 过滤掉长度大于 64 )
    
    most common seq length: (seq length, count num)
    [(17, 155827), (16, 155629), (15, 154656), (18, 153586), (19, 151796), (14, 149838), (20, 147657), (13, 143601), (21, 143144), (22, 139581)]
    seq length count:  (seq length, count num)
    [(100, 572), (99, 616), (98, 645), (97, 711), (96, 782), (95, 829), (94, 859), (93, 1004), (92, 1022), (91, 1117), (90, 1271), (89, 1305), (88, 1543), (87, 1572), (86, 1629), (85, 1698), (84, 1904), (83, 1976), (82, 2154), (81, 2365), (80, 2506), (79, 2846), (78, 3033), (77, 3184), (76, 3518), (75, 3630), (74, 4014), (73, 4170), (72, 4749), (71, 5061), (70, 5388), (69, 5590), (68, 6176), (67, 6689), (66, 7217), (65, 7744), (64, 8343), (63, 9033), (62, 9905), (61, 10646), (60, 11308), (59, 12336), (58, 13193), (57, 14414), (56, 15450), (55, 16969), (54, 18876), (53, 20326), (52, 20903), (51, 22590), (50, 24098), (49, 25983), (48, 28325), (47, 30461), (46, 32027), (45, 35146), (44, 37382), (43, 40306), (42, 43267), (41, 46180), (40, 50308), (39, 53441), (38, 57900), (37, 62886), (36, 66840), (35, 71266), (34, 76766), (33, 81583), (32, 87484), (31, 92126), (30, 97864), (29, 102435), (28, 107874), (27, 113869), (26, 119401), (25, 124660), (24, 130374), (23, 135714), (22, 139581), (21, 143144), (20, 147657), (19, 151796), (18, 153586), (17, 155827), (16, 155629), (15, 154656), (14, 149838), (13, 143601), (12, 138078), (11, 128739), (10, 114370), (9, 102814), (8, 79692), (7, 59132), (6, 26251), (5, 19701), (4, 8034), (3, 6680), (2, 2357), (1, 6330)]
    target sentence length distribution:
    most common seq length: (seq length, count num)
    [(16, 165532), (15, 164658), (14, 162805), (17, 162654), (18, 159080), (13, 156485), (19, 155026), (20, 152030), (12, 150160), (21, 146299)]
    seq length count:  (seq length, count num)
    [(100, 296), (99, 305), (98, 337), (97, 404), (96, 408), (95, 505), (94, 531), (93, 592), (92, 625), (91, 679), (90, 780), (89, 823), (88, 946), (87, 1030), (86, 1043), (85, 1231), (84, 1310), (83, 1419), (82, 1578), (81, 1669), (80, 1809), (79, 2021), (78, 2093), (77, 2262), (76, 2526), (75, 2686), (74, 2929), (73, 3126), (72, 3490), (71, 3657), (70, 4098), (69, 4449), (68, 4756), (67, 5061), (66, 5529), (65, 6028), (64, 6545), (63, 7198), (62, 7723), (61, 8621), (60, 9866), (59, 10771), (58, 11773), (57, 11776), (56, 12672), (55, 13951), (54, 15069), (53, 15959), (52, 17481), (51, 18943), (50, 20747), (49, 22558), (48, 24116), (47, 26549), (46, 28347), (45, 30953), (44, 33673), (43, 35959), (42, 38732), (41, 41909), (40, 46090), (39, 49991), (38, 53894), (37, 56454), (36, 60835), (35, 65440), (34, 70125), (33, 75182), (32, 81015), (31, 86883), (30, 91586), (29, 98251), (28, 104117), (27, 111076), (26, 118207), (25, 123028), (24, 128857), (23, 134906), (22, 140894), (21, 146299), (20, 152030), (19, 155026), (18, 159080), (17, 162654), (16, 165532), (15, 164658), (14, 162805), (13, 156485), (12, 150160), (11, 142053), (10, 134701), (9, 123547), (8, 97630), (7, 74183), (6, 33878), (5, 21733), (4, 8039), (3, 9064), (2, 2468), (1, 5062)]
    seq length <=64 num: 4343134
    
    验证数据(newstest2013): 
    N_valid = 2975 ( 源句子-目标句子 pair 的数目, 过滤掉长度大于 64 ) 
    
    most common seq length: (seq length, count num)
    [(20, 121), (13, 120), (14, 115), (15, 115), (16, 115), (11, 108), (19, 106), (21, 103), (18, 103), (17, 103)]
    seq length count:  (seq length, count num)
    [(106, 1), (92, 1), (91, 1), (90, 1), (88, 1), (86, 1), (80, 2), (79, 1), (78, 1), (76, 1), (75, 1), (74, 2), (72, 1), (70, 2), (68, 1), (66, 2), (64, 3), (63, 4), (62, 6), (61, 2), (60, 4), (59, 5), (58, 4), (57, 3), (56, 3), (55, 6), (54, 9), (53, 13), (52, 8), (51, 12), (50, 10), (49, 14), (48, 17), (47, 10), (46, 14), (45, 15), (44, 10), (43, 16), (42, 13), (41, 19), (40, 33), (39, 22), (38, 24), (37, 27), (36, 40), (35, 34), (34, 45), (33, 61), (32, 45), (31, 49), (30, 53), (29, 65), (28, 58), (27, 69), (26, 79), (25, 73), (24, 81), (23, 88), (22, 98), (21, 103), (20, 121), (19, 106), (18, 103), (17, 103), (16, 115), (15, 115), (14, 115), (13, 120), (12, 96), (11, 108), (10, 98), (9, 89), (8, 89), (7, 68), (6, 67), (5, 37), (4, 24), (3, 14), (2, 18), (1, 7)]
    target sentence length distribution:
    most common seq length: (seq length, count num)
    [(13, 129), (15, 122), (16, 117), (11, 115), (12, 111), (20, 108), (14, 107), (10, 106), (19, 105), (18, 104)]
    seq length count:  (seq length, count num)
    [(103, 1), (98, 1), (95, 2), (92, 1), (87, 1), (85, 1), (84, 1), (83, 1), (81, 1), (80, 1), (79, 1), (77, 1), (74, 1), (72, 1), (71, 1), (69, 3), (67, 2), (66, 2), (64, 2), (63, 5), (62, 3), (61, 3), (60, 3), (59, 3), (58, 4), (57, 8), (56, 3), (55, 4), (54, 5), (53, 7), (52, 7), (51, 13), (50, 4), (49, 7), (48, 9), (47, 9), (46, 16), (45, 20), (44, 12), (43, 16), (42, 21), (41, 26), (40, 30), (39, 30), (38, 25), (37, 29), (36, 32), (35, 38), (34, 34), (33, 53), (32, 37), (31, 54), (30, 62), (29, 61), (28, 77), (27, 67), (26, 62), (25, 64), (24, 94), (23, 78), (22, 80), (21, 89), (20, 108), (19, 105), (18, 104), (17, 97), (16, 117), (15, 122), (14, 107), (13, 129), (12, 111), (11, 115), (10, 106), (9, 93), (8, 80), (7, 95), (6, 64), (5, 54), (4, 21), (3, 19), (2, 17), (1, 7)]
    seq length <=64 num: 2975

    测试数据(newstest2014): 
    N_test = 2737 
    
    most common seq length: (seq length, count num)
    [(13, 113), (14, 112), (18, 112), (16, 101), (17, 100), (15, 100), (20, 95), (19, 94), (23, 94), (21, 92)]
    seq length count:  (seq length, count num)
    [(91, 1), (83, 1), (79, 1), (72, 2), (69, 1), (68, 2), (64, 2), (63, 2), (62, 1), (61, 2), (59, 2), (58, 9), (57, 2), (56, 4), (55, 7), (54, 2), (53, 4), (52, 7), (51, 11), (50, 14), (49, 8), (48, 12), (47, 14), (46, 15), (45, 15), (44, 26), (43, 29), (42, 21), (41, 14), (40, 29), (39, 33), (38, 35), (37, 35), (36, 44), (35, 41), (34, 46), (33, 47), (32, 53), (31, 66), (30, 63), (29, 59), (28, 73), (27, 77), (26, 58), (25, 85), (24, 86), (23, 94), (22, 92), (21, 92), (20, 95), (19, 94), (18, 112), (17, 100), (16, 101), (15, 100), (14, 112), (13, 113), (12, 82), (11, 81), (10, 73), (9, 69), (8, 52), (7, 50), (6, 36), (5, 15), (4, 7), (3, 4), (2, 2)]
    target sentence length distribution:
    most common seq length: (seq length, count num)
    [(19, 125), (14, 119), (15, 118), (13, 105), (17, 105), (12, 103), (10, 101), (16, 100), (20, 98), (21, 96)]
    seq length count:  (seq length, count num)
    [(75, 1), (72, 2), (70, 1), (68, 1), (64, 1), (63, 2), (60, 3), (59, 3), (58, 1), (57, 3), (56, 5), (55, 3), (54, 2), (53, 9), (52, 7), (51, 2), (50, 8), (49, 7), (48, 12), (47, 11), (46, 14), (45, 20), (44, 15), (43, 17), (42, 25), (41, 19), (40, 28), (39, 23), (38, 19), (37, 27), (36, 32), (35, 43), (34, 37), (33, 41), (32, 58), (31, 48), (30, 67), (29, 56), (28, 66), (27, 63), (26, 71), (25, 71), (24, 64), (23, 89), (22, 84), (21, 96), (20, 98), (19, 125), (18, 79), (17, 105), (16, 100), (15, 118), (14, 119), (13, 105), (12, 103), (11, 92), (10, 101), (9, 87), (8, 60), (7, 73), (6, 46), (5, 28), (4, 14), (3, 6), (2, 1)]
    seq length <=100 num: 2737
    
    (2) 数据预处理
    
    未做 unicode 标准化
    
    使用 wordpiece subword 算法分词
    源语言词表大小: n_vocab_source=30000
    目标语言词表大小: n_vocab_target=30000
    
    
    (3) 优化器参数
    
    epoch_num = 10
    token_in_batch = 12288
    
    label_smoothing=0 (未使用 Label Smoothing)
    
    optimizer= Adam with warmup_steps
    warmup_steps = 32000
    
    (5) 训练过程
    
    在每一个 epoch 结束时都对模型进行持久化(checkpoint), 并计算在验证集上的 bleu 得分

    Epoch 1/20
    
    candidates:
    [0] Eine Strategie für die Bush # # AT # # - # # AT # # Strategie , die sich die Wahl der Präsidenten der Präsidenten zu einer Wahl zu gewinnen hat , ist eine Antwort .
    [1] Die Regierung hat sich gegen die politische Politik der Bekämpfung von Betrug ausgesprochen .
    [2] Aber das Bruton Brüchtliche Brüchtlings ist ein Zeichen , dass die Regierung in den USA in den USA nur wenige Menschen von den Menschen getötet wird .
    [3] Tatsächlich sind die britischen Konservativen nur in den USA nur wenige Jahre in den USA ein paar Jahre in einem Jahr .
    [4] Es ist ein Grund , dass diese neuen Bestimmungen eine neue Gefahr haben werden , um die Auswirkungen auf die Auswirkungen auf die Auswirkungen zu haben .
    [5] In diesem Sinne wird die Maßnahmen der NATO die demokratische Kontrolle der demokratischen Souveränität gefährden .
    [6] In Großbritannien sind die USA in der USA die USA für die USA verantwortlich .
    [7] Es ist in diesem Sinne in der Tat ein Mehrheits - und Regierungschefs , die seit langem eine neue oder mehr bessere Lösung haben .
    [8] Diese „ globale “ Ereignisse in den vergangenen Wochen wurden am 25 . Juni in den USA aufgenommen .
    [9] Die Verwährung von 30 % der Todesstrafe in den Mitgliedstaaten wurde in den letzten Jahren abgelehnt .
    
    references:
    [0] ['Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzutreten']
    [1] ['Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit , den Wahlbetrug zu bekämpfen .']
    [2] ['Allerdings hält das Brennan Center letzteres für einen Mythos , indem es bekräftigt , dass der Wahlbetrug in den USA seltener ist als die Anzahl der vom Blitzschlag getöteten Menschen .']
    [3] ['Die Rechtsanwälte der Republikaner haben in 10 Jahren in den USA übrigens nur 300 Fälle von Wahlbetrug verzeichnet .']
    [4] ['Eins ist sicher : diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .']
    [5] ['In diesem Sinne untergraben diese Maßnahmen teilweise das demokratische System der USA .']
    [6] ['Im Gegensatz zu Kanada sind die US ##AT##-##AT## Bundesstaaten für die Durchführung der Wahlen in den einzelnen Staaten verantwortlich .']
    [7] ['In diesem Sinne hat die Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verkündet , die das Verfahren für die Registrierung oder den Urnengang erschweren .']
    [8] ['Dieses Phänomen hat nach den Wahlen vom November 2010 an Bedeutung gewonnen , bei denen 675 neue republikanische Vertreter in 26 Staaten verzeichnet werden konnten .']
    [9] ['Infolgedessen wurden 180 Gesetzesentwürfe allein im Jahr 2011 eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .']
    
    bleu_score:{'1-garm': 0.26854159083909623, '2-garm': 0.11392188693876013, '3-garm': 0.05229728013420901, '4-garm': 0.02507881902088424}
    
    14308/14308 [==============================] - 5006s 349ms/step - loss: 6.0814 - accuracy: 0.1654 - val_loss: 3.6112 - val_accuracy: 0.3595
    
    Epoch 2/20
    
    candidates:
    [0] Eine republikanische Strategie zur Gegenwahl von Obama .
    [1] Die republikanischen Führer haben ihre Politik durch die Notwendigkeit , die Wahlbetrug zu bekämpfen .
    [2] Doch das Brennan Centre hält diesen Mythos für einen Mythos , der besagt , dass die Wahlbetrug in den USA selten ist , als die Zahl der Menschen , die durch Blitz getötet wurden .
    [3] Tatsächlich identifizierte republikanische Anwälte nur 300 Fälle von Wahlbetrug in den USA in einem Jahrzehnt .
    [4] Eines ist sicher : Diese neuen Bestimmungen werden negative Auswirkungen auf die Wähler # # AT # # - # # AT # # Abrufung haben .
    [5] In diesem Sinne werden die Maßnahmen zum Teil das amerikanische demokratische System untergraben .
    [6] Im Gegensatz zu Kanada sind die USA für die Organisation der Bundeswahlen in den USA verantwortlich .
    [7] In diesem Sinne haben die meisten amerikanischen Regierungen seit 2009 neue Gesetze verabschiedet , um den Registrierungs - oder Abstimmungsprozess schwieriger zu gestalten .
    [8] Dieses Phänomen hat nach den Wahlen im November 2010 Dynamik gewonnen , die 675 neue republikanische Vertreter in 26 Staaten aufgenommen haben .
    [9] Damit wurden im Jahr 2011 180 Rechnungen die Ausübung des Rechts auf Abstimmung in 41 Staaten eingeschränkt .
    
    references:
    [0] ['Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzutreten']
    [1] ['Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit , den Wahlbetrug zu bekämpfen .']
    [2] ['Allerdings hält das Brennan Center letzteres für einen Mythos , indem es bekräftigt , dass der Wahlbetrug in den USA seltener ist als die Anzahl der vom Blitzschlag getöteten Menschen .']
    [3] ['Die Rechtsanwälte der Republikaner haben in 10 Jahren in den USA übrigens nur 300 Fälle von Wahlbetrug verzeichnet .']
    [4] ['Eins ist sicher : diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .']
    [5] ['In diesem Sinne untergraben diese Maßnahmen teilweise das demokratische System der USA .']
    [6] ['Im Gegensatz zu Kanada sind die US ##AT##-##AT## Bundesstaaten für die Durchführung der Wahlen in den einzelnen Staaten verantwortlich .']
    [7] ['In diesem Sinne hat die Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verkündet , die das Verfahren für die Registrierung oder den Urnengang erschweren .']
    [8] ['Dieses Phänomen hat nach den Wahlen vom November 2010 an Bedeutung gewonnen , bei denen 675 neue republikanische Vertreter in 26 Staaten verzeichnet werden konnten .']
    [9] ['Infolgedessen wurden 180 Gesetzesentwürfe allein im Jahr 2011 eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .']
    
    bleu_score:{'1-garm': 0.5097037435063259, '2-garm': 0.345818447938025, '3-garm': 0.24764333562235868, '4-garm': 0.18167533217234233}
    
    14308/14308 [==============================] - 4952s 346ms/step - loss: 2.9274 - accuracy: 0.4264 - val_loss: 2.0009 - val_accuracy: 0.5921
    
    Epoch 3/20
    
    candidates:
    [0] Eine republikanische Strategie zur Bekämpfung der Wiederwahl von Obama .
    [1] Die republikanischen Führer haben ihre Politik durch die Notwendigkeit der Bekämpfung von Wahlbetrug begründet .
    [2] Das Brennan Centre ist jedoch der Ansicht , dass ein Mythos , der besagt , dass Wahlbetrug in den USA selten ist als die Anzahl der Menschen , die durch Blitz getötet wurden .
    [3] Tatsächlich haben republikanische Anwälte in den USA in einem Jahrzehnt nur 300 Fälle von Wahlbetrug festgestellt .
    [4] Eines ist sicher : Diese neuen Bestimmungen werden negative Auswirkungen auf die Wähler # # AT # # - # # AT # # Wende haben .
    [5] In diesem Sinne werden die Maßnahmen das amerikanische demokratische System teilweise untergraben .
    [6] Im Gegensatz zu Kanada sind die USA für die Organisation der Bundeswahlen in den USA verantwortlich .
    [7] In diesem Sinne hat eine Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze erlassen , die die Registrierung oder Abstimmung erschwert haben .
    [8] Dieses Phänomen hat sich nach den Wahlen im November 2010 , die 675 neue republikanische Vertreter in 26 Staaten aufgenommen sahen , gewandelt .
    [9] So wurden allein im Jahr 2011 180 Rechnungen eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .
    
    references:
    [0] ['Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzutreten']
    [1] ['Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit , den Wahlbetrug zu bekämpfen .']
    [2] ['Allerdings hält das Brennan Center letzteres für einen Mythos , indem es bekräftigt , dass der Wahlbetrug in den USA seltener ist als die Anzahl der vom Blitzschlag getöteten Menschen .']
    [3] ['Die Rechtsanwälte der Republikaner haben in 10 Jahren in den USA übrigens nur 300 Fälle von Wahlbetrug verzeichnet .']
    [4] ['Eins ist sicher : diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .']
    [5] ['In diesem Sinne untergraben diese Maßnahmen teilweise das demokratische System der USA .']
    [6] ['Im Gegensatz zu Kanada sind die US ##AT##-##AT## Bundesstaaten für die Durchführung der Wahlen in den einzelnen Staaten verantwortlich .']
    [7] ['In diesem Sinne hat die Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verkündet , die das Verfahren für die Registrierung oder den Urnengang erschweren .']
    [8] ['Dieses Phänomen hat nach den Wahlen vom November 2010 an Bedeutung gewonnen , bei denen 675 neue republikanische Vertreter in 26 Staaten verzeichnet werden konnten .']
    [9] ['Infolgedessen wurden 180 Gesetzesentwürfe allein im Jahr 2011 eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .']
    
    bleu_score:{'1-garm': 0.533574145673907, '2-garm': 0.37194651188273836, '3-garm': 0.2730918552639083, '4-garm': 0.20502353090154635}
    
    14308/14308 [==============================] - 4953s 346ms/step - loss: 2.0505 - accuracy: 0.5617 - val_loss: 1.7286 - val_accuracy: 0.6338
    
    Epoch 4/20
    
    candidates:
    [0] Eine republikanische Strategie zur Bekämpfung der Wiederwahl von Obama
    [1] Die republikanischen Führer haben ihre Politik durch die Notwendigkeit gerechtfertigt , gegen Wahlbetrug vorzugehen .
    [2] Das Brennan Centre betrachtet dies jedoch als Mythos , in dem es heißt , dass Wahlbetrug in den Vereinigten Staaten selten ist als die Anzahl der durch Blitz getöteten Menschen .
    [3] Tatsächlich haben republikanische Anwälte nur 300 Fälle von Wahlbetrug in den USA in einem Jahrzehnt festgestellt .
    [4] Eines ist sicher : Diese neuen Bestimmungen werden negative Auswirkungen auf die Wahlbeteiligung haben .
    [5] In diesem Sinne werden die Maßnahmen das amerikanische demokratische System teilweise untergraben .
    [6] Im Gegensatz zu Kanada sind die USA für die Durchführung von Bundeswahlen in den Vereinigten Staaten verantwortlich .
    [7] In diesem Sinne haben die meisten amerikanischen Regierungen seit 2009 neue Gesetze verabschiedet , wodurch die Registrierung oder der Abstimmungsprozess schwieriger werden .
    [8] Dieses Phänomen gewann nach den Wahlen vom November 2010 , die 675 neue republikanische Vertreter in 26 Staaten hinzugefügt .
    [9] Infolgedessen wurden allein im Jahr 2011 180 Rechnungen eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .
    
    references:
    [0] ['Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzutreten']
    [1] ['Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit , den Wahlbetrug zu bekämpfen .']
    [2] ['Allerdings hält das Brennan Center letzteres für einen Mythos , indem es bekräftigt , dass der Wahlbetrug in den USA seltener ist als die Anzahl der vom Blitzschlag getöteten Menschen .']
    [3] ['Die Rechtsanwälte der Republikaner haben in 10 Jahren in den USA übrigens nur 300 Fälle von Wahlbetrug verzeichnet .']
    [4] ['Eins ist sicher : diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .']
    [5] ['In diesem Sinne untergraben diese Maßnahmen teilweise das demokratische System der USA .']
    [6] ['Im Gegensatz zu Kanada sind die US ##AT##-##AT## Bundesstaaten für die Durchführung der Wahlen in den einzelnen Staaten verantwortlich .']
    [7] ['In diesem Sinne hat die Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verkündet , die das Verfahren für die Registrierung oder den Urnengang erschweren .']
    [8] ['Dieses Phänomen hat nach den Wahlen vom November 2010 an Bedeutung gewonnen , bei denen 675 neue republikanische Vertreter in 26 Staaten verzeichnet werden konnten .']
    [9] ['Infolgedessen wurden 180 Gesetzesentwürfe allein im Jahr 2011 eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .']
    
    bleu_score:{'1-garm': 0.5479583262802901, '2-garm': 0.38666465626219776, '3-garm': 0.2865508102130119, '4-garm': 0.21715090503146514}
    
    14308/14308 [==============================] - 4874s 340ms/step - loss: 1.8312 - accuracy: 0.5950 - val_loss: 1.6167 - val_accuracy: 0.6548
    
    Epoch 5/20
    
    candidates:
    [0] Eine republikanische Strategie zur Bekämpfung der Wiederwahl Obama .
    [1] Die Republikaner haben ihre Politik durch die Notwendigkeit der Bekämpfung von Wahlbetrug begründet .
    [2] Das Brennan Centre hält dies jedoch für einen Mythos , der besagt , dass Wahlbetrug in den USA selten ist als die Zahl der Menschen , die durch Blitze getötet werden .
    [3] Tatsächlich haben republikanische Anwälte in den USA innerhalb eines Jahrzehnts lediglich 300 Fälle von Wahlbetrug festgestellt .
    [4] Eines ist sicher : Diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .
    [5] In diesem Sinne werden die Maßnahmen zum Teil das demokratische System der USA untergraben .
    [6] Anders als in Kanada sind die USA für die Organisation von Bundeswahlen in den Vereinigten Staaten verantwortlich .
    [7] In diesem Sinne hat eine Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verabschiedet , die die Registrierung oder den Abstimmungsprozess schwieriger machen .
    [8] Dieses Phänomen hat nach den Wahlen im November 2010 , wo in 26 Staaten 675 neue republikanische Vertreter hinzukamen , einen Schwung bekommen .
    [9] Infolgedessen wurden allein 2011 180 Rechnungen eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .
    
    references:
    [0] ['Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzutreten']
    [1] ['Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit , den Wahlbetrug zu bekämpfen .']
    [2] ['Allerdings hält das Brennan Center letzteres für einen Mythos , indem es bekräftigt , dass der Wahlbetrug in den USA seltener ist als die Anzahl der vom Blitzschlag getöteten Menschen .']
    [3] ['Die Rechtsanwälte der Republikaner haben in 10 Jahren in den USA übrigens nur 300 Fälle von Wahlbetrug verzeichnet .']
    [4] ['Eins ist sicher : diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .']
    [5] ['In diesem Sinne untergraben diese Maßnahmen teilweise das demokratische System der USA .']
    [6] ['Im Gegensatz zu Kanada sind die US ##AT##-##AT## Bundesstaaten für die Durchführung der Wahlen in den einzelnen Staaten verantwortlich .']
    [7] ['In diesem Sinne hat die Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verkündet , die das Verfahren für die Registrierung oder den Urnengang erschweren .']
    [8] ['Dieses Phänomen hat nach den Wahlen vom November 2010 an Bedeutung gewonnen , bei denen 675 neue republikanische Vertreter in 26 Staaten verzeichnet werden konnten .']
    [9] ['Infolgedessen wurden 180 Gesetzesentwürfe allein im Jahr 2011 eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .']
    
    bleu_score:{'1-garm': 0.5491877236167895, '2-garm': 0.38985865692517785, '3-garm': 0.2915026703275076, '4-garm': 0.22281865990661862}
    
    14308/14308 [==============================] - 4955s 346ms/step - loss: 1.7195 - accuracy: 0.6128 - val_loss: 1.5558 - val_accuracy: 0.6620
    
    Epoch 6/20
    
    candidates:
    [0] Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzuwirken .
    [1] Die republikanischen Machthaber haben ihre Politik mit der Notwendigkeit , Wahlbetrug zu bekämpfen , gerechtfertigt .
    [2] Das Brennan Centre sieht dies jedoch als einen Mythos an , da es in den USA seltener Wahlbetrug als die Anzahl der durch Blitze getöteten Menschen ist .
    [3] Tatsächlich haben republikanische Anwälte in einem Jahrzehnt nur 300 Fälle von Wahlbetrug in den Vereinigten Staaten festgestellt .
    [4] Eines ist sicher : Diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .
    [5] In diesem Sinne werden die Maßnahmen das demokratische System der Amerikaner teilweise unterminieren .
    [6] Im Gegensatz zu Kanada sind die USA für die Durchführung von Bundeswahlen in den Vereinigten Staaten verantwortlich .
    [7] In diesem Sinne hat eine Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verabschiedet , wodurch der Registrierungs - oder Wahlprozess erschwert wird .
    [8] Dieses Phänomen gewann nach den Wahlen vom November 2010 , die 675 neue republikanische Vertreter in 26 Staaten hinzugefügt .
    [9] Infolgedessen wurden allein 2011 180 Rechnungen eingeführt , die die Ausübung des Wahlrechts in 41 Staaten beschränken .
    
    references:
    [0] ['Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzutreten']
    [1] ['Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit , den Wahlbetrug zu bekämpfen .']
    [2] ['Allerdings hält das Brennan Center letzteres für einen Mythos , indem es bekräftigt , dass der Wahlbetrug in den USA seltener ist als die Anzahl der vom Blitzschlag getöteten Menschen .']
    [3] ['Die Rechtsanwälte der Republikaner haben in 10 Jahren in den USA übrigens nur 300 Fälle von Wahlbetrug verzeichnet .']
    [4] ['Eins ist sicher : diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .']
    [5] ['In diesem Sinne untergraben diese Maßnahmen teilweise das demokratische System der USA .']
    [6] ['Im Gegensatz zu Kanada sind die US ##AT##-##AT## Bundesstaaten für die Durchführung der Wahlen in den einzelnen Staaten verantwortlich .']
    [7] ['In diesem Sinne hat die Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verkündet , die das Verfahren für die Registrierung oder den Urnengang erschweren .']
    [8] ['Dieses Phänomen hat nach den Wahlen vom November 2010 an Bedeutung gewonnen , bei denen 675 neue republikanische Vertreter in 26 Staaten verzeichnet werden konnten .']
    [9] ['Infolgedessen wurden 180 Gesetzesentwürfe allein im Jahr 2011 eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .']
    
    bleu_score:{'1-garm': 0.554298097760655, '2-garm': 0.39739963876830436, '3-garm': 0.2991375821755899, '4-garm': 0.23033802605002227}
    
    14308/14308 [==============================] - 4990s 348ms/step - loss: 1.6714 - accuracy: 0.6201 - val_loss: 1.5183 - val_accuracy: 0.6692
    
    Epoch 7/20
    
    candidates:
    [0] Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzuwirken .
    [1] Die republikanischen Führer rechtfertigten ihre Politik durch die Notwendigkeit , Wahlbetrug zu bekämpfen .
    [2] Das Brennan Centre hält dies jedoch für einen Mythos , der besagt , dass Wahlbetrug in den USA selten ist als die Zahl der durch Blitz getöteten Menschen .
    [3] Tatsächlich identifizierten republikanische Anwälte in den USA in einem Jahrzehnt nur 300 Fälle von Wahlbetrug .
    [4] Eines ist sicher : Diese neuen Bestimmungen werden negative Auswirkungen auf die Wahlbeteiligung haben .
    [5] In diesem Sinne werden die Maßnahmen teilweise das demokratische amerikanische System untergraben .
    [6] Im Gegensatz zu Kanada sind die USA für die Organisation von Bundeswahlen in den Vereinigten Staaten verantwortlich .
    [7] In diesem Sinne hat eine Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verabschiedet , die die Registrierung oder den Abstimmungsprozess erschwert haben .
    [8] Dieses Phänomen gewann nach den Wahlen vom November 2010 , die neue Vertreter der Republikanischen 675 in 26 Staaten hinzugefügt .
    [9] Infolgedessen wurden allein 2011 180 Rechnungen eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .
    
    references:
    [0] ['Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzutreten']
    [1] ['Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit , den Wahlbetrug zu bekämpfen .']
    [2] ['Allerdings hält das Brennan Center letzteres für einen Mythos , indem es bekräftigt , dass der Wahlbetrug in den USA seltener ist als die Anzahl der vom Blitzschlag getöteten Menschen .']
    [3] ['Die Rechtsanwälte der Republikaner haben in 10 Jahren in den USA übrigens nur 300 Fälle von Wahlbetrug verzeichnet .']
    [4] ['Eins ist sicher : diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .']
    [5] ['In diesem Sinne untergraben diese Maßnahmen teilweise das demokratische System der USA .']
    [6] ['Im Gegensatz zu Kanada sind die US ##AT##-##AT## Bundesstaaten für die Durchführung der Wahlen in den einzelnen Staaten verantwortlich .']
    [7] ['In diesem Sinne hat die Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verkündet , die das Verfahren für die Registrierung oder den Urnengang erschweren .']
    [8] ['Dieses Phänomen hat nach den Wahlen vom November 2010 an Bedeutung gewonnen , bei denen 675 neue republikanische Vertreter in 26 Staaten verzeichnet werden konnten .']
    [9] ['Infolgedessen wurden 180 Gesetzesentwürfe allein im Jahr 2011 eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .']
    
    bleu_score:{'1-garm': 0.5519273515042561, '2-garm': 0.39433335749918846, '3-garm': 0.29585712023534977, '4-garm': 0.22683082941569513}
    
    14308/14308 [==============================] - 5480s 383ms/step - loss: 1.6207 - accuracy: 0.6284 - val_loss: 1.5041 - val_accuracy: 0.6709
    
    Epoch 8/20
    
    candidates:
    [0] Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzuwirken .
    [1] Die republikanischen Führer haben ihre Politik durch die Notwendigkeit der Bekämpfung von Wahlbetrug gerechtfertigt .
    [2] Das Brennan Centre hält dies jedoch für einen Mythos , der besagt , dass Wahlbetrug in den Vereinigten Staaten selten vorkommt , als die Anzahl der durch Blitze getöteten Menschen .
    [3] Tatsächlich haben republikanische Anwälte in den USA in einem Jahrzehnt nur 300 Fälle von Wahlbetrug festgestellt .
    [4] Eines ist sicher : Diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .
    [5] In diesem Sinne werden die Maßnahmen das demokratische amerikanische System teilweise untergraben .
    [6] Im Gegensatz zu Kanada sind die USA für die Durchführung von Bundeswahlen in den Vereinigten Staaten verantwortlich .
    [7] In diesem Sinne hat eine Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verabschiedet , die die Registrierung oder den Wahlprozess erschweren .
    [8] Dieses Phänomen gewann nach den Wahlen vom November 2010 , die 675 neue republikanische Vertreter in 26 Staaten hinzugefügt .
    [9] Infolgedessen wurden allein im Jahr 2011 180 Rechnungen eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .
    
    references:
    [0] ['Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzutreten']
    [1] ['Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit , den Wahlbetrug zu bekämpfen .']
    [2] ['Allerdings hält das Brennan Center letzteres für einen Mythos , indem es bekräftigt , dass der Wahlbetrug in den USA seltener ist als die Anzahl der vom Blitzschlag getöteten Menschen .']
    [3] ['Die Rechtsanwälte der Republikaner haben in 10 Jahren in den USA übrigens nur 300 Fälle von Wahlbetrug verzeichnet .']
    [4] ['Eins ist sicher : diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .']
    [5] ['In diesem Sinne untergraben diese Maßnahmen teilweise das demokratische System der USA .']
    [6] ['Im Gegensatz zu Kanada sind die US ##AT##-##AT## Bundesstaaten für die Durchführung der Wahlen in den einzelnen Staaten verantwortlich .']
    [7] ['In diesem Sinne hat die Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verkündet , die das Verfahren für die Registrierung oder den Urnengang erschweren .']
    [8] ['Dieses Phänomen hat nach den Wahlen vom November 2010 an Bedeutung gewonnen , bei denen 675 neue republikanische Vertreter in 26 Staaten verzeichnet werden konnten .']
    [9] ['Infolgedessen wurden 180 Gesetzesentwürfe allein im Jahr 2011 eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .']
    
    bleu_score:{'1-garm': 0.5557299822370624, '2-garm': 0.3988805394362578, '3-garm': 0.30130934499429657, '4-garm': 0.23290716632051853}
    
    14308/14308 [==============================] - 4901s 342ms/step - loss: 1.5905 - accuracy: 0.6333 - val_loss: 1.4733 - val_accuracy: 0.6741
    
    Epoch 9/20
    
    candidates:
    [0] Eine republikanische Strategie gegen die Wiederwahl von Obama .
    [1] Die republikanischen Führer haben ihre Politik mit der Notwendigkeit , Wahlbetrug zu bekämpfen , gerechtfertigt .
    [2] Das Brennan Centre hält dies jedoch für einen Mythos , der besagt , dass Wahlbetrug in den USA selten höher ist als die Zahl der durch Blitzkatastrophe getöteten Menschen .
    [3] Tatsächlich identifizierten republikanische Anwälte in den USA in einem Jahrzehnt nur 300 Fälle von Wahlbetrug .
    [4] Eines ist sicher : Diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .
    [5] In diesem Sinne werden die Maßnahmen das amerikanische demokratische System teilweise untergraben .
    [6] Im Gegensatz zu Kanada sind die USA für die Organisation von föderalen Wahlen in den Vereinigten Staaten verantwortlich .
    [7] In diesem Sinne hat die Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verabschiedet , die die Registrierung oder den Abstimmungsprozess erschweren .
    [8] Dieses Phänomen gewann nach den Wahlen vom November 2010 , die 675 neue republikanische Vertreter in 26 Staaten hinzufügten , an Schwung .
    [9] Infolgedessen wurden allein im Jahr 2011 180 Rechnungen eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .
    
    references:
    [0] ['Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzutreten']
    [1] ['Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit , den Wahlbetrug zu bekämpfen .']
    [2] ['Allerdings hält das Brennan Center letzteres für einen Mythos , indem es bekräftigt , dass der Wahlbetrug in den USA seltener ist als die Anzahl der vom Blitzschlag getöteten Menschen .']
    [3] ['Die Rechtsanwälte der Republikaner haben in 10 Jahren in den USA übrigens nur 300 Fälle von Wahlbetrug verzeichnet .']
    [4] ['Eins ist sicher : diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .']
    [5] ['In diesem Sinne untergraben diese Maßnahmen teilweise das demokratische System der USA .']
    [6] ['Im Gegensatz zu Kanada sind die US ##AT##-##AT## Bundesstaaten für die Durchführung der Wahlen in den einzelnen Staaten verantwortlich .']
    [7] ['In diesem Sinne hat die Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verkündet , die das Verfahren für die Registrierung oder den Urnengang erschweren .']
    [8] ['Dieses Phänomen hat nach den Wahlen vom November 2010 an Bedeutung gewonnen , bei denen 675 neue republikanische Vertreter in 26 Staaten verzeichnet werden konnten .']
    [9] ['Infolgedessen wurden 180 Gesetzesentwürfe allein im Jahr 2011 eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .']
    
    bleu_score:{'1-garm': 0.5552845658906022, '2-garm': 0.40055386156363215, '3-garm': 0.30344291469508233, '4-garm': 0.23474556361540716}
    
    14308/14308 [==============================] - 4880s 341ms/step - loss: 1.5631 - accuracy: 0.6378 - val_loss: 1.4638 - val_accuracy: 0.6757
    
        
    (6) 模型评价
    
    在测试集上评价模型
    
    1.epoch=9 时的模型
    
    candidates:
    [0] Orlando Bloom und Miranda Kerr lieben sich noch immer .
    [1] Schauspieler Orlando Bloom und Model Miranda Kerr wollen ihre eigene Wege gehen .
    [2] Bloom und Kerr haben sich jedoch in einem Interview noch immer liebend gut aufgehoben .
    [3] Miranda Kerr und Orlando Bloom sind Eltern von zwei Jahren Flynn .
    [4] Der Schauspieler Orlando Bloom kündigte seine Trennung von seiner Frau , Supermodel Miranda Kerr .
    [5] In einem Interview mit dem US # # AT # # - # # AT # # Journalist Katie Couric , der am Freitag ( Ortszeit ) ausgestrahlt wird , sagte Bloom : & quot ; Manchmal geht das Leben nicht so , wie wir es planen oder hoffen & quot ; .
    [6] Er und Kerr lieben sich noch immer , betonte das 36 # # AT # # - # # AT # # jährige .
    [7] & quot ; Wir werden uns gegenseitig unterstützen und lieben einander als Eltern zu Flynn & quot ; .
    [8] Kerr und Bloom sind seit 2010 verheiratet und ihr Sohn Flynn wurde im Jahr 2011 geboren .
    [9] Jet # # AT # # - # # AT # # Hersteller feuerten über Sitzbreite mit großen Aufträgen auf dem Spiel .
    
    references:
    [0] ['Orlando Bloom und Miranda Kerr lieben sich noch immer']
    [1] ['Schauspieler Orlando Bloom und Model Miranda Kerr wollen künftig getrennte Wege gehen .']
    [2] ['In einem Interview sagte Bloom jedoch , dass er und Kerr sich noch immer lieben .']
    [3] ['Miranda Kerr und Orlando Bloom sind Eltern des zweijährigen Flynn .']
    [4] ['Schauspieler Orlando Bloom hat sich zur Trennung von seiner Frau , Topmodel Miranda Kerr , geäußert .']
    [5] ['In einem Interview mit US ##AT##-##AT## Journalistin Katie Couric , das am Freitag ( Ortszeit ) ausgestrahlt werden sollte , sagte Bloom , &quot; das Leben verläuft manchmal nicht genau so , wie wir es planen oder erhoffen &quot; .']
    [6] ['Kerr und er selbst liebten sich noch immer , betonte der 36 ##AT##-##AT## Jährige .']
    [7] ['&quot; Wir werden uns gegenseitig unterstützen und lieben als Eltern von Flynn &quot; .']
    [8] ['Kerr und Bloom sind seit 2010 verheiratet , im Jahr 2011 wurde ihr Söhnchen Flynn geboren .']
    [9] ['Jumbo ##AT##-##AT## Hersteller streiten im Angesicht großer Bestellungen über Sitzbreite']
    
    bleu_score:{'1-garm': 0.5395491205908075, '2-garm': 0.3885112483113567, '3-garm': 0.2922480214446967, '4-garm': 0.22472565783740672}
        
        
#### 实验 2  warmup_steps=8000
    
    (0) 模型
       
    (1) 数据集 
    
    训练数据:    
    N_train = 4343134 ( 源句子-目标句子 pair 的数目, 过滤掉长度大于 64 )
    
    验证数据(newstest2013): 
    N_valid = 2975 ( 源句子-目标句子 pair 的数目, 过滤掉长度大于 64 ) 

    测试数据(newstest2014): 
    N_test = 2737 
    
    
    (2) 数据预处理
    
    未做 unicode 标准化
    
    使用 wordpiece subword 算法分词
    源语言词表大小: n_vocab_source=30000
    目标语言词表大小: n_vocab_target=30000
    
    
    (3) 优化器参数
    
    epoch_num = 10
    token_in_batch = 12288
    
    label_smoothing=0.1(开启 label_smoothing)
    
    optimizer= Adam with warmup_steps
    warmup_steps = 8000
    
    (5) 训练过程
    
    在每一个 epoch 结束时都对模型进行持久化(checkpoint), 并计算在验证集上的 bleu 得分
    
    Epoch 1/9
    
    candidates:
    [0] Eine weitere politische Strategie der Obama # # AT # # - # # AT # # Administration ist die Wiederaufnahme der Wahlen .
    [1] Die Wahl der Politiker ist eine politische Entscheidung , die die Bürger gegen die Sanktionen schützt .
    [2] In der Tat ist die Tatsache , dass die USA in der Regel eine große Anzahl von Fällen von Gewalt gegen die Bevölkerung in der Region haben .
    [3] In den letzten Jahren haben die meisten Regierungen nur eine einzige einzige einzige demokratische Opposition in der Welt .
    [4] Eine weitere Sorge besteht darin , dass die neuen Vorschriften nicht geändert werden .
    [5] In diesem Fall wird die demokratische Kontrolle der USA untergraben .
    [6] In der Regel ist die US # # AT # # - # # AT # # Regierung in den USA nicht in der Lage , die Todesstrafe zu finanzieren .
    [7] Es ist eine Schande , dass die neuen Demokratien in den USA eine neue Verfassung haben , die sich auf die Bekämpfung der Korruption und die Bekämpfung von Korruption stützt .
    [8] Die jüngsten Ereignisse , die in den letzten Tagen in der Regierung von New York am 27 . November in New York stattgefunden haben , sind ermutigend .
    [9] In diesem Fall hat die Entscheidung , die in den letzten drei Monaten in Kraft tretenden Sanktionen zu streichen , nur noch eine einzige Ausnahme für die in der Praxis verhängten Ausgaben von 30 % erreicht .
    
    references:
    [0] ['Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzutreten']
    [1] ['Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit , den Wahlbetrug zu bekämpfen .']
    [2] ['Allerdings hält das Brennan Center letzteres für einen Mythos , indem es bekräftigt , dass der Wahlbetrug in den USA seltener ist als die Anzahl der vom Blitzschlag getöteten Menschen .']
    [3] ['Die Rechtsanwälte der Republikaner haben in 10 Jahren in den USA übrigens nur 300 Fälle von Wahlbetrug verzeichnet .']
    [4] ['Eins ist sicher : diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .']
    [5] ['In diesem Sinne untergraben diese Maßnahmen teilweise das demokratische System der USA .']
    [6] ['Im Gegensatz zu Kanada sind die US ##AT##-##AT## Bundesstaaten für die Durchführung der Wahlen in den einzelnen Staaten verantwortlich .']
    [7] ['In diesem Sinne hat die Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verkündet , die das Verfahren für die Registrierung oder den Urnengang erschweren .']
    [8] ['Dieses Phänomen hat nach den Wahlen vom November 2010 an Bedeutung gewonnen , bei denen 675 neue republikanische Vertreter in 26 Staaten verzeichnet werden konnten .']
    [9] ['Infolgedessen wurden 180 Gesetzesentwürfe allein im Jahr 2011 eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .']
    
    bleu_score:{'1-garm': 0.2165214448862194, '2-garm': 0.07437178745623062, '3-garm': 0.028844275772728287, '4-garm': 0.011851486502152566}
    14308/14308 [==============================] - 5017s 350ms/step - loss: 5.0359 - accuracy: 0.2200 - val_loss: 3.2916 - val_accuracy: 0.3705
    
    Epoch 2/9
    
    candidates:
    [0] Obama sollte eine Reform der amerikanischen Wahlkampfpolitik vorschlagen .
    [1] Die Wähler haben die Wahl der Regierung Bush kritisiert , ihre Politik gegen die Militärregierung zu revolutionieren .
    [2] In der Tat ist die Tatsache , dass die USA einen solchen Fall als & quot ; vermeintliche & quot ; , als die Regierung des Landes die Selbstmordattentäter in den USA verschwand , ein Skandal , der sich jedoch als & quot ; vermeintlich & quot ; bezeichnet .
    [3] Tatsächlich haben die USA nur wenige Wochen lang die verschässigten Dissidenten in den USA wegen der Gewalt gegen die Hälfte der Fälle von Gewalt gegen Frauen verurteilt .
    [4] Eine weitere Änderung ist , dass diese Änderungen eine positive Diskriminierung darstellen , die sich negativ auf die neuen Demokratien auswirken wird .
    [5] Tatsächlich werden die Maßnahmen gegen den amerikanischen Protektionismus untergraben .
    [6] In den USA ist die Entscheidung der USA , die Mitglieder des Internationalen Bundes in den USA zu werden , in den USA jedoch in den USA .
    [7] Die Entscheidung , die seit den Wahlen in den USA getroffen wurde , ist seit Jahren demokratisch oder demokratisch , und die neuen Mitgliedstaaten haben sich seit Jahren gegen die Gesetzgebung der USA ausgesprochen .
    [8] Die Wahl der Regierung wurde am 26 . November 2009 in New York , New York , 26 . November 2009 , 26 . November 2009 , 26 . November 2009 , 26 . November 2009 , 26 . November 2009 , 26 . November 2009 , 26 . November 2009 , 26 . November 2009 , 26 . November 2009 , 26 . November 2009 , 26 . November 2009 .
    [9] Die Entscheidung , die in den einzelnen Mitgliedstaaten gemäß Artikel 180 Absatz 3 des EG # # AT # # - # # AT # # Vertrags getroffen wurde , wurde erst in den Monaten nach ihrem Berufungsverzug in den Jahren 2009 # # AT # # - # # AT # # 2013 getroffen .
    
    references:
    [0] ['Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzutreten']
    [1] ['Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit , den Wahlbetrug zu bekämpfen .']
    [2] ['Allerdings hält das Brennan Center letzteres für einen Mythos , indem es bekräftigt , dass der Wahlbetrug in den USA seltener ist als die Anzahl der vom Blitzschlag getöteten Menschen .']
    [3] ['Die Rechtsanwälte der Republikaner haben in 10 Jahren in den USA übrigens nur 300 Fälle von Wahlbetrug verzeichnet .']
    [4] ['Eins ist sicher : diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .']
    [5] ['In diesem Sinne untergraben diese Maßnahmen teilweise das demokratische System der USA .']
    [6] ['Im Gegensatz zu Kanada sind die US ##AT##-##AT## Bundesstaaten für die Durchführung der Wahlen in den einzelnen Staaten verantwortlich .']
    [7] ['In diesem Sinne hat die Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verkündet , die das Verfahren für die Registrierung oder den Urnengang erschweren .']
    [8] ['Dieses Phänomen hat nach den Wahlen vom November 2010 an Bedeutung gewonnen , bei denen 675 neue republikanische Vertreter in 26 Staaten verzeichnet werden konnten .']
    [9] ['Infolgedessen wurden 180 Gesetzesentwürfe allein im Jahr 2011 eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .']
    
    bleu_score:{'1-garm': 0.1516825800976024, '2-garm': 0.056800576479681404, '3-garm': 0.023305284159249585, '4-garm': 0.010303621170107158}
    14308/14308 [==============================] - 5013s 350ms/step - loss: 2.9222 - accuracy: 0.3947 - val_loss: 3.0991 - val_accuracy: 0.3945
    
    从第2个 Epoch 开始, 在验证集上的效果反而下降了
    
#### 实验 3  warmup_steps=40000
    
    (0) 模型
       
    (1) 数据集 
    
    训练数据:    
    N_train = 4343134 ( 源句子-目标句子 pair 的数目, 过滤掉长度大于 64 )
    
    验证数据(newstest2013): 
    N_valid = 2975 ( 源句子-目标句子 pair 的数目, 过滤掉长度大于 64 ) 

    测试数据(newstest2014): 
    N_test = 2737 
    
    
    (2) 数据预处理
    
    未做 unicode 标准化
    
    使用 wordpiece subword 算法分词
    源语言词表大小: n_vocab_source=30000
    目标语言词表大小: n_vocab_target=30000
    
    
    (3) 优化器参数
    
    epoch_num = 10
    token_in_batch = 12288
    
    label_smoothing=0
    
    optimizer= Adam with warmup_steps
    warmup_steps = 40000
    
    (5) 训练过程
    
    在每一个 epoch 结束时都对模型进行持久化(checkpoint), 并计算在验证集上的 bleu 得分
  
    Epoch 1/9
    
    bleu_score:{'1-garm': 0.2195131501664165, '2-garm': 0.08042179503666612, '3-garm': 0.031169729652626026, '4-garm': 0.012521833705404559}
    14308/14308 [==============================] - 4969s 346ms/step - loss: 6.2813 - accuracy: 0.1565 - val_loss: 3.9922 - val_accuracy: 0.3157
    
    Epoch 2/9
    
    
    bleu_score:{'1-garm': 0.4981359129314444, '2-garm': 0.331822680610765, '3-garm': 0.23457356666625387, '4-garm': 0.17008474856283343}
    14308/14308 [==============================] - 4907s 343ms/step - loss: 3.3035 - accuracy: 0.3713 - val_loss: 2.1419 - val_accuracy: 0.5756
    
    Epoch 3/9
    
    bleu_score:{'1-garm': 0.5323057746805994, '2-garm': 0.3669486888964614, '3-garm': 0.26730690268362384, '4-garm': 0.19921099031761372}
    14308/14308 [==============================] - 4903s 342ms/step - loss: 2.1377 - accuracy: 0.5487 - val_loss: 1.8212 - val_accuracy: 0.6226
    
    Epoch 4/9
    
    bleu_score:{'1-garm': 0.540609142077259, '2-garm': 0.37971041409838285, '3-garm': 0.2807277992330214, '4-garm': 0.2121459781090728}
    14308/14308 [==============================] - 4944s 345ms/step - loss: 1.8885 - accuracy: 0.5865 - val_loss: 1.6561 - val_accuracy: 0.6449
    
    Epoch 5/9
    
    bleu_score:{'1-garm': 0.5477085743668852, '2-garm': 0.38942790622817974, '3-garm': 0.29077501974677555, '4-garm': 0.2217931116541858}
    14308/14308 [==============================] - 4929s 344ms/step - loss: 1.7633 - accuracy: 0.6055 - val_loss: 1.5842 - val_accuracy: 0.6562
    
    Epoch 6/9
    
    bleu_score:{'1-garm': 0.5505345525433566, '2-garm': 0.39334670647637615, '3-garm': 0.2956567520322611, '4-garm': 0.22703597714987525}
    14308/14308 [==============================] - 4871s 340ms/step - loss: 1.6884 - accuracy: 0.6177 - val_loss: 1.5401 - val_accuracy: 0.6632
    
    Epoch 7/9
    
    bleu_score:{'1-garm': 0.551928244780108, '2-garm': 0.39560350310828923, '3-garm': 0.2977172412130709, '4-garm': 0.22885158960297677}
    14308/14308 [==============================] - 4874s 340ms/step - loss: 1.6381 - accuracy: 0.6256 - val_loss: 1.5095 - val_accuracy: 0.6677
    
    Epoch 8/9
    
    candidates:
    [0] Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzuwirken .
    [1] Die republikanischen Führer haben ihre Politik durch die Notwendigkeit der Bekämpfung von Wahlbetrug gerechtfertigt .
    [2] Das Brennan # # AT # # - # # AT # # Zentrum betrachtet dies jedoch als einen Mythos , in dem es heißt , dass der Wahlbetrug in den Vereinigten Staaten seltener ist als die Zahl der durch Blitztoten getöteten Menschen .
    [3] Tatsächlich haben republikanische Juristen in einem Jahrzehnt nur 300 Fälle von Wahlbetrug in den USA festgestellt .
    [4] Eines ist sicher : Diese neuen Bestimmungen werden negative Auswirkungen auf die Wahlbeteiligung haben .
    [5] In diesem Sinne werden die Maßnahmen das demokratische System der USA teilweise untergraben .
    [6] Im Gegensatz zu Kanada sind die USA für die Organisation von Bundeswahlen in den Vereinigten Staaten verantwortlich .
    [7] In diesem Sinne hat eine Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze erlassen , was die Registrierung oder Abstimmung erschwert .
    [8] Dieses Phänomen hat nach den Wahlen im November 2010 , die 675 neue republikanische Vertreter in 26 Staaten hinzukamen , an Dynamik gewonnen .
    [9] Im Ergebnis wurden allein 2011 180 Rechnungen eingeführt , die die Ausübung des Stimmrechts in 41 Staaten einschränken .
    
    references:
    [0] ['Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzutreten']
    [1] ['Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit , den Wahlbetrug zu bekämpfen .']
    [2] ['Allerdings hält das Brennan Center letzteres für einen Mythos , indem es bekräftigt , dass der Wahlbetrug in den USA seltener ist als die Anzahl der vom Blitzschlag getöteten Menschen .']
    [3] ['Die Rechtsanwälte der Republikaner haben in 10 Jahren in den USA übrigens nur 300 Fälle von Wahlbetrug verzeichnet .']
    [4] ['Eins ist sicher : diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .']
    [5] ['In diesem Sinne untergraben diese Maßnahmen teilweise das demokratische System der USA .']
    [6] ['Im Gegensatz zu Kanada sind die US ##AT##-##AT## Bundesstaaten für die Durchführung der Wahlen in den einzelnen Staaten verantwortlich .']
    [7] ['In diesem Sinne hat die Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verkündet , die das Verfahren für die Registrierung oder den Urnengang erschweren .']
    [8] ['Dieses Phänomen hat nach den Wahlen vom November 2010 an Bedeutung gewonnen , bei denen 675 neue republikanische Vertreter in 26 Staaten verzeichnet werden konnten .']
    [9] ['Infolgedessen wurden 180 Gesetzesentwürfe allein im Jahr 2011 eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .']
    
    bleu_score:{'1-garm': 0.5562467653548834, '2-garm': 0.3986587081222473, '3-garm': 0.30004918333012925, '4-garm': 0.23089974309850575}
    14308/14308 [==============================] - 4878s 341ms/step - loss: 1.6064 - accuracy: 0.6307 - val_loss: 1.4883 - val_accuracy: 0.6721
    
    Epoch 9/9
    
    bleu_score:{'1-garm': 0.554436618435967, '2-garm': 0.3963182096696606, '3-garm': 0.29788542142942087, '4-garm': 0.22892224065149505}
    14308/14308 [==============================] - 4877s 341ms/step - loss: 1.5812 - accuracy: 0.6343 - val_loss: 1.4805 - val_accuracy: 0.6748


    (6) 模型评价
    
    在测试集上评价模型
    
    1.epoch=8 时的模型

    candidates:
    [0] Orlando Bloom und Miranda Kerr lieben einander noch immer .
    [1] Schauspieler Orlando Bloom und Model Miranda Kerr wollen ihre eigenen Wege gehen .
    [2] In einem Interview hat Bloom jedoch gesagt , dass er und Kerr sich immer noch lieben .
    [3] Miranda Kerr und Orlando Bloom sind Eltern von zweijährigen Flynn .
    [4] Der Schauspieler Orlando Bloom hat seine Trennung von seiner Frau , Supermodel Miranda Kerr , angekündigt .
    [5] In einem Interview mit dem US # # AT # # - # # AT # # Journalisten Katie Couric , das am Freitag ( Ortszeit ) ausgestrahlt werden soll , sagte Bloom : & quot ; Manchmal geht das Leben nicht so weit , wie wir es planen oder hoffen & quot ; .
    [6] Er und Kerr lieben einander noch immer , betonten die 36 # # AT # # - # # AT # # Jährigen .
    [7] & quot ; Wir werden einander unterstützen und uns als Eltern nach Flynn lieben & quot ; .
    [8] Kerr und Bloom sind seit 2010 verheiratet und ihr Sohn Flynn wurde 2011 geboren .
    [9] Jet # # AT # # - # # AT # # Hersteller haben die Sitzbreite mit großen Auftragseingängen über die gewünschte Sitzbreite gefehlt .
    
    references:
    [0] ['Orlando Bloom und Miranda Kerr lieben sich noch immer']
    [1] ['Schauspieler Orlando Bloom und Model Miranda Kerr wollen künftig getrennte Wege gehen .']
    [2] ['In einem Interview sagte Bloom jedoch , dass er und Kerr sich noch immer lieben .']
    [3] ['Miranda Kerr und Orlando Bloom sind Eltern des zweijährigen Flynn .']
    [4] ['Schauspieler Orlando Bloom hat sich zur Trennung von seiner Frau , Topmodel Miranda Kerr , geäußert .']
    [5] ['In einem Interview mit US ##AT##-##AT## Journalistin Katie Couric , das am Freitag ( Ortszeit ) ausgestrahlt werden sollte , sagte Bloom , &quot; das Leben verläuft manchmal nicht genau so , wie wir es planen oder erhoffen &quot; .']
    [6] ['Kerr und er selbst liebten sich noch immer , betonte der 36 ##AT##-##AT## Jährige .']
    [7] ['&quot; Wir werden uns gegenseitig unterstützen und lieben als Eltern von Flynn &quot; .']
    [8] ['Kerr und Bloom sind seit 2010 verheiratet , im Jahr 2011 wurde ihr Söhnchen Flynn geboren .']
    [9] ['Jumbo ##AT##-##AT## Hersteller streiten im Angesicht großer Bestellungen über Sitzbreite']
    
    bleu_score:{'1-garm': 0.5415640697330838, '2-garm': 0.3895663301583339, '3-garm': 0.2923390303642965, '4-garm': 0.22448129889266707}
    
    
### 2.2 验证词表大小的效果

#### 实验 4  n_vocab=37000
    
    (0) 模型
       
    (1) 数据集 
    
    训练数据:    
    N_train = 4343134 ( 源句子-目标句子 pair 的数目, 过滤掉长度大于 64 )
    
    验证数据(newstest2013): 
    N_valid = 2975 ( 源句子-目标句子 pair 的数目, 过滤掉长度大于 64 ) 

    测试数据(newstest2014): 
    N_test = 2737 
    
    
    (2) 数据预处理
    
    未做 unicode 标准化
    
    使用 wordpiece subword 算法分词
    源语言词表大小: n_vocab_source=37000 (实际 35487)
    目标语言词表大小: n_vocab_target=37000 (实际 36601)
    
    
    (3) 优化器参数
    
    epoch_num = 10
    token_in_batch = 12288
    
    label_smoothing=0 
    
    optimizer= Adam with warmup_steps
    warmup_steps = 32000
    
    (5) 训练过程
    
    在每一个 epoch 结束时都对模型进行持久化(checkpoint), 并计算在验证集上的 bleu 得分

    model architecture param:
    num_layers:6, d_model:512, num_heads:8, dff:2048, n_vocab_source:35487, n_vocab_target:36601
    -------------------------
    valid source seq num :2970
    
    Epoch 1/10
    
    bleu_score:{'1-garm': 0.1683870165402313, '2-garm': 0.06390137373776375, '3-garm': 0.0253629647915209, '4-garm': 0.010221367625056089}
    14085/14085 [==============================] - 5201s 368ms/step - loss: 6.1717 - accuracy: 0.1662 - val_loss: 3.8570 - val_accuracy: 0.3362
    
    Epoch 2/10
    
    bleu_score:{'1-garm': 0.5068661480327165, '2-garm': 0.34190691591071226, '3-garm': 0.2442783535013225, '4-garm': 0.178940424171506}
    14085/14085 [==============================] - 5169s 367ms/step - loss: 3.1217 - accuracy: 0.3997 - val_loss: 2.0272 - val_accuracy: 0.5898
    
    Epoch 3/10
    
    bleu_score:{'1-garm': 0.5358901522008866, '2-garm': 0.3735847560173235, '3-garm': 0.27500689047809695, '4-garm': 0.2069295799734204}
    14085/14085 [==============================] - 5075s 360ms/step - loss: 2.0924 - accuracy: 0.5572 - val_loss: 1.7876 - val_accuracy: 0.6255
    
    Epoch 4/10
    
    bleu_score:{'1-garm': 0.5428312833043801, '2-garm': 0.3833746069865631, '3-garm': 0.28479532592060197, '4-garm': 0.21600456766498558}
    14085/14085 [==============================] - 5105s 362ms/step - loss: 1.8549 - accuracy: 0.5925 - val_loss: 1.6488 - val_accuracy: 0.6488
    
    Epoch 5/10
    
    bleu_score:{'1-garm': 0.5502223363780843, '2-garm': 0.3902829187597053, '3-garm': 0.29064842125355095, '4-garm': 0.22103570267701456}
    14085/14085 [==============================] - 5096s 361ms/step - loss: 1.7512 - accuracy: 0.6086 - val_loss: 1.5922 - val_accuracy: 0.6583
    
    Epoch 6/10
    
    bleu_score:{'1-garm': 0.5499385114761176, '2-garm': 0.3928398199460398, '3-garm': 0.2946990120465195, '4-garm': 0.22585845488390383}
    14085/14085 [==============================] - 5057s 359ms/step - loss: 1.6931 - accuracy: 0.6176 - val_loss: 1.5510 - val_accuracy: 0.6625
    
    Epoch 7/10
    
    bleu_score:{'1-garm': 0.5493828358825784, '2-garm': 0.3926541031286197, '3-garm': 0.294567144149417, '4-garm': 0.2254314317085915}
    14085/14085 [==============================] - 5051s 358ms/step - loss: 1.6461 - accuracy: 0.6250 - val_loss: 1.5360 - val_accuracy: 0.6673
    
    Epoch 8/10
    
    bleu_score:{'1-garm': 0.5561410555064664, '2-garm': 0.39845970122247476, '3-garm': 0.3001720118600563, '4-garm': 0.23082356140677165}
    14085/14085 [==============================] - 5055s 359ms/step - loss: 1.6071 - accuracy: 0.6311 - val_loss: 1.5061 - val_accuracy: 0.6716
    
    Epoch 9/10
    
    bleu_score:{'1-garm': 0.5574309854710782, '2-garm': 0.40090990692673656, '3-garm': 0.3027317557264984, '4-garm': 0.23323993069432444}
    14085/14085 [==============================] - 5116s 363ms/step - loss: 1.5867 - accuracy: 0.6341 - val_loss: 1.4933 - val_accuracy: 0.6710
    
    Epoch 10/10
    
    candidates:
    [0] Eine republikanische Strategie , um der Wiederwahl Obamas entgegenzuwirken
    [1] Die republikanischen Führer haben ihre Politik durch die Notwendigkeit der Bekämpfung von Wahlbetrug gerechtfertigt .
    [2] Das Brennan # # AT # # - # # AT # # Zentrum betrachtet dies jedoch als einen Mythos , in dem es heißt , Wahlbetrug sei in den USA seltener als die Anzahl der durch Blitz getöteten Menschen .
    [3] Tatsächlich identifizierten republikanische Anwälte in den Vereinigten Staaten in einem Jahrzehnt nur 300 Fälle von Wahlbetrug .
    [4] Eines ist sicher : Diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .
    [5] In diesem Sinne werden die Maßnahmen das demokratische System der USA teilweise untergraben .
    [6] Im Gegensatz zu Kanada sind die USA für die Durchführung von Bundeswahlen in den Vereinigten Staaten verantwortlich .
    [7] In diesem Sinne hat eine Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze erlassen , die den Registrierungs - oder Abstimmungsprozess erschweren .
    [8] Dieses Phänomen hat nach den Wahlen im November 2010 an Dynamik gewonnen , die 675 neue republikanische Repräsentanten in 26 Staaten hinzugefügt haben .
    [9] Infolgedessen wurden allein im Jahr 2011 180 Rechnungen eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .
    
    references:
    [0] ['Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzutreten']
    [1] ['Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit , den Wahlbetrug zu bekämpfen .']
    [2] ['Allerdings hält das Brennan Center letzteres für einen Mythos , indem es bekräftigt , dass der Wahlbetrug in den USA seltener ist als die Anzahl der vom Blitzschlag getöteten Menschen .']
    [3] ['Die Rechtsanwälte der Republikaner haben in 10 Jahren in den USA übrigens nur 300 Fälle von Wahlbetrug verzeichnet .']
    [4] ['Eins ist sicher : diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .']
    [5] ['In diesem Sinne untergraben diese Maßnahmen teilweise das demokratische System der USA .']
    [6] ['Im Gegensatz zu Kanada sind die US ##AT##-##AT## Bundesstaaten für die Durchführung der Wahlen in den einzelnen Staaten verantwortlich .']
    [7] ['In diesem Sinne hat die Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verkündet , die das Verfahren für die Registrierung oder den Urnengang erschweren .']
    [8] ['Dieses Phänomen hat nach den Wahlen vom November 2010 an Bedeutung gewonnen , bei denen 675 neue republikanische Vertreter in 26 Staaten verzeichnet werden konnten .']
    [9] ['Infolgedessen wurden 180 Gesetzesentwürfe allein im Jahr 2011 eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .']
    
    bleu_score:{'1-garm': 0.5571344667613294, '2-garm': 0.4018284804733413, '3-garm': 0.30465140901848564, '4-garm': 0.2358452381280454}
    14085/14085 [==============================] - 5107s 362ms/step - loss: 1.5748 - accuracy: 0.6361 - val_loss: 1.4750 - val_accuracy: 0.6772
    

    (6) 模型评价
    
    在测试集上评价模型
    
    1.epoch=10 时的模型
    
    candidates:
    [0] Orlando Bloom und Miranda Kerr lieben sich noch immer .
    [1] Schauspieler Orlando Bloom und Model Miranda Kerr wollen ihre eigenen Wege gehen .
    [2] In einem Interview hat Bloom jedoch gesagt , dass er und Kerr sich immer noch lieben .
    [3] Miranda Kerr und Orlando Bloom sind Eltern von zwei Jahren Flynn .
    [4] Der Schauspieler Orlando Bloom kündigte seine Trennung von seiner Frau , Supermodel Miranda Kerr .
    [5] In einem Interview mit dem US # # AT # # - # # AT # # Journalisten Katie Couric , der am Freitag ausgestrahlt werden soll ( lokale Zeit ) , sagte Bloom : & quot ; Manchmal geht das Leben nicht genau so , wie wir es uns vorstellen oder hoffen & quot ; .
    [6] Er und Kerr lieben einander noch immer , betonten den 36 # # AT # # - # # AT # # jährigen .
    [7] & quot ; Wir werden uns gegenseitig unterstützen und uns als Eltern Flynn gegenüber lieben & quot ; .
    [8] Kerr und Bloom sind seit 2010 verheiratet und ihr Sohn Flynn wurde 2011 geboren .
    [9] Jet # # AT # # - # # AT # # Hersteller feuerten über die Sitzbreite und große Aufträge auf dem Spiel .
    
    references:
    [0] ['Orlando Bloom und Miranda Kerr lieben sich noch immer']
    [1] ['Schauspieler Orlando Bloom und Model Miranda Kerr wollen künftig getrennte Wege gehen .']
    [2] ['In einem Interview sagte Bloom jedoch , dass er und Kerr sich noch immer lieben .']
    [3] ['Miranda Kerr und Orlando Bloom sind Eltern des zweijährigen Flynn .']
    [4] ['Schauspieler Orlando Bloom hat sich zur Trennung von seiner Frau , Topmodel Miranda Kerr , geäußert .']
    [5] ['In einem Interview mit US ##AT##-##AT## Journalistin Katie Couric , das am Freitag ( Ortszeit ) ausgestrahlt werden sollte , sagte Bloom , &quot; das Leben verläuft manchmal nicht genau so , wie wir es planen oder erhoffen &quot; .']
    [6] ['Kerr und er selbst liebten sich noch immer , betonte der 36 ##AT##-##AT## Jährige .']
    [7] ['&quot; Wir werden uns gegenseitig unterstützen und lieben als Eltern von Flynn &quot; .']
    [8] ['Kerr und Bloom sind seit 2010 verheiratet , im Jahr 2011 wurde ihr Söhnchen Flynn geboren .']
    [9] ['Jumbo ##AT##-##AT## Hersteller streiten im Angesicht großer Bestellungen über Sitzbreite']
    
    bleu_score:{'1-garm': 0.5469872404631482, '2-garm': 0.3961186944290748, '3-garm': 0.29923995659784275, '4-garm': 0.2311882663397863}
    

#### 实验 5  源语言和目标语言使用同个词表
    
    (0) 模型
       
    (1) 数据集 
    
    训练数据:    
    N_train = 4343134 ( 源句子-目标句子 pair 的数目, 过滤掉长度大于 64 )
    
    验证数据(newstest2013): 
    N_valid = 2975 ( 源句子-目标句子 pair 的数目, 过滤掉长度大于 64 ) 

    测试数据(newstest2014): 
    N_test = 2737 
    
    
    (2) 数据预处理
    
    未做 unicode 标准化
    
    使用 wordpiece subword 算法分词
    词表大小: n_vocab_target=37000 (实际 35476)
    
    
    (3) 优化器参数
    
    epoch_num = 10
    token_in_batch = 12288
    
    label_smoothing=0 
    
    optimizer= Adam with warmup_steps
    warmup_steps = 32000
    
    (5) 训练过程
    
    在每一个 epoch 结束时都对模型进行持久化(checkpoint), 并计算在验证集上的 bleu 得分

    model architecture param:
    num_layers:6, d_model:512, num_heads:8, dff:2048, n_vocab_source:35476, n_vocab_target:35476
    -------------------------
    valid source seq num :2970
    
    Epoch 1/10
    
    bleu_score:{'1-garm': 0.1680898918348872, '2-garm': 0.0674290486595373, '3-garm': 0.028911744224791674, '4-garm': 0.012968486818104084}
    14490/14490 [==============================] - 5260s 362ms/step - loss: 6.1170 - accuracy: 0.1645 - val_loss: 3.6555 - val_accuracy: 0.3475
    
    Epoch 2/10
    
    bleu_score:{'1-garm': 0.5060259826344126, '2-garm': 0.3421529682326908, '3-garm': 0.24533154374021746, '4-garm': 0.18033458376910486}
    14490/14490 [==============================] - 5163s 356ms/step - loss: 2.9371 - accuracy: 0.4230 - val_loss: 1.9480 - val_accuracy: 0.6025
    
    ........
    
    Epoch 9/10
    
    candidates:
    [0] Eine republikanische Strategie gegen die Wiederwahl von Obama
    [1] Die republikanischen Führer haben ihre Politik mit der Notwendigkeit , Wahlbetrug zu bekämpfen , gerechtfertigt .
    [2] Das Brennan Centre betrachtet dies jedoch als Mythos , in dem es heißt , dass Wahlbetrug in den Vereinigten Staaten seltener ist als die Zahl der durch Blitztodes getöteten Menschen .
    [3] Tatsächlich haben republikanische Rechtsanwälte in den Vereinigten Staaten in einem Jahrzehnt nur 300 Fälle von Wahlbetrug festgestellt .
    [4] Eines ist sicher : Diese neuen Bestimmungen werden sich negativ auf die Wahlurne auswirken .
    [5] In diesem Sinne werden die Maßnahmen das demokratische System der USA teilweise untergraben .
    [6] Im Gegensatz zu Kanada sind die USA für die Organisation von Bundeswahlen in den Vereinigten Staaten verantwortlich .
    [7] In diesem Sinne hat eine Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verabschiedet , die das Registrierungs - oder Abstimmungsverfahren schwieriger machen .
    [8] Dieses Phänomen gewann nach den Wahlen im November 2010 , die in 26 Staaten 675 neue republikanische Vertreter hinzufügten , an Dynamik .
    [9] Infolgedessen wurden allein im Jahr 2011 180 Rechnungen eingeführt , die die Ausübung des Stimmrechts in 41 Staaten einschränken .
    
    references:
    [0] ['Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzutreten']
    [1] ['Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit , den Wahlbetrug zu bekämpfen .']
    [2] ['Allerdings hält das Brennan Center letzteres für einen Mythos , indem es bekräftigt , dass der Wahlbetrug in den USA seltener ist als die Anzahl der vom Blitzschlag getöteten Menschen .']
    [3] ['Die Rechtsanwälte der Republikaner haben in 10 Jahren in den USA übrigens nur 300 Fälle von Wahlbetrug verzeichnet .']
    [4] ['Eins ist sicher : diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .']
    [5] ['In diesem Sinne untergraben diese Maßnahmen teilweise das demokratische System der USA .']
    [6] ['Im Gegensatz zu Kanada sind die US ##AT##-##AT## Bundesstaaten für die Durchführung der Wahlen in den einzelnen Staaten verantwortlich .']
    [7] ['In diesem Sinne hat die Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verkündet , die das Verfahren für die Registrierung oder den Urnengang erschweren .']
    [8] ['Dieses Phänomen hat nach den Wahlen vom November 2010 an Bedeutung gewonnen , bei denen 675 neue republikanische Vertreter in 26 Staaten verzeichnet werden konnten .']
    [9] ['Infolgedessen wurden 180 Gesetzesentwürfe allein im Jahr 2011 eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .']
    
    bleu_score:{'1-garm': 0.558017222932788, '2-garm': 0.40187455258401666, '3-garm': 0.30337950121526447, '4-garm': 0.23356804097223027}
    14490/14490 [==============================] - 5140s 354ms/step - loss: 1.5473 - accuracy: 0.6421 - val_loss: 1.4321 - val_accuracy: 0.6838
    
    Epoch 10/10
    
    
    candidates:
    [0] Eine republikanische Strategie gegen die Wiederwahl von Obama
    [1] Die republikanischen Führer haben ihre Politik durch die Notwendigkeit der Bekämpfung von Wahlbetrug gerechtfertigt .
    [2] Das Brennan # # AT # # - # # AT # # Zentrum betrachtet dies jedoch als Mythos , in dem es heißt , dass Wahlbetrug in den Vereinigten Staaten seltener ist als die Zahl der durch Blitz getöteten Menschen .
    [3] Tatsächlich haben republikanische Juristen in den Vereinigten Staaten in einem Jahrzehnt nur 300 Fälle von Wahlbetrug festgestellt .
    [4] Eines ist sicher : Diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .
    [5] In diesem Sinne werden die Maßnahmen das demokratische System der USA teilweise unterminieren .
    [6] Im Gegensatz zu Kanada sind die USA für die Organisation von Wahlen in den Vereinigten Staaten verantwortlich .
    [7] In diesem Sinne hat die Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze erlassen , die die Registrierung oder das Abstimmungsverfahren erschweren .
    [8] Dieses Phänomen gewann nach den Wahlen im November 2010 an Dynamik , wo 675 neue republikanische Vertreter in 26 Staaten hinzukamen .
    [9] Infolgedessen wurden allein im Jahr 2011 180 Rechnungen eingeführt , die die Ausübung des Stimmrechts in 41 Staaten einschränken .
    
    references:
    [0] ['Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzutreten']
    [1] ['Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit , den Wahlbetrug zu bekämpfen .']
    [2] ['Allerdings hält das Brennan Center letzteres für einen Mythos , indem es bekräftigt , dass der Wahlbetrug in den USA seltener ist als die Anzahl der vom Blitzschlag getöteten Menschen .']
    [3] ['Die Rechtsanwälte der Republikaner haben in 10 Jahren in den USA übrigens nur 300 Fälle von Wahlbetrug verzeichnet .']
    [4] ['Eins ist sicher : diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .']
    [5] ['In diesem Sinne untergraben diese Maßnahmen teilweise das demokratische System der USA .']
    [6] ['Im Gegensatz zu Kanada sind die US ##AT##-##AT## Bundesstaaten für die Durchführung der Wahlen in den einzelnen Staaten verantwortlich .']
    [7] ['In diesem Sinne hat die Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verkündet , die das Verfahren für die Registrierung oder den Urnengang erschweren .']
    [8] ['Dieses Phänomen hat nach den Wahlen vom November 2010 an Bedeutung gewonnen , bei denen 675 neue republikanische Vertreter in 26 Staaten verzeichnet werden konnten .']
    [9] ['Infolgedessen wurden 180 Gesetzesentwürfe allein im Jahr 2011 eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .']
    
    bleu_score:{'1-garm': 0.5554560996980719, '2-garm': 0.40043934069352943, '3-garm': 0.3032004908748171, '4-garm': 0.23450509264679664}
    14490/14490 [==============================] - 5144s 355ms/step - loss: 1.5242 - accuracy: 0.6457 - val_loss: 1.4095 - val_accuracy: 0.6858


    (6) 模型评价
    
    在测试集上评价模型
    
    1.epoch=10 时的模型
    
    candidates:
    [0] Orlando Bloom und Miranda Kerr lieben einander noch immer .
    [1] Schauspieler Orlando Bloom und Model Miranda Kerr wollen ihre eigenen Wege gehen .
    [2] In einem Interview hat Bloom jedoch gesagt , dass er und Kerr einander noch lieben .
    [3] Miranda Kerr und Orlando Bloom sind Eltern von zweijährigem Flynn .
    [4] Schauspieler Orlando Bloom kündigte seine Trennung von seiner Frau , Supermodel Miranda Kerr .
    [5] In einem Interview mit dem US # # AT # # - # # AT # # Journalisten Katie Couric , der am Freitag ( Ortszeit ) ausgestrahlt werden soll , sagte Bloom : & quot ; Manchmal geht das Leben nicht genau so , wie wir es planen oder hoffen & quot ; .
    [6] Er und Kerr lieben einander immer noch , betonten den 36 # # AT # # - # # AT # # jährigen .
    [7] & quot ; Wir werden einander unterstützen und einander als Eltern zu Flynn lieben & quot ; .
    [8] Kerr und Bloom sind seit 2010 verheiratet und ihr Sohn Flynn wurde im Jahr 2011 geboren .
    [9] Jet # # AT # # - # # AT # # Hersteller feuerten über die Breite der Sitze mit großen Aufträgen auf dem Spiel .
    
    references:
    [0] ['Orlando Bloom und Miranda Kerr lieben sich noch immer']
    [1] ['Schauspieler Orlando Bloom und Model Miranda Kerr wollen künftig getrennte Wege gehen .']
    [2] ['In einem Interview sagte Bloom jedoch , dass er und Kerr sich noch immer lieben .']
    [3] ['Miranda Kerr und Orlando Bloom sind Eltern des zweijährigen Flynn .']
    [4] ['Schauspieler Orlando Bloom hat sich zur Trennung von seiner Frau , Topmodel Miranda Kerr , geäußert .']
    [5] ['In einem Interview mit US ##AT##-##AT## Journalistin Katie Couric , das am Freitag ( Ortszeit ) ausgestrahlt werden sollte , sagte Bloom , &quot; das Leben verläuft manchmal nicht genau so , wie wir es planen oder erhoffen &quot; .']
    [6] ['Kerr und er selbst liebten sich noch immer , betonte der 36 ##AT##-##AT## Jährige .']
    [7] ['&quot; Wir werden uns gegenseitig unterstützen und lieben als Eltern von Flynn &quot; .']
    [8] ['Kerr und Bloom sind seit 2010 verheiratet , im Jahr 2011 wurde ihr Söhnchen Flynn geboren .']
    [9] ['Jumbo ##AT##-##AT## Hersteller streiten im Angesicht großer Bestellungen über Sitzbreite']
    
    bleu_score:{'1-garm': 0.5475656656574749, '2-garm': 0.39582382942795136, '3-garm': 0.2987618703748444, '4-garm': 0.2305833028485888}


### 2.3 验证 shared Embedding 的效果

#### 实验 6 shared Embedding,  n_vocab=37000
    
    (0) 模型
       1.编码器的 Embedding, 解码器的 Embedding , 和解码器的输出层共享权重矩阵
       
    (1) 数据集 
    
    训练数据:    
    N_train = 4343134 ( 源句子-目标句子 pair 的数目, 过滤掉长度大于 64 )
    
    验证数据(newstest2013): 
    N_valid = 2975 ( 源句子-目标句子 pair 的数目, 过滤掉长度大于 64 ) 

    测试数据(newstest2014): 
    N_test = 2737 
    
    
    (2) 数据预处理
    
    未做 unicode 标准化
    
    使用 wordpiece subword 算法分词, 源语言和目标语言使用同个词表
    词表大小:  n_vocab_target=37000 (实际 35476)
    
    
    (3) 优化器参数
    
    epoch_num = 10
    token_in_batch = 12288
    
    label_smoothing=0 
    
    optimizer= Adam with warmup_steps
    warmup_steps = 32000
    
    (5) 训练过程
    
    在每一个 epoch 结束时都对模型进行持久化(checkpoint), 并计算在验证集上的 bleu 得分

    model architecture param:
    num_layers:6, d_model:512, num_heads:8, dff:2048, n_vocab_source:35476, n_vocab_target:35476
    -------------------------
    valid source seq num :2970
    
    Epoch 1/10
    
    bleu_score:{'1-garm': 0.13655764873000303, '2-garm': 0.05928061939339903, '3-garm': 0.027103914545669603, '4-garm': 0.012794694978292755}
    14490/14490 [==============================] - 5073s 349ms/step - loss: 5.9040 - accuracy: 0.1782 - val_loss: 3.7195 - val_accuracy: 0.3340
    
    Epoch 2/10
    
    bleu_score:{'1-garm': 0.5091067745518008, '2-garm': 0.34474924125766776, '3-garm': 0.24715372434856592, '4-garm': 0.18143255657614282}
    14490/14490 [==============================] - 5008s 345ms/step - loss: 2.9520 - accuracy: 0.4226 - val_loss: 1.9734 - val_accuracy: 0.5983
    
    Epoch 3/10
    
    bleu_score:{'1-garm': 0.5377169826413887, '2-garm': 0.3769300682656494, '3-garm': 0.27767244804642527, '4-garm': 0.20913348426281989}
    14490/14490 [==============================] - 4942s 341ms/step - loss: 2.0322 - accuracy: 0.5661 - val_loss: 1.7075 - val_accuracy: 0.6410
    
    Epoch 4/10
    
    bleu_score:{'1-garm': 0.5442663050523814, '2-garm': 0.38615304810337236, '3-garm': 0.2878699154640548, '4-garm': 0.2192020226760711}
    14490/14490 [==============================] - 4930s 340ms/step - loss: 1.8206 - accuracy: 0.5982 - val_loss: 1.6009 - val_accuracy: 0.6566
    
    Epoch 5/10
    
    bleu_score:{'1-garm': 0.5489125354714482, '2-garm': 0.39116660208877907, '3-garm': 0.29316559924267394, '4-garm': 0.22453590895830336}
    14490/14490 [==============================] - 4931s 340ms/step - loss: 1.7259 - accuracy: 0.6126 - val_loss: 1.5330 - val_accuracy: 0.6670
    
    Epoch 6/10
    
    bleu_score:{'1-garm': 0.5529390236867108, '2-garm': 0.39520434454807685, '3-garm': 0.29704468417120444, '4-garm': 0.22826697386894615}
    14490/14490 [==============================] - 4998s 345ms/step - loss: 1.6656 - accuracy: 0.6218 - val_loss: 1.5102 - val_accuracy: 0.6696
    
    Epoch 7/10
    
    bleu_score:{'1-garm': 0.553812410889348, '2-garm': 0.39650143863027676, '3-garm': 0.2985092707432185, '4-garm': 0.22932759255246188}
    14490/14490 [==============================] - 4990s 344ms/step - loss: 1.6371 - accuracy: 0.6263 - val_loss: 1.4687 - val_accuracy: 0.6752
    
    Epoch 8/10
    
    bleu_score:{'1-garm': 0.5558318421577915, '2-garm': 0.397957731585552, '3-garm': 0.2997704754532075, '4-garm': 0.2303258216612328}
    14490/14490 [==============================] - 4965s 342ms/step - loss: 1.6009 - accuracy: 0.6323 - val_loss: 1.4533 - val_accuracy: 0.6777
    
    Epoch 9/10
    
    candidates:
    [0] Eine republikanische Strategie gegen die Wiederwahl von Obama
    [1] Die republikanischen Führer rechtfertigten ihre Politik mit der Notwendigkeit , Wahlbetrug zu bekämpfen .
    [2] Das Brennan # # AT # # - # # AT # # Zentrum betrachtet dies jedoch als einen Mythos , in dem festgestellt wird , dass Wahlbetrug in den Vereinigten Staaten seltener vorkommt als die Anzahl der durch Blitze getöteten Personen .
    [3] Tatsächlich haben republikanische Anwälte innerhalb eines Jahrzehnts nur 300 Fälle von Wahlbetrug in den USA festgestellt .
    [4] Eines ist sicher : Diese neuen Bestimmungen werden negative Auswirkungen auf die Wahlbeteiligung haben .
    [5] In diesem Sinne werden die Maßnahmen das demokratische System Amerikas teilweise untergraben .
    [6] Anders als in Kanada sind die USA für die Durchführung von Bundeswahlen in den USA verantwortlich .
    [7] In diesem Sinne hat eine Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verabschiedet , die die Registrierung oder Abstimmung erschweren .
    [8] Dieses Phänomen hat nach den Wahlen im November 2010 an Dynamik gewonnen , die in 26 Staaten 675 neue republikanische Vertreter aufgenommen haben .
    [9] Infolgedessen wurden allein im Jahr 2011 180 Gesetzesänderungen eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .
    
    references:
    [0] ['Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzutreten']
    [1] ['Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit , den Wahlbetrug zu bekämpfen .']
    [2] ['Allerdings hält das Brennan Center letzteres für einen Mythos , indem es bekräftigt , dass der Wahlbetrug in den USA seltener ist als die Anzahl der vom Blitzschlag getöteten Menschen .']
    [3] ['Die Rechtsanwälte der Republikaner haben in 10 Jahren in den USA übrigens nur 300 Fälle von Wahlbetrug verzeichnet .']
    [4] ['Eins ist sicher : diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .']
    [5] ['In diesem Sinne untergraben diese Maßnahmen teilweise das demokratische System der USA .']
    [6] ['Im Gegensatz zu Kanada sind die US ##AT##-##AT## Bundesstaaten für die Durchführung der Wahlen in den einzelnen Staaten verantwortlich .']
    [7] ['In diesem Sinne hat die Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verkündet , die das Verfahren für die Registrierung oder den Urnengang erschweren .']
    [8] ['Dieses Phänomen hat nach den Wahlen vom November 2010 an Bedeutung gewonnen , bei denen 675 neue republikanische Vertreter in 26 Staaten verzeichnet werden konnten .']
    [9] ['Infolgedessen wurden 180 Gesetzesentwürfe allein im Jahr 2011 eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .']
    
    bleu_score:{'1-garm': 0.5569517384231241, '2-garm': 0.40165611952738756, '3-garm': 0.3040523255740775, '4-garm': 0.2347718856509938}
    14490/14490 [==============================] - 4922s 339ms/step - loss: 1.5825 - accuracy: 0.6349 - val_loss: 1.4335 - val_accuracy: 0.6824
    
    Epoch 10/10
    
    candidates:
    [0] Eine republikanische Strategie zur Bekämpfung der Wiederwahl von Obama
    [1] Die republikanische Führung rechtfertigte ihre Politik mit der Notwendigkeit , den Wahlbetrug zu bekämpfen .
    [2] Das Zentrum von Brennan betrachtet dies jedoch als einen Mythos , in dem es heißt , dass Wahlbetrug in den Vereinigten Staaten seltener ist als die Anzahl der Menschen , die durch Blitzer getötet wurden .
    [3] Tatsächlich haben republikanische Anwälte innerhalb eines Jahrzehnts nur 300 Fälle von Wahlbetrug in den Vereinigten Staaten festgestellt .
    [4] Eines ist sicher : Diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .
    [5] In diesem Sinne werden die Maßnahmen das amerikanische demokratische System teilweise untergraben .
    [6] Anders als in Kanada sind die USA für die Durchführung von Bundestagswahlen in den Vereinigten Staaten verantwortlich .
    [7] In diesem Sinne hat eine Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze erlassen , die die Registrierung oder das Abstimmungsverfahren erschweren .
    [8] Dieses Phänomen hat nach den Wahlen im November 2010 an Schwung gewonnen , als in 26 Staaten in den letzten Jahren in den letzten zehn Jahren in den letzten Jahren in den letzten Jahren in den letzten Jahren in den letzten Jahren in den letzten Jahren in den letzten Jahren in den letzten Jahren in den USA und in den letzten Jahren in den USA in den letzten Jahren in den USA eingekerkerten Ländern der Republikanerschaft zukamen .
    [9] Infolgedessen wurden allein im Jahr 2011 180 Gesetzesvorlagen erlassen , die die Ausübung des Stimmrechts in 41 Staaten einschränken .
    
    references:
    [0] ['Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzutreten']
    [1] ['Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit , den Wahlbetrug zu bekämpfen .']
    [2] ['Allerdings hält das Brennan Center letzteres für einen Mythos , indem es bekräftigt , dass der Wahlbetrug in den USA seltener ist als die Anzahl der vom Blitzschlag getöteten Menschen .']
    [3] ['Die Rechtsanwälte der Republikaner haben in 10 Jahren in den USA übrigens nur 300 Fälle von Wahlbetrug verzeichnet .']
    [4] ['Eins ist sicher : diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .']
    [5] ['In diesem Sinne untergraben diese Maßnahmen teilweise das demokratische System der USA .']
    [6] ['Im Gegensatz zu Kanada sind die US ##AT##-##AT## Bundesstaaten für die Durchführung der Wahlen in den einzelnen Staaten verantwortlich .']
    [7] ['In diesem Sinne hat die Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verkündet , die das Verfahren für die Registrierung oder den Urnengang erschweren .']
    [8] ['Dieses Phänomen hat nach den Wahlen vom November 2010 an Bedeutung gewonnen , bei denen 675 neue republikanische Vertreter in 26 Staaten verzeichnet werden konnten .']
    [9] ['Infolgedessen wurden 180 Gesetzesentwürfe allein im Jahr 2011 eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .']
    
    bleu_score:{'1-garm': 0.5576558158869667, '2-garm': 0.4019070622460768, '3-garm': 0.3038052118633201, '4-garm': 0.2343316027392797}
    14490/14490 [==============================] - 4915s 339ms/step - loss: 1.5580 - accuracy: 0.6388 - val_loss: 1.4209 - val_accuracy: 0.6835


    (6) 模型评价
    
    在测试集上评价模型
    
    1.epoch=10 时的模型


    candidates:
    [0] Orlando Bloom und Miranda Kerr lieben einander noch immer .
    [1] Actors Orlando Bloom und Model Miranda Kerr wollen ihre eigenen Wege gehen .
    [2] In einem Interview hat Bloom jedoch gesagt , dass er und Kerr sich noch immer lieben .
    [3] Miranda Kerr und Orlando Bloom sind Eltern von zwei Jahren altem Flynn .
    [4] Actor Orlando Bloom kündigte seine Trennung von seiner Frau , Supermodel Miranda Kerr .
    [5] In einem Interview mit US # # AT # # - # # AT # # Journalistin Katie Couric , die am Freitag ( Ortszeit ) ausgestrahlt werden soll , sagte Bloom : & quot ; Manchmal geht das Leben nicht genau so , wie wir planen oder hoffen & quot ; .
    [6] Er und Kerr lieben sich noch immer , betonten die 36 # # AT # # - # # AT # # Jährigen .
    [7] & quot ; Wir werden uns gegenseitig unterstützen und uns als Eltern zu Flynn lieben & quot ; .
    [8] Kerr und Bloom sind seit 2010 verheiratet und ihr Sohn Flynn wurde 2011 geboren .
    [9] Die Flugzeughersteller haben mit großen Aufträgen über die Breite der Sitze geschraubt .
    
    references:
    [0] ['Orlando Bloom und Miranda Kerr lieben sich noch immer']
    [1] ['Schauspieler Orlando Bloom und Model Miranda Kerr wollen künftig getrennte Wege gehen .']
    [2] ['In einem Interview sagte Bloom jedoch , dass er und Kerr sich noch immer lieben .']
    [3] ['Miranda Kerr und Orlando Bloom sind Eltern des zweijährigen Flynn .']
    [4] ['Schauspieler Orlando Bloom hat sich zur Trennung von seiner Frau , Topmodel Miranda Kerr , geäußert .']
    [5] ['In einem Interview mit US ##AT##-##AT## Journalistin Katie Couric , das am Freitag ( Ortszeit ) ausgestrahlt werden sollte , sagte Bloom , &quot; das Leben verläuft manchmal nicht genau so , wie wir es planen oder erhoffen &quot; .']
    [6] ['Kerr und er selbst liebten sich noch immer , betonte der 36 ##AT##-##AT## Jährige .']
    [7] ['&quot; Wir werden uns gegenseitig unterstützen und lieben als Eltern von Flynn &quot; .']
    [8] ['Kerr und Bloom sind seit 2010 verheiratet , im Jahr 2011 wurde ihr Söhnchen Flynn geboren .']
    [9] ['Jumbo ##AT##-##AT## Hersteller streiten im Angesicht großer Bestellungen über Sitzbreite']
    bleu_score:{'1-garm': 0.5513792652634332, '2-garm': 0.3998721335063865, '3-garm': 0.30248284053117147, '4-garm': 0.23343748496320382} 
    


### 2.4 验证 label smoothing 的效果

#### 实验 7 label_smoothing = 0.1 
    
    (0) 模型
       1.编码器的 Embedding, 解码器的 Embedding , 和解码器的输出层共享权重矩阵
       
    (1) 数据集 
    
    训练数据:    
    N_train = 4343134 ( 源句子-目标句子 pair 的数目, 过滤掉长度大于 64 )
    
    验证数据(newstest2013): 
    N_valid = 2975 ( 源句子-目标句子 pair 的数目, 过滤掉长度大于 64 ) 

    测试数据(newstest2014): 
    N_test = 2737 
    
    
    (2) 数据预处理
    
    未做 unicode 标准化
    
    使用 wordpiece subword 算法分词, 源语言和目标语言使用同个词表
    词表大小:  n_vocab_target=37000 (实际 35476)
    
    
    (3) 优化器参数
    
    epoch_num = 10
    token_in_batch = 12288
    
    label_smoothing=0.1 (开启 label smoothing)
    
    optimizer= Adam with warmup_steps
    warmup_steps = 32000
    
    (5) 训练过程
    
    在每一个 epoch 结束时都对模型进行持久化(checkpoint), 并计算在验证集上的 bleu 得分

    model architecture param:
    num_layers:6, d_model:512, num_heads:8, dff:2048, n_vocab_source:35476, n_vocab_target:35476
    -------------------------
    valid source seq num :2970
    
    Epoch 1/10
    
    bleu_score:{'1-garm': 0.2006561665719585, '2-garm': 0.08602052570951862, '3-garm': 0.03929024762343202, '4-garm': 0.018326779462654014}
    14490/14490 [==============================] - 8971s 618ms/step - loss: 6.3711 - accuracy: 0.1790 - val_loss: 4.5287 - val_accuracy: 0.3556
    
    Epoch 2/10
    
    bleu_score:{'1-garm': 0.5135950768514643, '2-garm': 0.3483559323952012, '3-garm': 0.2493729996223665, '4-garm': 0.18255401359548282}
    14490/14490 [==============================] - 8918s 615ms/step - loss: 3.8810 - accuracy: 0.4224 - val_loss: 3.0582 - val_accuracy: 0.5989
    
    Epoch 3/10
    
    bleu_score:{'1-garm': 0.5220400156924284, '2-garm': 0.3649850162637401, '3-garm': 0.268946732935944, '4-garm': 0.20270679303655537}
    14490/14490 [==============================] - 8991s 620ms/step - loss: 3.0676 - accuracy: 0.5664 - val_loss: 2.8259 - val_accuracy: 0.6396
    
    Epoch 4/10
    
    bleu_score:{'1-garm': 0.5455532265845653, '2-garm': 0.38598115801527416, '3-garm': 0.2872762795114825, '4-garm': 0.21841550615265667}
    14490/14490 [==============================] - 8800s 607ms/step - loss: 2.8695 - accuracy: 0.5994 - val_loss: 2.7197 - val_accuracy: 0.6558
    
    Epoch 5/10
    
    bleu_score:{'1-garm': 0.5493968203243589, '2-garm': 0.3915985179573576, '3-garm': 0.29376139511796356, '4-garm': 0.22566422170085843}
    14490/14490 [==============================] - 8923s 615ms/step - loss: 2.7845 - accuracy: 0.6136 - val_loss: 2.6626 - val_accuracy: 0.6655
    
    Epoch 6/10
    
    bleu_score:{'1-garm': 0.5530445536650409, '2-garm': 0.39642125196241784, '3-garm': 0.29813128727495947, '4-garm': 0.22885352713094306}
    14490/14490 [==============================] - 8965s 618ms/step - loss: 2.7281 - accuracy: 0.6237 - val_loss: 2.6374 - val_accuracy: 0.6707
    
    Epoch 7/10
    
    bleu_score:{'1-garm': 0.5562790468002489, '2-garm': 0.3982317639733293, '3-garm': 0.29944542301065113, '4-garm': 0.22997922980318777}
    14490/14490 [==============================] - 8806s 607ms/step - loss: 2.7069 - accuracy: 0.6265 - val_loss: 2.6192 - val_accuracy: 0.6736
    
    Epoch 8/10
    
    candidates:
    [0] Eine republikanische Strategie , der Wiederwahl Obamas entgegenzuwirken .
    [1] Die republikanischen Führer haben ihre Politik mit der Notwendigkeit begründet , den Wahlbetrug zu bekämpfen .
    [2] Das Brennan # # AT # # - # # AT # # Zentrum betrachtet dies jedoch als einen Mythos , der besagt , dass Wahlbetrug in den USA seltener ist als die Anzahl der durch Blitze getöteten Menschen .
    [3] Tatsächlich identifizierten republikanische Anwälte in einem Jahrzehnt nur 300 Fälle von Wahlbetrug in den USA .
    [4] Eines ist sicher : Diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .
    [5] In diesem Sinne werden die Maßnahmen das amerikanische demokratische System teilweise untergraben .
    [6] Im Gegensatz zu Kanada sind die USA für die Durchführung von Bundeswahlen in den USA verantwortlich .
    [7] In diesem Sinne hat eine Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze erlassen , die den Registrierungs - oder Abstimmungsprozess erschweren .
    [8] Dieses Phänomen gewann an Dynamik nach den Wahlen vom November 2010 , die 675 neue republikanische Vertreter in 26 Staaten hinzugefügt .
    [9] Infolgedessen wurden allein 2011 180 Rechnungen eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .
    
    references:
    [0] ['Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzutreten']
    [1] ['Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit , den Wahlbetrug zu bekämpfen .']
    [2] ['Allerdings hält das Brennan Center letzteres für einen Mythos , indem es bekräftigt , dass der Wahlbetrug in den USA seltener ist als die Anzahl der vom Blitzschlag getöteten Menschen .']
    [3] ['Die Rechtsanwälte der Republikaner haben in 10 Jahren in den USA übrigens nur 300 Fälle von Wahlbetrug verzeichnet .']
    [4] ['Eins ist sicher : diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .']
    [5] ['In diesem Sinne untergraben diese Maßnahmen teilweise das demokratische System der USA .']
    [6] ['Im Gegensatz zu Kanada sind die US ##AT##-##AT## Bundesstaaten für die Durchführung der Wahlen in den einzelnen Staaten verantwortlich .']
    [7] ['In diesem Sinne hat die Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verkündet , die das Verfahren für die Registrierung oder den Urnengang erschweren .']
    [8] ['Dieses Phänomen hat nach den Wahlen vom November 2010 an Bedeutung gewonnen , bei denen 675 neue republikanische Vertreter in 26 Staaten verzeichnet werden konnten .']
    [9] ['Infolgedessen wurden 180 Gesetzesentwürfe allein im Jahr 2011 eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .']
    
    bleu_score:{'1-garm': 0.557211700100863, '2-garm': 0.40155309939771383, '3-garm': 0.303264716385846, '4-garm': 0.23424748761732678}
    14490/14490 [==============================] - 8809s 608ms/step - loss: 2.6763 - accuracy: 0.6325 - val_loss: 2.5914 - val_accuracy: 0.6784
    
    Epoch 9/10
    
    candidates:
    [0] Eine republikanische Strategie gegen die Wiederwahl von Obama
    [1] Die republikanischen Führer rechtfertigten ihre Politik durch die Notwendigkeit , den Wahlbetrug zu bekämpfen .
    [2] Allerdings hält das Zentrum von Brennan diesen Mythos für selten , da es in den Vereinigten Staaten von Amerika Wahlbetrug als die Zahl der durch Blitz getöteten Menschen angibt .
    [3] Tatsächlich haben republikanische Anwälte in den Vereinigten Staaten innerhalb eines Jahrzehnts nur 300 Fälle von Wahlbetrug festgestellt .
    [4] Eines ist sicher : Diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .
    [5] In diesem Sinne werden die Maßnahmen das amerikanische demokratische System teilweise untergraben .
    [6] Anders als in Kanada sind die USA für die Durchführung von Bundeswahlen in den Vereinigten Staaten verantwortlich .
    [7] In diesem Sinne haben die meisten amerikanischen Regierungen seit 2009 neue Gesetze verabschiedet , die den Registrierungs - oder Wahlprozess erschweren .
    [8] Dieses Phänomen hat nach den Wahlen vom November 2010 , die in 26 Staaten 675 neue republikanische Vertreter hinzufügten , an Dynamik gewonnen .
    [9] Infolgedessen wurden allein 2011 180 Rechnungen eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .
    
    references:
    [0] ['Eine republikanische Strategie , um der Wiederwahl von Obama entgegenzutreten']
    [1] ['Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit , den Wahlbetrug zu bekämpfen .']
    [2] ['Allerdings hält das Brennan Center letzteres für einen Mythos , indem es bekräftigt , dass der Wahlbetrug in den USA seltener ist als die Anzahl der vom Blitzschlag getöteten Menschen .']
    [3] ['Die Rechtsanwälte der Republikaner haben in 10 Jahren in den USA übrigens nur 300 Fälle von Wahlbetrug verzeichnet .']
    [4] ['Eins ist sicher : diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken .']
    [5] ['In diesem Sinne untergraben diese Maßnahmen teilweise das demokratische System der USA .']
    [6] ['Im Gegensatz zu Kanada sind die US ##AT##-##AT## Bundesstaaten für die Durchführung der Wahlen in den einzelnen Staaten verantwortlich .']
    [7] ['In diesem Sinne hat die Mehrheit der amerikanischen Regierungen seit 2009 neue Gesetze verkündet , die das Verfahren für die Registrierung oder den Urnengang erschweren .']
    [8] ['Dieses Phänomen hat nach den Wahlen vom November 2010 an Bedeutung gewonnen , bei denen 675 neue republikanische Vertreter in 26 Staaten verzeichnet werden konnten .']
    [9] ['Infolgedessen wurden 180 Gesetzesentwürfe allein im Jahr 2011 eingeführt , die die Ausübung des Wahlrechts in 41 Staaten einschränken .']
    
    bleu_score:{'1-garm': 0.5489170292972204, '2-garm': 0.3952939894148382, '3-garm': 0.29866265009621734, '4-garm': 0.2306417575999063}
    14490/14490 [==============================] - 8807s 607ms/step - loss: 2.6558 - accuracy: 0.6357 - val_loss: 2.5821 - val_accuracy: 0.6810
    

    (6) 模型评价
    
    在测试集上评价模型
    
    1.epoch=9 时的模型

    candidates:
    [0] Orlando Bloom und Miranda Kerr lieben sich noch immer .
    [1] Schauspieler Orlando Bloom und Model Miranda Kerr wollen ihre eigenen Wege gehen .
    [2] In einem Interview hat Bloom jedoch gesagt , dass er und Kerr sich immer noch gegenseitig lieben .
    [3] Miranda Kerr und Orlando Bloom sind Eltern von zweijährigen Flynn .
    [4] Er ist ein US # # AT # # - # # AT # # amerikanischer Schauspieler .
    [5] In einem Interview mit dem US # # AT # # - # # AT # # Journalisten Katie Couric , der am Freitag ( Ortszeit ) ausgestrahlt werden soll , sagte Bloom : & quot ; Manchmal geht das Leben nicht genau so , wie wir es planen oder hoffen . & quot ;
    [6] Er und Kerr lieben sich noch immer , betonten die 36 # # AT # # - # # AT # # jährigen .
    [7] & quot ; Wir werden uns gegenseitig unterstützen und uns als Eltern zu Flynn lieben & quot ; .
    [8] Kerr und Bloom sind seit 2010 verheiratet und ihr Sohn Flynn wurde 2011 geboren .
    [9] Jethersteller feuerten über Sitzbreite mit großen Bestellungen auf dem Spiel .
    
    references:
    [0] ['Orlando Bloom und Miranda Kerr lieben sich noch immer']
    [1] ['Schauspieler Orlando Bloom und Model Miranda Kerr wollen künftig getrennte Wege gehen .']
    [2] ['In einem Interview sagte Bloom jedoch , dass er und Kerr sich noch immer lieben .']
    [3] ['Miranda Kerr und Orlando Bloom sind Eltern des zweijährigen Flynn .']
    [4] ['Schauspieler Orlando Bloom hat sich zur Trennung von seiner Frau , Topmodel Miranda Kerr , geäußert .']
    [5] ['In einem Interview mit US ##AT##-##AT## Journalistin Katie Couric , das am Freitag ( Ortszeit ) ausgestrahlt werden sollte , sagte Bloom , &quot; das Leben verläuft manchmal nicht genau so , wie wir es planen oder erhoffen &quot; .']
    [6] ['Kerr und er selbst liebten sich noch immer , betonte der 36 ##AT##-##AT## Jährige .']
    [7] ['&quot; Wir werden uns gegenseitig unterstützen und lieben als Eltern von Flynn &quot; .']
    [8] ['Kerr und Bloom sind seit 2010 verheiratet , im Jahr 2011 wurde ihr Söhnchen Flynn geboren .']
    [9] ['Jumbo ##AT##-##AT## Hersteller streiten im Angesicht großer Bestellungen über Sitzbreite']
    
    bleu_score:{'1-garm': 0.5414792685729293, '2-garm': 0.3914052946157209, '3-garm': 0.2954092339091706, '4-garm': 0.2280875321475785}
    
### 实验 6 - 实验 7 训练出的模型的结构有 bug

    lib\layers\embedding_layer_xrh.py 中的 embedding 层的输出结果 遗漏了 *sqrt(n_h), 详见论文 3.4 Embeddings and Softmax

### 2.5 验证 调整训练集中句子的长度限制    

#### 实验 8  过滤掉训练数据长度大于 256 的句子 
    
    (0) 模型
       1.编码器的 Embedding, 解码器的 Embedding , 和解码器的输出层共享权重矩阵
       
    (1) 数据集 
    
    训练数据:    
    N_train =  ( 源句子-目标句子 pair 的数目, 过滤掉长度大于 256 )
    
    seq length <=256 num: 4468840
    
    参考 transformer 源码(tensor2tensor)中配置为 256
    
    验证数据(newstest2013): 
    N_valid = 3000 ( 源句子-目标句子 pair 的数目, 过滤掉长度大于 256 ) 

    测试数据(newstest2014): 
    N_test = 2737 
    
    
    (2) 数据预处理
    
    未做 unicode 标准化
    
    使用 wordpiece subword 算法分词, 源语言和目标语言使用同个词表
    词表大小:  n_vocab_target=37000 (实际 35282)
    
    
    (3) 优化器参数
    
    epoch_num = 12
    token_in_batch = 8192
    
    label_smoothing=0 (不开启 label smoothing)
    
    optimizer= Adam with warmup_steps
    warmup_steps = 32000
    
    (5) 训练过程
    
    在每一个 epoch 结束时都对模型进行持久化(checkpoint), 并计算在验证集上的 bleu 得分
    
    model architecture param:
    num_layers:6, d_model:512, num_heads:8, dff:2048, n_vocab_source:35282, n_vocab_target:35282
    -------------------------
    valid source seq num :2995
    
    Epoch 1/12

    bleu_score:{'1-garm': 0.30240283779789134, '2-garm': 0.13691220033192164, '3-garm': 0.06640747743703057, '4-garm': 0.03307597312818358}
    23472/23472 [==============================] - 5752s 245ms/step - loss: 5.6378 - accuracy: 0.1934 - val_loss: 3.3171 - val_accuracy: 0.3923
    
    Epoch 2/12
    
    bleu_score:{'1-garm': 0.3949911880356944, '2-garm': 0.20155265756622218, '3-garm': 0.11161838260339722, '4-garm': 0.06485916044661477}
    23472/23472 [==============================] - 5686s 242ms/step - loss: 2.9239 - accuracy: 0.4053 - val_loss: 2.4861 - val_accuracy: 0.4934
    
    Epoch 3/12
    
    bleu_score:{'1-garm': 0.44645119944069794, '2-garm': 0.25988673796995243, '3-garm': 0.16271462967876585, '4-garm': 0.10538262996675529}
    23472/23472 [==============================] - 5792s 247ms/step - loss: 2.3685 - accuracy: 0.4841 - val_loss: 2.1730 - val_accuracy: 0.5464
    
    Epoch 4/12
    
    bleu_score:{'1-garm': 0.4797806294578898, '2-garm': 0.29592025430084945, '3-garm': 0.19647091707404554, '4-garm': 0.1349239563774463}
    23472/23472 [==============================] - 5834s 248ms/step - loss: 2.1229 - accuracy: 0.5293 - val_loss: 1.9912 - val_accuracy: 0.5800
    
    Epoch 5/12
    
    bleu_score:{'1-garm': 0.5020441341771639, '2-garm': 0.3231025869507344, '3-garm': 0.22252454552508633, '4-garm': 0.15814632128670225}
    23472/23472 [==============================] - 5793s 247ms/step - loss: 1.9800 - accuracy: 0.5559 - val_loss: 1.8717 - val_accuracy: 0.6019
    
    Epoch 6/12
    bleu_score:{'1-garm': 0.5111733982423534, '2-garm': 0.33881239087370935, '3-garm': 0.23841606459618667, '4-garm': 0.1723950531311091}
    23472/23472 [==============================] - 5763s 245ms/step - loss: 1.8890 - accuracy: 0.5733 - val_loss: 1.7947 - val_accuracy: 0.6149
    
    Epoch 7/12
    bleu_score:{'1-garm': 0.5175940666336225, '2-garm': 0.34940555890851593, '3-garm': 0.25017441643547766, '4-garm': 0.1839298555155588}
    23472/23472 [==============================] - 5776s 246ms/step - loss: 1.8272 - accuracy: 0.5851 - val_loss: 1.7359 - val_accuracy: 0.6258
    
    Epoch 8/12
    bleu_score:{'1-garm': 0.527918198100953, '2-garm': 0.36106814498639644, '3-garm': 0.2603707309245969, '4-garm': 0.19235713744378433}
    23472/23472 [==============================] - 5796s 247ms/step - loss: 1.7734 - accuracy: 0.5954 - val_loss: 1.6811 - val_accuracy: 0.6356
    
    Epoch 9/12
    bleu_score:{'1-garm': 0.5311254555180157, '2-garm': 0.36667290488970633, '3-garm': 0.26681750049470504, '4-garm': 0.19879909265793808}
    23472/23472 [==============================] - 5782s 246ms/step - loss: 1.7411 - accuracy: 0.6016 - val_loss: 1.6470 - val_accuracy: 0.6417
    
    Epoch 10/12
    bleu_score:{'1-garm': 0.5336162599424457, '2-garm': 0.37026199924295444, '3-garm': 0.2711879631544183, '4-garm': 0.20329872298819326}
    23472/23472 [==============================] - 5788s 246ms/step - loss: 1.7138 - accuracy: 0.6065 - val_loss: 1.6296 - val_accuracy: 0.6470
    
    Epoch 11/12
    bleu_score:{'1-garm': 0.5354671477360787, '2-garm': 0.37227278567950917, '3-garm': 0.2727797902278031, '4-garm': 0.20419649111692406}
    23472/23472 [==============================] - 5787s 246ms/step - loss: 1.6892 - accuracy: 0.6110 - val_loss: 1.6013 - val_accuracy: 0.6474
    
    Epoch 12/12
    bleu_score:{'1-garm': 0.5417585048880663, '2-garm': 0.37966269032264865, '3-garm': 0.28029132845747573, '4-garm': 0.21186754544002312}
    23472/23472 [==============================] - 5787s 246ms/step - loss: 1.6699 - accuracy: 0.6146 - val_loss: 1.5786 - val_accuracy: 0.6551
    

    (6) 模型评价
    
    在测试集上评价模型
    
    1.epoch=12 时的模型
    
    candidates:
    [0] Orlando Bloom und Miranda Kerr lieben einander noch .
    [1] Schauspieler Orlando Bloom und Miranda Modeler Kr wollen ihre Wege getrennt gehen .
    [2] Doch Bloom hat in einem Interview gesagt , dass er und Kerr immer noch Liebe miteinander haben .
    [3] Miranda Kerr Flynn und Orlando Bloom sind Eltern von zwei Jahren .
    [4] Der Schauspieler Miranda Bloom kündigte seine Trennung von seiner Frau , Supermodel Orlando Kerr an .
    [5] In einem Interview mit US # # AT # # - # # AT # # Journalist Katie Couric , der am Freitag ( lokale Zeit ) ausgestrahlt wird , sagte Bloom : & quot ; Manchmal geht das Leben nicht genau so , wie wir planen oder hoffen . & quot ;
    [6] Er und Kerr liebten sich noch , betonten die 36 # # AT # # - # # AT # # jährige .
    [7] & quot ; Wir werden uns gegenseitig unterstützen und einander als Eltern zu Flynn lieben & quot ; .
    [8] Kerr Flynn und Bloom sind seit 2010 verheiratet und ihr Sohn wurde im Jahr 2011 geboren .
    [9] Jet # # AT # # - # # AT # # Hersteller feuchte über den Sitzplatz mit großen Aufträgen
    
    references:
    [0] ['Orlando Bloom und Miranda Kerr lieben sich noch immer']
    [1] ['Schauspieler Orlando Bloom und Model Miranda Kerr wollen künftig getrennte Wege gehen .']
    [2] ['In einem Interview sagte Bloom jedoch , dass er und Kerr sich noch immer lieben .']
    [3] ['Miranda Kerr und Orlando Bloom sind Eltern des zweijährigen Flynn .']
    [4] ['Schauspieler Orlando Bloom hat sich zur Trennung von seiner Frau , Topmodel Miranda Kerr , geäußert .']
    [5] ['In einem Interview mit US ##AT##-##AT## Journalistin Katie Couric , das am Freitag ( Ortszeit ) ausgestrahlt werden sollte , sagte Bloom , &quot; das Leben verläuft manchmal nicht genau so , wie wir es planen oder erhoffen &quot; .']
    [6] ['Kerr und er selbst liebten sich noch immer , betonte der 36 ##AT##-##AT## Jährige .']
    [7] ['&quot; Wir werden uns gegenseitig unterstützen und lieben als Eltern von Flynn &quot; .']
    [8] ['Kerr und Bloom sind seit 2010 verheiratet , im Jahr 2011 wurde ihr Söhnchen Flynn geboren .']
    [9] ['Jumbo ##AT##-##AT## Hersteller streiten im Angesicht großer Bestellungen über Sitzbreite']
    
    bleu_score:{'1-garm': 0.5284628099173554, '2-garm': 0.3662729203311738, '3-garm': 0.266520470137184, '4-garm': 0.19832193715994678}
        
        
#### 实验 9  过滤掉训练数据长度大于 256 的句子, 并且开启 label smoothing
    
    (0) 模型
       1.编码器的 Embedding, 解码器的 Embedding , 和解码器的输出层共享权重矩阵
       
    (1) 数据集 
    
    训练数据:    
    N_train =  ( 源句子-目标句子 pair 的数目, 过滤掉长度大于 256 )
    
    seq length <=256 num: 4468840
    
    参考 transformer 源码(tensor2tensor)中配置为 256
    
    验证数据(newstest2013): 
    N_valid = 3000 ( 源句子-目标句子 pair 的数目, 过滤掉长度大于 256 ) 

    测试数据(newstest2014): 
    N_test = 2737 
    
    
    (2) 数据预处理
    
    未做 unicode 标准化
    
    使用 wordpiece subword 算法分词, 源语言和目标语言使用同个词表
    词表大小:  n_vocab_target=37000 (实际 35282)
    
    
    (3) 优化器参数
    
    epoch_num = 12
    token_in_batch = 8192
    
    label_smoothing=0.1 (开启 label smoothing)
    
    optimizer= Adam with warmup_steps
    warmup_steps = 48000
    
    (5) 训练过程
    
    在每一个 epoch 结束时都对模型进行持久化(checkpoint), 并计算在验证集上的 bleu 得分
    

    (6) 模型评价
    
    在测试集上评价模型
    
    1.epoch=12 时的模型
    
    candidates:
    [0] Orlando Bloom und Miranda Kerr lieben sich noch immer .
    [1] Schauspieler Orlando Bloom und Miranda Model Kerr wollen , dass sie ihre eigenen Wege gehen .
    [2] In einem Interview hat Bloom jedoch gesagt , dass er und Kerr einander immer noch lieben .
    [3] Miranda Kerr und Orlando Bloom sind Eltern von zwei Jahren Flynn .
    [4] Er hat seine Trennung von seiner Frau , Miranda Kerr , angekündigt .
    [5] In einem Interview mit US # # AT # # - # # AT # # Journalist Katie Couric , das am Freitag ( lokale Zeit ) gesendet werden soll , sagte Bloom : & quot ; Manchmal geht das Leben nicht genau so , wie wir planen oder hoffen & quot ; .
    [6] Er und Kerr lieben sich noch immer , betonten den 36 # # AT # # - # # AT # # jährigen .
    [7] & quot ; Wir werden uns gegenseitig unterstützen und uns als Eltern zu Flynn lieben & quot ; .
    [8] Kerr und Bloom sind seit 2010 verheiratet und ihr Sohn Flynn wurde 2011 geboren .
    [9] Jet # # AT # # - # # AT # # Hersteller feuchten über Sitzbreite mit großen Auftragsbeständen auf dem Spiel
    
    references:
    [0] ['Orlando Bloom und Miranda Kerr lieben sich noch immer']
    [1] ['Schauspieler Orlando Bloom und Model Miranda Kerr wollen künftig getrennte Wege gehen .']
    [2] ['In einem Interview sagte Bloom jedoch , dass er und Kerr sich noch immer lieben .']
    [3] ['Miranda Kerr und Orlando Bloom sind Eltern des zweijährigen Flynn .']
    [4] ['Schauspieler Orlando Bloom hat sich zur Trennung von seiner Frau , Topmodel Miranda Kerr , geäußert .']
    [5] ['In einem Interview mit US ##AT##-##AT## Journalistin Katie Couric , das am Freitag ( Ortszeit ) ausgestrahlt werden sollte , sagte Bloom , &quot; das Leben verläuft manchmal nicht genau so , wie wir es planen oder erhoffen &quot; .']
    [6] ['Kerr und er selbst liebten sich noch immer , betonte der 36 ##AT##-##AT## Jährige .']
    [7] ['&quot; Wir werden uns gegenseitig unterstützen und lieben als Eltern von Flynn &quot; .']
    [8] ['Kerr und Bloom sind seit 2010 verheiratet , im Jahr 2011 wurde ihr Söhnchen Flynn geboren .']
    [9] ['Jumbo ##AT##-##AT## Hersteller streiten im Angesicht großer Bestellungen über Sitzbreite']
    bleu_score:{'1-garm': 0.539172338417209, '2-garm': 0.38213888603861373, '3-garm': 0.2832620502111256, '4-garm': 0.21511854175510492}
    
