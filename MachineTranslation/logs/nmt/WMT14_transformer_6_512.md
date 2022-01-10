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
    
    label_smoothing=0.1 
    
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
    
    label_smoothing=0.1 
    
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