## 1.基本参数

### 1.1 数据集参数
    
    TED-Portuguese-English(pt_to_en) 数据集
    
    
### 1.2 模型参数

配置文件: config/transformer_seq2seq.ini (tag='TEST-1') 
          
    # 源句子的最大编码位置
    maximum_position_source = 1000
    
    # 目标句子的最大编码位置
    maximum_position_target = 1000
    
    # 堆叠的编码器(解码器)的层数
    num_layers = 4
    
    # 模型整体的隐藏层的维度
    d_model = 128
    
    #并行注意力层的个数(头数)
    num_heads = 8
    
    #Position-wise Feed-Forward 的中间层的维度
    dff = 512

## 2.实验记录

### 2.1 验证 不同划分 batch 策略的效果

#### 实验 1  固定 batch size
    
    (0) 模型
       
    
    (1) 数据集 
    
    训练数据:    
    N_train = 51785 ( 源句子-目标句子对 的数目 )
    
    source sentence length distribution:
    most common seq length: (seq length, count num)
    [(8, 2956), (9, 2880), (10, 2839), (11, 2614), (7, 2607), (6, 2526), (12, 2494), (13, 2244), (14, 2133), (5, 1816)]
    seq length count:  (seq length, count num)
    [(191, 1), (188, 1), (176, 1), (152, 1), (149, 1), (146, 1), (145, 2), (143, 1), (141, 1), (139, 1), (137, 1), (136, 1), (135, 2), (134, 1), (133, 1), (131, 2), (130, 1), (129, 1), (127, 2), (125, 6), (124, 3), (123, 2), (122, 1), (121, 2), (120, 5), (119, 3), (117, 4), (116, 2), (115, 1), (114, 4), (113, 1), (112, 3), (111, 1), (109, 3), (108, 3), (107, 5), (106, 2), (105, 5), (104, 10), (103, 6), (102, 5), (101, 4), (100, 5), (99, 5), (98, 5), (97, 1), (96, 3), (95, 3), (94, 4), (93, 4), (92, 5), (91, 8), (90, 5), (89, 6), (88, 7), (87, 8), (86, 14), (85, 10), (84, 11), (83, 14), (82, 12), (81, 15), (80, 16), (79, 12), (78, 12), (77, 9), (76, 15), (75, 18), (74, 18), (73, 21), (72, 24), (71, 27), (70, 29), (69, 28), (68, 27), (67, 22), (66, 31), (65, 32), (64, 41), (63, 42), (62, 42), (61, 47), (60, 50), (59, 47), (58, 50), (57, 70), (56, 58), (55, 57), (54, 88), (53, 90), (52, 70), (51, 103), (50, 104), (49, 108), (48, 124), (47, 149), (46, 153), (45, 166), (44, 172), (43, 197), (42, 197), (41, 233), (40, 240), (39, 262), (38, 288), (37, 321), (36, 334), (35, 355), (34, 430), (33, 440), (32, 489), (31, 539), (30, 576), (29, 610), (28, 632), (27, 696), (26, 760), (25, 901), (24, 936), (23, 988), (22, 1064), (21, 1182), (20, 1214), (19, 1384), (18, 1443), (17, 1541), (16, 1673), (15, 1778), (14, 2133), (13, 2244), (12, 2494), (11, 2614), (10, 2839), (9, 2880), (8, 2956), (7, 2607), (6, 2526), (5, 1816), (4, 1085), (3, 977), (2, 486), (1, 4)]
    target sentence length distribution:
    most common seq length: (seq length, count num)
    [(10, 2753), (8, 2733), (9, 2713), (11, 2679), (12, 2595), (7, 2518), (13, 2348), (6, 2218), (14, 2202), (15, 1901)]
    seq length count:  (seq length, count num)
    [(204, 1), (199, 1), (174, 1), (166, 1), (159, 1), (156, 1), (155, 1), (149, 1), (148, 1), (145, 1), (143, 3), (142, 3), (137, 1), (136, 1), (134, 3), (133, 2), (132, 1), (131, 1), (130, 2), (128, 2), (127, 1), (126, 3), (125, 1), (124, 4), (123, 3), (122, 1), (121, 7), (120, 5), (119, 5), (118, 2), (117, 3), (116, 2), (115, 3), (114, 3), (112, 2), (111, 5), (110, 3), (109, 1), (108, 5), (107, 3), (106, 2), (105, 2), (104, 2), (103, 5), (102, 5), (101, 4), (100, 3), (99, 7), (98, 4), (97, 5), (96, 5), (95, 8), (94, 3), (93, 6), (92, 8), (91, 5), (90, 6), (89, 7), (88, 5), (87, 10), (86, 13), (85, 8), (84, 13), (83, 9), (82, 12), (81, 13), (80, 12), (79, 14), (78, 20), (77, 21), (76, 17), (75, 15), (74, 17), (73, 25), (72, 33), (71, 36), (70, 25), (69, 35), (68, 35), (67, 29), (66, 39), (65, 36), (64, 37), (63, 42), (62, 57), (61, 59), (60, 44), (59, 66), (58, 72), (57, 56), (56, 59), (55, 71), (54, 74), (53, 81), (52, 96), (51, 103), (50, 115), (49, 130), (48, 142), (47, 152), (46, 144), (45, 171), (44, 182), (43, 196), (42, 230), (41, 232), (40, 248), (39, 262), (38, 304), (37, 365), (36, 392), (35, 403), (34, 438), (33, 456), (32, 522), (31, 539), (30, 605), (29, 653), (28, 717), (27, 759), (26, 824), (25, 917), (24, 953), (23, 1018), (22, 1174), (21, 1241), (20, 1305), (19, 1416), (18, 1521), (17, 1688), (16, 1816), (15, 1901), (14, 2202), (13, 2348), (12, 2595), (11, 2679), (10, 2753), (9, 2713), (8, 2733), (7, 2518), (6, 2218), (5, 1509), (4, 692), (3, 932), (2, 175)]
    seq length <=205 num: 51785
    
    验证数据: 
    N_valid = 1193  
    
    source sentence length distribution:
    most common seq length: (seq length, count num)
    [(9, 76), (12, 75), (8, 72), (7, 61), (13, 59), (14, 59), (11, 59), (10, 57), (6, 52), (15, 45)]
    seq length count:  (seq length, count num)
    [(120, 1), (86, 1), (80, 1), (78, 2), (76, 1), (74, 1), (70, 2), (69, 2), (68, 1), (66, 1), (62, 2), (60, 1), (59, 2), (55, 2), (54, 2), (53, 3), (52, 2), (51, 1), (50, 3), (49, 2), (46, 5), (45, 3), (43, 1), (42, 5), (41, 11), (40, 5), (39, 1), (38, 3), (37, 6), (36, 10), (35, 2), (34, 7), (33, 10), (32, 5), (31, 12), (30, 10), (29, 15), (28, 13), (27, 15), (26, 20), (25, 20), (24, 23), (23, 19), (22, 27), (21, 31), (20, 41), (19, 29), (18, 34), (17, 35), (16, 28), (15, 45), (14, 59), (13, 59), (12, 75), (11, 59), (10, 57), (9, 76), (8, 72), (7, 61), (6, 52), (5, 39), (4, 25), (3, 26), (2, 9)]
    target sentence length distribution:
    most common seq length: (seq length, count num)
    [(11, 76), (6, 73), (10, 64), (12, 64), (14, 60), (7, 60), (9, 56), (8, 55), (15, 53), (16, 47)]
    seq length count:  (seq length, count num)
    [(114, 1), (87, 2), (77, 1), (72, 1), (68, 1), (67, 1), (66, 2), (62, 1), (61, 2), (59, 1), (58, 2), (57, 2), (56, 2), (54, 4), (53, 2), (51, 1), (50, 1), (49, 2), (48, 2), (47, 1), (46, 3), (45, 3), (44, 1), (43, 8), (42, 5), (41, 5), (40, 2), (39, 4), (38, 7), (37, 6), (36, 8), (35, 7), (34, 4), (33, 12), (32, 6), (31, 8), (30, 15), (29, 10), (28, 15), (27, 17), (26, 22), (25, 20), (24, 27), (23, 29), (22, 27), (21, 24), (20, 20), (19, 36), (18, 35), (17, 44), (16, 47), (15, 53), (14, 60), (13, 42), (12, 64), (11, 76), (10, 64), (9, 56), (8, 55), (7, 60), (6, 73), (5, 43), (4, 13), (3, 23), (2, 2)]
    seq length <=205 num: 1193

    测试数据: 
    N_test = 1803 
    
    source sentence length distribution:
    most common seq length: (seq length, count num)
    [(11, 91), (13, 91), (8, 90), (7, 88), (10, 87), (9, 87), (14, 82), (12, 77), (6, 76), (15, 72)]
    seq length count:  (seq length, count num)
    [(115, 1), (113, 2), (105, 1), (100, 1), (83, 1), (81, 1), (79, 1), (77, 1), (74, 1), (73, 1), (71, 2), (70, 1), (69, 2), (68, 1), (67, 2), (66, 1), (64, 1), (62, 1), (60, 1), (59, 1), (58, 3), (57, 2), (56, 1), (55, 3), (54, 2), (53, 2), (52, 6), (51, 1), (50, 5), (49, 3), (48, 5), (47, 3), (46, 5), (45, 7), (44, 3), (43, 8), (42, 12), (41, 8), (40, 9), (39, 11), (38, 13), (37, 11), (36, 12), (35, 11), (34, 14), (33, 17), (32, 21), (31, 21), (30, 21), (29, 26), (28, 25), (27, 19), (26, 42), (25, 30), (24, 29), (23, 42), (22, 45), (21, 44), (20, 47), (19, 40), (18, 61), (17, 56), (16, 61), (15, 72), (14, 82), (13, 91), (12, 77), (11, 91), (10, 87), (9, 87), (8, 90), (7, 88), (6, 76), (5, 45), (4, 34), (3, 39), (2, 13)]
    target sentence length distribution:
    most common seq length: (seq length, count num)
    [(8, 92), (9, 92), (10, 91), (15, 87), (13, 82), (14, 81), (11, 81), (12, 81), (17, 74), (16, 70)]
    seq length count:  (seq length, count num)
    [(135, 1), (121, 1), (119, 1), (105, 2), (92, 1), (87, 1), (85, 1), (82, 1), (79, 1), (77, 1), (76, 1), (75, 2), (71, 1), (69, 2), (68, 2), (66, 1), (65, 2), (62, 2), (61, 1), (60, 3), (58, 3), (57, 2), (56, 5), (55, 2), (53, 5), (52, 3), (51, 5), (50, 1), (49, 2), (48, 5), (47, 3), (46, 10), (45, 9), (44, 6), (43, 11), (42, 4), (41, 8), (40, 5), (39, 12), (38, 12), (37, 11), (36, 15), (35, 17), (34, 14), (33, 21), (32, 21), (31, 19), (30, 22), (29, 29), (28, 29), (27, 26), (26, 23), (25, 42), (24, 29), (23, 39), (22, 49), (21, 48), (20, 43), (19, 43), (18, 60), (17, 74), (16, 70), (15, 87), (14, 81), (13, 82), (12, 81), (11, 81), (10, 91), (9, 92), (8, 92), (7, 70), (6, 50), (5, 49), (4, 24), (3, 34), (2, 4)]
    seq length <=205 num: 1803
    
    (2) 数据预处理
    
    未做 unicode 标准化
    
    源语言词表大小: n_vocab_source=8000
    目标语言词表大小: n_vocab_target=8000
    
    
    (3) 优化器参数
    
    epoch_num = 20
    batch_size = 64
    optimizer= Adam with warmup_steps
    warmup_steps = 4000
    
    (5) 训练过程
    
    在每一个 epoch 结束时都对模型进行持久化(checkpoint), 并计算在验证集上的 bleu 得分
    
    Epoch 1/20
    
    bleu_score:{'1-garm': 0.0950721153846154, '2-garm': 0.023533219016160366, '3-garm': 0.006305676428705639, '4-garm': 0.001703890249600079}
    810/810 [==============================] - 87s 100ms/step - loss: 2.1029 - accuracy: 0.0583 - val_loss: 1.5307 - val_accuracy: 0.2121
    
    Epoch 2/20
    
    bleu_score:{'1-garm': 0.13347028065307862, '2-garm': 0.044761776809654874, '3-garm': 0.016925244630941784, '4-garm': 0.005622638990346062}
    810/810 [==============================] - 75s 92ms/step - loss: 1.3948 - accuracy: 0.2268 - val_loss: 1.4048 - val_accuracy: 0.2711
    
    Epoch 3/20
    
    bleu_score:{'1-garm': 0.20908289889125486, '2-garm': 0.08780961587537969, '3-garm': 0.040913224993288866, '4-garm': 0.01929513113823629}
    810/810 [==============================] - 70s 86ms/step - loss: 1.2673 - accuracy: 0.2751 - val_loss: 1.2028 - val_accuracy: 0.3174
    
    Epoch 4/20
    
    bleu_score:{'1-garm': 0.3150774685886071, '2-garm': 0.17356526700500566, '3-garm': 0.10216469956023307, '4-garm': 0.06062491914741682}
    810/810 [==============================] - 69s 85ms/step - loss: 1.1123 - accuracy: 0.3266 - val_loss: 1.0319 - val_accuracy: 0.3956
    
    Epoch 5/20
    
    bleu_score:{'1-garm': 0.3887401638679322, '2-garm': 0.2438906045566233, '3-garm': 0.16088018314936284, '4-garm': 0.1075484286140663}
    810/810 [==============================] - 67s 83ms/step - loss: 0.9841 - accuracy: 0.3900 - val_loss: 0.9395 - val_accuracy: 0.4436
    
    Epoch 6/20
    
    bleu_score:{'1-garm': 0.41374917975836645, '2-garm': 0.2786933749980563, '3-garm': 0.19547523986347515, '4-garm': 0.13941066488445936}
    810/810 [==============================] - 71s 88ms/step - loss: 0.8623 - accuracy: 0.4372 - val_loss: 0.8218 - val_accuracy: 0.4927
    
    Epoch 7/20
    
    bleu_score:{'1-garm': 0.46609887827170754, '2-garm': 0.3224701912056759, '3-garm': 0.2329304338831242, '4-garm': 0.17054330304997625}
    810/810 [==============================] - 68s 83ms/step - loss: 0.7629 - accuracy: 0.4886 - val_loss: 0.7780 - val_accuracy: 0.5225
    
    Epoch 8/20
    
    bleu_score:{'1-garm': 0.5041036717062635, '2-garm': 0.3583543426523779, '3-garm': 0.26398906370832165, '4-garm': 0.1972279454182121}
    810/810 [==============================] - 62s 76ms/step - loss: 0.6784 - accuracy: 0.5267 - val_loss: 0.7465 - val_accuracy: 0.5417
    
    Epoch 9/20
    
    bleu_score:{'1-garm': 0.5158999691262736, '2-garm': 0.37220935973146363, '3-garm': 0.27732812442174365, '4-garm': 0.2095753563265695}
    810/810 [==============================] - 65s 80ms/step - loss: 0.6208 - accuracy: 0.5528 - val_loss: 0.7027 - val_accuracy: 0.5584
    
    Epoch 10/20
    
    bleu_score:{'1-garm': 0.5468793214713745, '2-garm': 0.39819469553855735, '3-garm': 0.30050151776265877, '4-garm': 0.229900977777386}
    810/810 [==============================] - 62s 77ms/step - loss: 0.5743 - accuracy: 0.5753 - val_loss: 0.6888 - val_accuracy: 0.5696
    
    Epoch 11/20
    
    bleu_score:{'1-garm': 0.5498709648209353, '2-garm': 0.40490910927650936, '3-garm': 0.308224601112564, '4-garm': 0.23762156983816013}
    810/810 [==============================] - 61s 75ms/step - loss: 0.5471 - accuracy: 0.5930 - val_loss: 0.6680 - val_accuracy: 0.5794
    
    Epoch 12/20
    
    bleu_score:{'1-garm': 0.5545222465353756, '2-garm': 0.4080943605655203, '3-garm': 0.3097736691019512, '4-garm': 0.23834790106390066}
    810/810 [==============================] - 61s 75ms/step - loss: 0.5183 - accuracy: 0.6064 - val_loss: 0.6476 - val_accuracy: 0.5846
    
    Epoch 13/20
    
    bleu_score:{'1-garm': 0.5623782580465634, '2-garm': 0.41665305050057877, '3-garm': 0.3195802252527693, '4-garm': 0.24890236014541978}
    810/810 [==============================] - 61s 75ms/step - loss: 0.5032 - accuracy: 0.6199 - val_loss: 0.6744 - val_accuracy: 0.5889
    
    Epoch 14/20
    
    bleu_score:{'1-garm': 0.5746005202526941, '2-garm': 0.42689980072610434, '3-garm': 0.32795752134040035, '4-garm': 0.2553205865671538}
    810/810 [==============================] - 61s 76ms/step - loss: 0.4731 - accuracy: 0.6291 - val_loss: 0.6582 - val_accuracy: 0.5921
    
    Epoch 15/20
    
    bleu_score:{'1-garm': 0.5778877887788779, '2-garm': 0.431035834422657, '3-garm': 0.33210234136797506, '4-garm': 0.2580143450387721}
    810/810 [==============================] - 61s 76ms/step - loss: 0.4591 - accuracy: 0.6392 - val_loss: 0.6433 - val_accuracy: 0.5973
    
    Epoch 16/20
    
    bleu_score:{'1-garm': 0.5580875781948168, '2-garm': 0.4145685374159218, '3-garm': 0.3180673526746957, '4-garm': 0.24770131107030893}
    810/810 [==============================] - 60s 75ms/step - loss: 0.4498 - accuracy: 0.6478 - val_loss: 0.6343 - val_accuracy: 0.5990
    
    Epoch 17/20
    
    bleu_score:{'1-garm': 0.5742831915981226, '2-garm': 0.43013785232535934, '3-garm': 0.3318480242635666, '4-garm': 0.2595334204206681}
    810/810 [==============================] - 59s 72ms/step - loss: 0.4401 - accuracy: 0.6547 - val_loss: 0.6292 - val_accuracy: 0.6020
    
    Epoch 18/20
    
    bleu_score:{'1-garm': 0.5814530474674656, '2-garm': 0.4363803589260646, '3-garm': 0.33832159625889013, '4-garm': 0.26628792688904696}
    810/810 [==============================] - 61s 75ms/step - loss: 0.4189 - accuracy: 0.6632 - val_loss: 0.6103 - val_accuracy: 0.6021
    
    Epoch 19/20
    
    bleu_score:{'1-garm': 0.5798799924995313, '2-garm': 0.4336658576218962, '3-garm': 0.3350765216151607, '4-garm': 0.26165638955631937}
    810/810 [==============================] - 61s 75ms/step - loss: 0.4138 - accuracy: 0.6694 - val_loss: 0.6554 - val_accuracy: 0.6059
    
    Epoch 20/20
    
    candidates:
    [0] that ' s twice as long as the existence of men on this planet .
    [1] where do you find these glucket conditions ?
    [2] but of course , life is more than the chemistry of chemical .
    [3] every troubling of information .
    [4] i was n ' t an activist .
    [5] the fight for rights equality has n ' t only been seeing with homosexual marriage .
    [6] in fact , by the contrary , i ' ll pass them .
    [7] this was n ' t a bor nightmare .
    [8] this is a family portrait of family .
    [9] we do n ' t have freedom , they said .
    
    references:
    [0] ["that 's twice as long as humans have been on this planet ."]
    [1] ['now , where do you find such goldilocks conditions ?']
    [2] ['but of course , life is more than just exotic chemistry .']
    [3] ['each rung contains information .']
    [4] ['i was not an activist .']
    [5] ['the fight for equal rights is not just about gay marriage .']
    [6] ['in fact , on the contrary , it highlights them .']
    [7] ["this was n't a brothel , per se ."]
    [8] ['this is a family portrait .']
    [9] ["`` `` '' we have no freedom , '' '' they said . ''"]
    
    bleu_score:{'1-garm': 0.5817757009345794, '2-garm': 0.4371124884124944, '3-garm': 0.3387078414206649, '4-garm': 0.2659632358184676}
    810/810 [==============================] - 59s 73ms/step - loss: 0.3995 - accuracy: 0.6744 - val_loss: 0.6018 - val_accuracy: 0.6064

        
    (6) 模型评价
    
    1.epoch=20 时的模型
    
    candidates:
    [0] until the end of this year , there will almost a million people in this planet running over the educationally by social media .
    [1] the only thing they all have in common and they will die .
    [2] although that power is a thought to be a lot of thinking or how much morva , i think it has very profound implications that are worth exploring .
    [3] so what made me think about this was a post - auth in a second grade , the end of the year of the mahark of kler , which was a journalist of science and technology that was a cancer doctor that was dead .
    [4] and what mlerney did , was to do with her family and writes a postt a post that was published a little bit after his die .
    [5] here ' s what he wrote to start that process .
    [6] ` ` he said , ` ` ' ' here i ' m dead . i ' m this is my last little posting in my blog . ' '
    [7] ` ` before i asked more , i asked that my body was finally going to be down because of my torcious cancer , at that moment my family and public friends and i ' ve written that i ' ve written the first part of the web of the web of the text , in archavir . ' ' ' ' now as a journalist , as a single one , the best journalist of the misquirequirequirequirement is written for medicatingly , has been more than ever been being made of any more than ever been able to create , and more than ever done today than ever before we ' ve done any more than ever done with any more than ever - generation every day than any more than ever done . ' '
    [8] consider some data by a moment .
    [9] now there are 48 hours of video being shabs for youtube every minute .
    
    references:
    [0] ["by the end of this year , there 'll be nearly a billion people on this planet that actively use social networking sites ."]
    [1] ["the one thing that all of them have in common is that they 're going to die ."]
    [2] ['while that might be a somewhat morbid thought , i think it has some really profound implications that are worth exploring .']
    [3] ['what first got me thinking about this was a blog post authored earlier this year by derek k. miller , who was a science and technology journalist who died of cancer .']
    [4] ['and what miller did was have his family and friends write a post that went out shortly after he died .']
    [5] ["here 's what he wrote in starting that out ."]
    [6] ["`` he said , `` '' here it is . i 'm dead , and this is my last post to my blog . ''"]
    [7] ["`` in advance , i asked that once my body finally shut down from the punishments of my cancer , then my family and friends publish this prepared message i wrote — the first part of the process of turning this from an active website to an archive . '' '' now , while as a journalist , miller 's archive may have been better written and more carefully curated than most , the fact of the matter is that all of us today are creating an archive that 's something completely different than anything that 's been created by any previous generation . ''"]
    [8] ['consider a few stats for a moment .']
    [9] ['right now there are 48 hours of video being uploaded to youtube every single minute .']
    bleu_score:{'1-garm': 0.573685690188361, '2-garm': 0.428009723122818, '3-garm': 0.3285183096153272, '4-garm': 0.2545992035470089}
    
   
#### 实验 2  动态 batch size
    
    (0) 模型
       
    
    (1) 数据集 
    
    训练数据:    
    N_train = 51785 ( 源句子-目标句子对 的数目 )
    
    source sentence length distribution:
    most common seq length: (seq length, count num)
    [(8, 2956), (9, 2880), (10, 2839), (11, 2614), (7, 2607), (6, 2526), (12, 2494), (13, 2244), (14, 2133), (5, 1816)]
    seq length count:  (seq length, count num)
    [(191, 1), (188, 1), (176, 1), (152, 1), (149, 1), (146, 1), (145, 2), (143, 1), (141, 1), (139, 1), (137, 1), (136, 1), (135, 2), (134, 1), (133, 1), (131, 2), (130, 1), (129, 1), (127, 2), (125, 6), (124, 3), (123, 2), (122, 1), (121, 2), (120, 5), (119, 3), (117, 4), (116, 2), (115, 1), (114, 4), (113, 1), (112, 3), (111, 1), (109, 3), (108, 3), (107, 5), (106, 2), (105, 5), (104, 10), (103, 6), (102, 5), (101, 4), (100, 5), (99, 5), (98, 5), (97, 1), (96, 3), (95, 3), (94, 4), (93, 4), (92, 5), (91, 8), (90, 5), (89, 6), (88, 7), (87, 8), (86, 14), (85, 10), (84, 11), (83, 14), (82, 12), (81, 15), (80, 16), (79, 12), (78, 12), (77, 9), (76, 15), (75, 18), (74, 18), (73, 21), (72, 24), (71, 27), (70, 29), (69, 28), (68, 27), (67, 22), (66, 31), (65, 32), (64, 41), (63, 42), (62, 42), (61, 47), (60, 50), (59, 47), (58, 50), (57, 70), (56, 58), (55, 57), (54, 88), (53, 90), (52, 70), (51, 103), (50, 104), (49, 108), (48, 124), (47, 149), (46, 153), (45, 166), (44, 172), (43, 197), (42, 197), (41, 233), (40, 240), (39, 262), (38, 288), (37, 321), (36, 334), (35, 355), (34, 430), (33, 440), (32, 489), (31, 539), (30, 576), (29, 610), (28, 632), (27, 696), (26, 760), (25, 901), (24, 936), (23, 988), (22, 1064), (21, 1182), (20, 1214), (19, 1384), (18, 1443), (17, 1541), (16, 1673), (15, 1778), (14, 2133), (13, 2244), (12, 2494), (11, 2614), (10, 2839), (9, 2880), (8, 2956), (7, 2607), (6, 2526), (5, 1816), (4, 1085), (3, 977), (2, 486), (1, 4)]
    target sentence length distribution:
    most common seq length: (seq length, count num)
    [(10, 2753), (8, 2733), (9, 2713), (11, 2679), (12, 2595), (7, 2518), (13, 2348), (6, 2218), (14, 2202), (15, 1901)]
    seq length count:  (seq length, count num)
    [(204, 1), (199, 1), (174, 1), (166, 1), (159, 1), (156, 1), (155, 1), (149, 1), (148, 1), (145, 1), (143, 3), (142, 3), (137, 1), (136, 1), (134, 3), (133, 2), (132, 1), (131, 1), (130, 2), (128, 2), (127, 1), (126, 3), (125, 1), (124, 4), (123, 3), (122, 1), (121, 7), (120, 5), (119, 5), (118, 2), (117, 3), (116, 2), (115, 3), (114, 3), (112, 2), (111, 5), (110, 3), (109, 1), (108, 5), (107, 3), (106, 2), (105, 2), (104, 2), (103, 5), (102, 5), (101, 4), (100, 3), (99, 7), (98, 4), (97, 5), (96, 5), (95, 8), (94, 3), (93, 6), (92, 8), (91, 5), (90, 6), (89, 7), (88, 5), (87, 10), (86, 13), (85, 8), (84, 13), (83, 9), (82, 12), (81, 13), (80, 12), (79, 14), (78, 20), (77, 21), (76, 17), (75, 15), (74, 17), (73, 25), (72, 33), (71, 36), (70, 25), (69, 35), (68, 35), (67, 29), (66, 39), (65, 36), (64, 37), (63, 42), (62, 57), (61, 59), (60, 44), (59, 66), (58, 72), (57, 56), (56, 59), (55, 71), (54, 74), (53, 81), (52, 96), (51, 103), (50, 115), (49, 130), (48, 142), (47, 152), (46, 144), (45, 171), (44, 182), (43, 196), (42, 230), (41, 232), (40, 248), (39, 262), (38, 304), (37, 365), (36, 392), (35, 403), (34, 438), (33, 456), (32, 522), (31, 539), (30, 605), (29, 653), (28, 717), (27, 759), (26, 824), (25, 917), (24, 953), (23, 1018), (22, 1174), (21, 1241), (20, 1305), (19, 1416), (18, 1521), (17, 1688), (16, 1816), (15, 1901), (14, 2202), (13, 2348), (12, 2595), (11, 2679), (10, 2753), (9, 2713), (8, 2733), (7, 2518), (6, 2218), (5, 1509), (4, 692), (3, 932), (2, 175)]
    seq length <=205 num: 51785
    
    验证数据: 
    N_valid = 1193  
    
    source sentence length distribution:
    most common seq length: (seq length, count num)
    [(9, 76), (12, 75), (8, 72), (7, 61), (13, 59), (14, 59), (11, 59), (10, 57), (6, 52), (15, 45)]
    seq length count:  (seq length, count num)
    [(120, 1), (86, 1), (80, 1), (78, 2), (76, 1), (74, 1), (70, 2), (69, 2), (68, 1), (66, 1), (62, 2), (60, 1), (59, 2), (55, 2), (54, 2), (53, 3), (52, 2), (51, 1), (50, 3), (49, 2), (46, 5), (45, 3), (43, 1), (42, 5), (41, 11), (40, 5), (39, 1), (38, 3), (37, 6), (36, 10), (35, 2), (34, 7), (33, 10), (32, 5), (31, 12), (30, 10), (29, 15), (28, 13), (27, 15), (26, 20), (25, 20), (24, 23), (23, 19), (22, 27), (21, 31), (20, 41), (19, 29), (18, 34), (17, 35), (16, 28), (15, 45), (14, 59), (13, 59), (12, 75), (11, 59), (10, 57), (9, 76), (8, 72), (7, 61), (6, 52), (5, 39), (4, 25), (3, 26), (2, 9)]
    target sentence length distribution:
    most common seq length: (seq length, count num)
    [(11, 76), (6, 73), (10, 64), (12, 64), (14, 60), (7, 60), (9, 56), (8, 55), (15, 53), (16, 47)]
    seq length count:  (seq length, count num)
    [(114, 1), (87, 2), (77, 1), (72, 1), (68, 1), (67, 1), (66, 2), (62, 1), (61, 2), (59, 1), (58, 2), (57, 2), (56, 2), (54, 4), (53, 2), (51, 1), (50, 1), (49, 2), (48, 2), (47, 1), (46, 3), (45, 3), (44, 1), (43, 8), (42, 5), (41, 5), (40, 2), (39, 4), (38, 7), (37, 6), (36, 8), (35, 7), (34, 4), (33, 12), (32, 6), (31, 8), (30, 15), (29, 10), (28, 15), (27, 17), (26, 22), (25, 20), (24, 27), (23, 29), (22, 27), (21, 24), (20, 20), (19, 36), (18, 35), (17, 44), (16, 47), (15, 53), (14, 60), (13, 42), (12, 64), (11, 76), (10, 64), (9, 56), (8, 55), (7, 60), (6, 73), (5, 43), (4, 13), (3, 23), (2, 2)]
    seq length <=205 num: 1193

    测试数据: 
    N_test = 1803 
    
    source sentence length distribution:
    most common seq length: (seq length, count num)
    [(11, 91), (13, 91), (8, 90), (7, 88), (10, 87), (9, 87), (14, 82), (12, 77), (6, 76), (15, 72)]
    seq length count:  (seq length, count num)
    [(115, 1), (113, 2), (105, 1), (100, 1), (83, 1), (81, 1), (79, 1), (77, 1), (74, 1), (73, 1), (71, 2), (70, 1), (69, 2), (68, 1), (67, 2), (66, 1), (64, 1), (62, 1), (60, 1), (59, 1), (58, 3), (57, 2), (56, 1), (55, 3), (54, 2), (53, 2), (52, 6), (51, 1), (50, 5), (49, 3), (48, 5), (47, 3), (46, 5), (45, 7), (44, 3), (43, 8), (42, 12), (41, 8), (40, 9), (39, 11), (38, 13), (37, 11), (36, 12), (35, 11), (34, 14), (33, 17), (32, 21), (31, 21), (30, 21), (29, 26), (28, 25), (27, 19), (26, 42), (25, 30), (24, 29), (23, 42), (22, 45), (21, 44), (20, 47), (19, 40), (18, 61), (17, 56), (16, 61), (15, 72), (14, 82), (13, 91), (12, 77), (11, 91), (10, 87), (9, 87), (8, 90), (7, 88), (6, 76), (5, 45), (4, 34), (3, 39), (2, 13)]
    target sentence length distribution:
    most common seq length: (seq length, count num)
    [(8, 92), (9, 92), (10, 91), (15, 87), (13, 82), (14, 81), (11, 81), (12, 81), (17, 74), (16, 70)]
    seq length count:  (seq length, count num)
    [(135, 1), (121, 1), (119, 1), (105, 2), (92, 1), (87, 1), (85, 1), (82, 1), (79, 1), (77, 1), (76, 1), (75, 2), (71, 1), (69, 2), (68, 2), (66, 1), (65, 2), (62, 2), (61, 1), (60, 3), (58, 3), (57, 2), (56, 5), (55, 2), (53, 5), (52, 3), (51, 5), (50, 1), (49, 2), (48, 5), (47, 3), (46, 10), (45, 9), (44, 6), (43, 11), (42, 4), (41, 8), (40, 5), (39, 12), (38, 12), (37, 11), (36, 15), (35, 17), (34, 14), (33, 21), (32, 21), (31, 19), (30, 22), (29, 29), (28, 29), (27, 26), (26, 23), (25, 42), (24, 29), (23, 39), (22, 49), (21, 48), (20, 43), (19, 43), (18, 60), (17, 74), (16, 70), (15, 87), (14, 81), (13, 82), (12, 81), (11, 81), (10, 91), (9, 92), (8, 92), (7, 70), (6, 50), (5, 49), (4, 24), (3, 34), (2, 4)]
    seq length <=205 num: 1803
    
    (2) 数据预处理
    
    未做 unicode 标准化
    
    源语言词表大小: n_vocab_source=8000
    目标语言词表大小: n_vocab_target=8000
    
    
    (3) 优化器参数
    
    epoch_num = 20
    token_in_batch = 2048
    
    optimizer= Adam with warmup_steps
    warmup_steps = 4000
    
    (5) 训练过程
    
    在每一个 epoch 结束时都对模型进行持久化(checkpoint), 并计算在验证集上的 bleu 得分
    
    Epoch 1/20
    
    candidates:
    [0] and i ' s the world , and the world .
    [1] and i ' s the world .
    [2] and i ' s the world , and the world .
    [3] and i ' s the .
    [4] and i ' s the .
    [5] and i ' s the world , and the world .
    [6] and i ' s the world , and the world .
    [7] and i ' s the world , and the world .
    [8] and i ' s the .
    [9] and i ' s the .
    
    references:
    [0] ["that 's twice as long as humans have been on this planet ."]
    [1] ['now , where do you find such goldilocks conditions ?']
    [2] ['but of course , life is more than just exotic chemistry .']
    [3] ['each rung contains information .']
    [4] ['i was not an activist .']
    [5] ['the fight for equal rights is not just about gay marriage .']
    [6] ['in fact , on the contrary , it highlights them .']
    [7] ["this was n't a brothel , per se ."]
    [8] ['this is a family portrait .']
    [9] ["`` `` '' we have no freedom , '' '' they said . ''"]
    
    bleu_score:{'1-garm': 0.09441699085269686, '2-garm': 0.024689166977374315, '3-garm': 0.006872101012598311, '4-garm': 2.915098666321272e-79}
    747/747 [==============================] - 68s 82ms/step - loss: 7.1331 - accuracy: 0.0523 - val_loss: 4.9356 - val_accuracy: 0.1999
    
    Epoch 2/20
    
    candidates:
    [0] the first is a lot of the world .
    [1] so what ' s the world ?
    [2] but i ' m going to be a lot of the world .
    [3] it ' s a lot of the world .
    [4] i ' m a lot of a lot of a lot of a lot of .
    [5] we ' re a lot of the world .
    [6] so , we ' re a lot of the world .
    [7] i ' m a lot of a lot of a lot of a lot of a lot of a lot of a lot of .
    [8] this is a lot of a lot of a lot of a lot of a lot .
    [9] ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' '
    
    references:
    [0] ["that 's twice as long as humans have been on this planet ."]
    [1] ['now , where do you find such goldilocks conditions ?']
    [2] ['but of course , life is more than just exotic chemistry .']
    [3] ['each rung contains information .']
    [4] ['i was not an activist .']
    [5] ['the fight for equal rights is not just about gay marriage .']
    [6] ['in fact , on the contrary , it highlights them .']
    [7] ["this was n't a brothel , per se ."]
    [8] ['this is a family portrait .']
    [9] ["`` `` '' we have no freedom , '' '' they said . ''"]
    
    bleu_score:{'1-garm': 0.11835349403881297, '2-garm': 0.03631570746619619, '3-garm': 0.014218694089028127, '4-garm': 0.004858121037843629}
    747/747 [==============================] - 61s 81ms/step - loss: 4.7274 - accuracy: 0.2077 - val_loss: 4.3048 - val_accuracy: 0.2552
    
    Epoch 3/20
    
    candidates:
    [0] and the world is the world of the world of the world .
    [1] how we have the world ?
    [2] but i ' m going to be the world , but we ' re going to be the world , but the world is the world .
    [3] it ' s the same .
    [4] i was a little bit of my life .
    [5] the same is not the same thing to the world .
    [6] the first thing is , the same is the same , the same is the same .
    [7] this was a little bit of a little bit of my life , and it was a little bit of a little bit of a little bit of my life .
    [8] this is a little bit of the world .
    [9] ` ` ` ` ' ' ' we ' re not going to be able to do it . ' ' ' '
    
    references:
    [0] ["that 's twice as long as humans have been on this planet ."]
    [1] ['now , where do you find such goldilocks conditions ?']
    [2] ['but of course , life is more than just exotic chemistry .']
    [3] ['each rung contains information .']
    [4] ['i was not an activist .']
    [5] ['the fight for equal rights is not just about gay marriage .']
    [6] ['in fact , on the contrary , it highlights them .']
    [7] ["this was n't a brothel , per se ."]
    [8] ['this is a family portrait .']
    [9] ["`` `` '' we have no freedom , '' '' they said . ''"]
    
    bleu_score:{'1-garm': 0.09840627808891653, '2-garm': 0.03548071026115966, '3-garm': 0.014663955257612384, '4-garm': 0.0055202457815427914}
    747/747 [==============================] - 59s 79ms/step - loss: 4.2084 - accuracy: 0.2604 - val_loss: 4.0587 - val_accuracy: 0.2780
    
    Epoch 4/20
    
    candidates:
    [0] that ' s more than the most of the most of the world in the world .
    [1] what are they going to be able to be able to be able to be ?
    [2] but in fact , the world is the best thing that i ' m going to be able to be able to be .
    [3] we ' ve got to be able to be able to be important .
    [4] i was n ' t a good woman .
    [5] the most of the most of the most of the most of the best of the sccregits .
    [6] in fact , the same , the same , the same , the way .
    [7] this was n ' t a scre , a good , and i was .
    [8] this is a friend of her family .
    [9] we ' re not going to be able to be able to be .
    
    references:
    [0] ["that 's twice as long as humans have been on this planet ."]
    [1] ['now , where do you find such goldilocks conditions ?']
    [2] ['but of course , life is more than just exotic chemistry .']
    [3] ['each rung contains information .']
    [4] ['i was not an activist .']
    [5] ['the fight for equal rights is not just about gay marriage .']
    [6] ['in fact , on the contrary , it highlights them .']
    [7] ["this was n't a brothel , per se ."]
    [8] ['this is a family portrait .']
    [9] ["`` `` '' we have no freedom , '' '' they said . ''"]
    
    bleu_score:{'1-garm': 0.17345560931697984, '2-garm': 0.07980689308103255, '3-garm': 0.04099381211715395, '4-garm': 0.020800643707617396}
    747/747 [==============================] - 58s 77ms/step - loss: 3.9167 - accuracy: 0.2880 - val_loss: 3.6606 - val_accuracy: 0.3340
    
    Epoch 5/20
    
    candidates:
    [0] that ' s been made of the time of the world of the planet .
    [1] where do you get these these are in the dys ?
    [2] but in fact , the most of the most of the mare .
    [3] every single information .
    [4] i was n ' t a man .
    [5] the politicality of violence has never been going to see the polpol .
    [6] in fact , the fact , the real thing is .
    [7] this was not a propropre , it was n ' t the wrong .
    [8] this is a family .
    [9] ` ` ` ` ` ' ' do n ' t have to help . ' ' '
    
    references:
    [0] ["that 's twice as long as humans have been on this planet ."]
    [1] ['now , where do you find such goldilocks conditions ?']
    [2] ['but of course , life is more than just exotic chemistry .']
    [3] ['each rung contains information .']
    [4] ['i was not an activist .']
    [5] ['the fight for equal rights is not just about gay marriage .']
    [6] ['in fact , on the contrary , it highlights them .']
    [7] ["this was n't a brothel , per se ."]
    [8] ['this is a family portrait .']
    [9] ["`` `` '' we have no freedom , '' '' they said . ''"]
    
    bleu_score:{'1-garm': 0.30917490249536145, '2-garm': 0.17213597320506777, '3-garm': 0.10237440145688813, '4-garm': 0.061490361093782676}
    747/747 [==============================] - 56s 75ms/step - loss: 3.4990 - accuracy: 0.3432 - val_loss: 3.3327 - val_accuracy: 0.3874
    
    Epoch 6/20
    
    candidates:
    [0] that ' s the most exciting thing of the human life of the planet in this planet .
    [1] where do you find these conditions ?
    [2] but of course , life is more than memorta .
    [3] each human information .
    [4] i was n ' t a study .
    [5] the fight for education has n ' t only to see the proproponds with the propos .
    [6] in fact , in fact , the real , the real .
    [7] this was n ' t a biotront , it said .
    [8] this is a family .
    [9] ` ` ` ` ' ' we do n ' t have freedom . ' ' ' '
    
    references:
    [0] ["that 's twice as long as humans have been on this planet ."]
    [1] ['now , where do you find such goldilocks conditions ?']
    [2] ['but of course , life is more than just exotic chemistry .']
    [3] ['each rung contains information .']
    [4] ['i was not an activist .']
    [5] ['the fight for equal rights is not just about gay marriage .']
    [6] ['in fact , on the contrary , it highlights them .']
    [7] ["this was n't a brothel , per se ."]
    [8] ['this is a family portrait .']
    [9] ["`` `` '' we have no freedom , '' '' they said . ''"]
    
    bleu_score:{'1-garm': 0.3835532371893219, '2-garm': 0.24463231060136542, '3-garm': 0.1651770777773743, '4-garm': 0.11280059680557646}
    747/747 [==============================] - 56s 75ms/step - loss: 3.0804 - accuracy: 0.4038 - val_loss: 2.9595 - val_accuracy: 0.4494
    
    Epoch 7/20
    
    candidates:
    [0] that ' s about twice the fact of men in this planet .
    [1] where do you find these conditions ?
    [2] but of course , life is more than the chemistry of chemistry .
    [3] every single information has different information .
    [4] i was n ' t a activist .
    [5] the political struggle not only to see the wedding in the french .
    [6] in fact , the fact , the real , the self .
    [7] this was not a buce , exactly , that .
    [8] this is a family .
    [9] ` ` ` ` ' ' we do n ' t have freedom , ' ' ' ' '
    
    references:
    [0] ["that 's twice as long as humans have been on this planet ."]
    [1] ['now , where do you find such goldilocks conditions ?']
    [2] ['but of course , life is more than just exotic chemistry .']
    [3] ['each rung contains information .']
    [4] ['i was not an activist .']
    [5] ['the fight for equal rights is not just about gay marriage .']
    [6] ['in fact , on the contrary , it highlights them .']
    [7] ["this was n't a brothel , per se ."]
    [8] ['this is a family portrait .']
    [9] ["`` `` '' we have no freedom , '' '' they said . ''"]
    
    bleu_score:{'1-garm': 0.41896739349482087, '2-garm': 0.2799310081035386, '3-garm': 0.1942301942549813, '4-garm': 0.136743556475949}
    747/747 [==============================] - 57s 77ms/step - loss: 2.7152 - accuracy: 0.4549 - val_loss: 2.7447 - val_accuracy: 0.4756
    
    Epoch 8/20
    
    candidates:
    [0] that ' s twice by the time of the men in this planet .
    [1] where are you find these nanomorlin conditions ?
    [2] but of course , life is more than the chemistry of chemistry that impologen .
    [3] every trosellation contains information .
    [4] i was n ' t a activist .
    [5] the fight for equality is not just to see the wedding in exile .
    [6] in fact , in the contrary , real , real , impaired .
    [7] this was n ' t a bromoth , exactly said .
    [8] this is a family of family .
    [9] ` ` ` ` ' ' we do n ' t have freedom , they said . ' ' '
    
    references:
    [0] ["that 's twice as long as humans have been on this planet ."]
    [1] ['now , where do you find such goldilocks conditions ?']
    [2] ['but of course , life is more than just exotic chemistry .']
    [3] ['each rung contains information .']
    [4] ['i was not an activist .']
    [5] ['the fight for equal rights is not just about gay marriage .']
    [6] ['in fact , on the contrary , it highlights them .']
    [7] ["this was n't a brothel , per se ."]
    [8] ['this is a family portrait .']
    [9] ["`` `` '' we have no freedom , '' '' they said . ''"]
    
    bleu_score:{'1-garm': 0.4836718950887118, '2-garm': 0.3388050226134023, '3-garm': 0.24616561206595167, '4-garm': 0.18057747434727217}
    747/747 [==============================] - 51s 68ms/step - loss: 2.3941 - accuracy: 0.5034 - val_loss: 2.5634 - val_accuracy: 0.5051
    
    Epoch 9/20
    
    candidates:
    [0] that ' s twice as much of the existence of men in this planet .
    [1] where do you find these conditions ?
    [2] but of course , life is more than i ' m expense .
    [3] every trotetegrate information .
    [4] i was n ' t a activist .
    [5] the fight for political rights has not just to look at marriage .
    [6] in fact , unlike the way .
    [7] this was not a bulk , exactly said .
    [8] this is a glass party .
    [9] we do n ' t have freedom , they said .
    
    references:
    [0] ["that 's twice as long as humans have been on this planet ."]
    [1] ['now , where do you find such goldilocks conditions ?']
    [2] ['but of course , life is more than just exotic chemistry .']
    [3] ['each rung contains information .']
    [4] ['i was not an activist .']
    [5] ['the fight for equal rights is not just about gay marriage .']
    [6] ['in fact , on the contrary , it highlights them .']
    [7] ["this was n't a brothel , per se ."]
    [8] ['this is a family portrait .']
    [9] ["`` `` '' we have no freedom , '' '' they said . ''"]
    
    bleu_score:{'1-garm': 0.5322236448997012, '2-garm': 0.37776572470306014, '3-garm': 0.27978052350824395, '4-garm': 0.21026300298124395}
    747/747 [==============================] - 44s 59ms/step - loss: 2.1855 - accuracy: 0.5350 - val_loss: 2.4502 - val_accuracy: 0.5278
    
    Epoch 10/20
    
    candidates:
    [0] that ' s twice as long as the asian existence of men in this planet .
    [1] where do you find these conditions ?
    [2] but , of course , life is more than i have expool chemistry .
    [3] every trollation .
    [4] i was n ' t a activist .
    [5] the fight for the rights of rights has n ' t just to see the marriage with the green gay .
    [6] in fact , unlike the real , real .
    [7] this was n ' t a borgan , exactly said .
    [8] this is a family .
    [9] we do n ' t have freedom , they said .
    
    references:
    [0] ["that 's twice as long as humans have been on this planet ."]
    [1] ['now , where do you find such goldilocks conditions ?']
    [2] ['but of course , life is more than just exotic chemistry .']
    [3] ['each rung contains information .']
    [4] ['i was not an activist .']
    [5] ['the fight for equal rights is not just about gay marriage .']
    [6] ['in fact , on the contrary , it highlights them .']
    [7] ["this was n't a brothel , per se ."]
    [8] ['this is a family portrait .']
    [9] ["`` `` '' we have no freedom , '' '' they said . ''"]
    
    bleu_score:{'1-garm': 0.5487502385041023, '2-garm': 0.39774899604881697, '3-garm': 0.2976384114444515, '4-garm': 0.22560007910643276}
    747/747 [==============================] - 45s 60ms/step - loss: 2.0244 - accuracy: 0.5580 - val_loss: 2.3699 - val_accuracy: 0.5401
    
    Epoch 11/20
    
    candidates:
    [0] that ' s twice as long as the existence of the men in this planet .
    [1] where are you find these gloom conditions ?
    [2] but of course , life is more than iranstinocence .
    [3] every trocomplete information .
    [4] i was n ' t a activist .
    [5] the fight for equality do n ' t have to just look at the homosexual .
    [6] in fact , unlike them .
    [7] this was n ' t a bork , exactly .
    [8] this is a family portrait of family .
    [9] we do n ' t have freedom , they said .
    
    references:
    [0] ["that 's twice as long as humans have been on this planet ."]
    [1] ['now , where do you find such goldilocks conditions ?']
    [2] ['but of course , life is more than just exotic chemistry .']
    [3] ['each rung contains information .']
    [4] ['i was not an activist .']
    [5] ['the fight for equal rights is not just about gay marriage .']
    [6] ['in fact , on the contrary , it highlights them .']
    [7] ["this was n't a brothel , per se ."]
    [8] ['this is a family portrait .']
    [9] ["`` `` '' we have no freedom , '' '' they said . ''"]
    
    bleu_score:{'1-garm': 0.5542072339024505, '2-garm': 0.40423149171157596, '3-garm': 0.306277345195794, '4-garm': 0.23482174098473035}
    747/747 [==============================] - 41s 54ms/step - loss: 1.8801 - accuracy: 0.5821 - val_loss: 2.3169 - val_accuracy: 0.5524
    
    Epoch 12/20
    
    candidates:
    [0] that ' s twice as much as the existence of the men of the men on this planet .
    [1] where do you find these gloglog conditions ?
    [2] but of course , life is more than exotic chemistry .
    [3] every trocomplan information .
    [4] i was n ' t a activist .
    [5] the struggle of rights of rights does n ' t have to only see the gay gay .
    [6] in fact , in the contrary , it ' s true .
    [7] this was n ' t a borrified , exactly said .
    [8] this is a family portrait of family .
    [9] we do n ' t have freedom , they said .
    
    references:
    [0] ["that 's twice as long as humans have been on this planet ."]
    [1] ['now , where do you find such goldilocks conditions ?']
    [2] ['but of course , life is more than just exotic chemistry .']
    [3] ['each rung contains information .']
    [4] ['i was not an activist .']
    [5] ['the fight for equal rights is not just about gay marriage .']
    [6] ['in fact , on the contrary , it highlights them .']
    [7] ["this was n't a brothel , per se ."]
    [8] ['this is a family portrait .']
    [9] ["`` `` '' we have no freedom , '' '' they said . ''"]
    
    bleu_score:{'1-garm': 0.5582268970698723, '2-garm': 0.4096318864263004, '3-garm': 0.3111445161988613, '4-garm': 0.23911921980669582}
    747/747 [==============================] - 43s 58ms/step - loss: 1.7553 - accuracy: 0.6033 - val_loss: 2.2583 - val_accuracy: 0.5626
    
    Epoch 13/20
    
    candidates:
    [0] that ' s twice as long as the existence of the men in this planet .
    [1] where are you find these gloomots ?
    [2] but of course , life is more than i ' m exotic .
    [3] every trocomplete information .
    [4] i was n ' t a activist .
    [5] the battle of rights is not just watching with the gay .
    [6] actually , unlike the word , i lied .
    [7] this was not a bork , exactly said .
    [8] this is a family portrait of family .
    [9] we do n ' t have freedom , they said .
    
    references:
    [0] ["that 's twice as long as humans have been on this planet ."]
    [1] ['now , where do you find such goldilocks conditions ?']
    [2] ['but of course , life is more than just exotic chemistry .']
    [3] ['each rung contains information .']
    [4] ['i was not an activist .']
    [5] ['the fight for equal rights is not just about gay marriage .']
    [6] ['in fact , on the contrary , it highlights them .']
    [7] ["this was n't a brothel , per se ."]
    [8] ['this is a family portrait .']
    [9] ["`` `` '' we have no freedom , '' '' they said . ''"]
    
    bleu_score:{'1-garm': 0.5507373271889401, '2-garm': 0.4029495577113799, '3-garm': 0.3059529900945778, '4-garm': 0.2353768172656205}
    747/747 [==============================] - 48s 65ms/step - loss: 1.6773 - accuracy: 0.6165 - val_loss: 2.2662 - val_accuracy: 0.5579
    
    Epoch 14/20
    
    candidates:
    [0] that ' s twice as long as the existence of the men in this planet .
    [1] where are you find these glom conditions ?
    [2] but of course , life is more than i ' m thrilled .
    [3] each trocoruction information .
    [4] i was not an activist .
    [5] the justice fighting justice does n ' t only have to see the marriage sex .
    [6] in fact , unlike the word , it ' s the actual thing .
    [7] this was not a bormo , exactly said .
    [8] this is a family portrait of family .
    [9] we do n ' t have freedom , they said .
    
    references:
    [0] ["that 's twice as long as humans have been on this planet ."]
    [1] ['now , where do you find such goldilocks conditions ?']
    [2] ['but of course , life is more than just exotic chemistry .']
    [3] ['each rung contains information .']
    [4] ['i was not an activist .']
    [5] ['the fight for equal rights is not just about gay marriage .']
    [6] ['in fact , on the contrary , it highlights them .']
    [7] ["this was n't a brothel , per se ."]
    [8] ['this is a family portrait .']
    [9] ["`` `` '' we have no freedom , '' '' they said . ''"]
    
    bleu_score:{'1-garm': 0.5753411630450288, '2-garm': 0.4277047616572427, '3-garm': 0.3300325613029533, '4-garm': 0.2575814912397511}
    747/747 [==============================] - 43s 57ms/step - loss: 1.6001 - accuracy: 0.6296 - val_loss: 2.2200 - val_accuracy: 0.5677
    
    Epoch 15/20
    
    candidates:
    [0] that ' s twice as time of the elderly in this planet .
    [1] where are you find these goll cells ?
    [2] but of course , the life is more than i would hope to exotic chemistry .
    [3] every troubling information .
    [4] i was n ' t an activist .
    [5] the fight for rights of rights does n ' t just have to see with the marriage sex .
    [6] in fact , unlike , it ' s true .
    [7] this was n ' t a borrified , exactly said .
    [8] this is a family portrait of family .
    [9] we do n ' t have freedom , they said .
    
    references:
    [0] ["that 's twice as long as humans have been on this planet ."]
    [1] ['now , where do you find such goldilocks conditions ?']
    [2] ['but of course , life is more than just exotic chemistry .']
    [3] ['each rung contains information .']
    [4] ['i was not an activist .']
    [5] ['the fight for equal rights is not just about gay marriage .']
    [6] ['in fact , on the contrary , it highlights them .']
    [7] ["this was n't a brothel , per se ."]
    [8] ['this is a family portrait .']
    [9] ["`` `` '' we have no freedom , '' '' they said . ''"]
    
    bleu_score:{'1-garm': 0.5708965777316867, '2-garm': 0.4237632062214664, '3-garm': 0.32515169381967, '4-garm': 0.25237882057093153}
    747/747 [==============================] - 45s 60ms/step - loss: 1.5241 - accuracy: 0.6422 - val_loss: 2.2268 - val_accuracy: 0.5731
    
    Epoch 16/20
    
    candidates:
    [0] that ' s twice as long as the existence of the men on this planet .
    [1] where do you find these glopolit conditions ?
    [2] but of course , life is more than i get exotic .
    [3] each troubling information .
    [4] i was not an activist .
    [5] the fight for rights struggle does n ' t only see the marriage sex .
    [6] in fact , unlike the contrary , i ' m moving .
    [7] this was not a borus , exactly said .
    [8] this is a family portrait of family .
    [9] we do n ' t have freedom , they said .
    
    references:
    [0] ["that 's twice as long as humans have been on this planet ."]
    [1] ['now , where do you find such goldilocks conditions ?']
    [2] ['but of course , life is more than just exotic chemistry .']
    [3] ['each rung contains information .']
    [4] ['i was not an activist .']
    [5] ['the fight for equal rights is not just about gay marriage .']
    [6] ['in fact , on the contrary , it highlights them .']
    [7] ["this was n't a brothel , per se ."]
    [8] ['this is a family portrait .']
    [9] ["`` `` '' we have no freedom , '' '' they said . ''"]
    
    bleu_score:{'1-garm': 0.582125257164729, '2-garm': 0.4333067375095398, '3-garm': 0.33377503665441394, '4-garm': 0.26039015293697665}
    747/747 [==============================] - 46s 62ms/step - loss: 1.5001 - accuracy: 0.6461 - val_loss: 2.2070 - val_accuracy: 0.5773
    
    Epoch 17/20
    
    candidates:
    [0] that ' s twice as the existence of the men ' s time in this planet .
    [1] where do you meet these cloudburs ?
    [2] but of course , life is more than iranstible .
    [3] every trocord is connected .
    [4] i was n ' t an activist .
    [5] the fight for rights does n ' t have to see with the marriage sex .
    [6] in fact , contrary , it ' s real .
    [7] this was n ' t a toaster , exactly , said .
    [8] this is a family portrait of family .
    [9] we do n ' t have freedom , they said .
    
    references:
    [0] ["that 's twice as long as humans have been on this planet ."]
    [1] ['now , where do you find such goldilocks conditions ?']
    [2] ['but of course , life is more than just exotic chemistry .']
    [3] ['each rung contains information .']
    [4] ['i was not an activist .']
    [5] ['the fight for equal rights is not just about gay marriage .']
    [6] ['in fact , on the contrary , it highlights them .']
    [7] ["this was n't a brothel , per se ."]
    [8] ['this is a family portrait .']
    [9] ["`` `` '' we have no freedom , '' '' they said . ''"]
    
    bleu_score:{'1-garm': 0.5816679477325134, '2-garm': 0.4343116947807011, '3-garm': 0.3348092333562867, '4-garm': 0.26096838242396714}
    747/747 [==============================] - 44s 59ms/step - loss: 1.4481 - accuracy: 0.6534 - val_loss: 2.1977 - val_accuracy: 0.5780
    
    Epoch 18/20
    
    candidates:
    [0] that ' s twice as long as the existence of the men on this planet .
    [1] where are you find these glodys ?
    [2] but of course , life is more than iranstitrial chemistry .
    [3] every trocompanied information .
    [4] i was n ' t an activist .
    [5] the fight for rights does n ' t just see the wedding with the homosex .
    [6] in fact , unlikely , we ' re focusing .
    [7] this was not a borus , exactly said .
    [8] this is a family portrait of family .
    [9] we do n ' t have freedom , they said .
    
    references:
    [0] ["that 's twice as long as humans have been on this planet ."]
    [1] ['now , where do you find such goldilocks conditions ?']
    [2] ['but of course , life is more than just exotic chemistry .']
    [3] ['each rung contains information .']
    [4] ['i was not an activist .']
    [5] ['the fight for equal rights is not just about gay marriage .']
    [6] ['in fact , on the contrary , it highlights them .']
    [7] ["this was n't a brothel , per se ."]
    [8] ['this is a family portrait .']
    [9] ["`` `` '' we have no freedom , '' '' they said . ''"]
    
    bleu_score:{'1-garm': 0.5793677363814626, '2-garm': 0.43253117325590495, '3-garm': 0.33414948953613227, '4-garm': 0.2611155803480264}
    747/747 [==============================] - 46s 62ms/step - loss: 1.3957 - accuracy: 0.6642 - val_loss: 2.1994 - val_accuracy: 0.5785
    
    Epoch 19/20
    
    candidates:
    [0] that ' s twice as the existence of the men on the planet .
    [1] where do you find these gollblot conditions ?
    [2] but of course , life is more than imparatic chemistry than i had .
    [3] every troubling information .
    [4] i was n ' t an activist .
    [5] the fight for rights equality does n ' t just look at the gay marriage of the gay .
    [6] in fact , contrary , it ' s true .
    [7] this was n ' t a borbsent , exactly said .
    [8] this is a family portrait of family .
    [9] ` ` ` ` ' ' we do n ' t have freedom , ' ' ' they said . ' '
    
    references:
    [0] ["that 's twice as long as humans have been on this planet ."]
    [1] ['now , where do you find such goldilocks conditions ?']
    [2] ['but of course , life is more than just exotic chemistry .']
    [3] ['each rung contains information .']
    [4] ['i was not an activist .']
    [5] ['the fight for equal rights is not just about gay marriage .']
    [6] ['in fact , on the contrary , it highlights them .']
    [7] ["this was n't a brothel , per se ."]
    [8] ['this is a family portrait .']
    [9] ["`` `` '' we have no freedom , '' '' they said . ''"]
    
    bleu_score:{'1-garm': 0.577036124016308, '2-garm': 0.42968404639389257, '3-garm': 0.3296844392656561, '4-garm': 0.255874328735193}
    747/747 [==============================] - 43s 57ms/step - loss: 1.3628 - accuracy: 0.6700 - val_loss: 2.1853 - val_accuracy: 0.5785
    
    Epoch 20/20
    
    candidates:
    [0] that ' s twice as long as the existence of the men on this planet .
    [1] where do you meet these gollutot ?
    [2] but of course , life is more than i gave maraph chemistry .
    [3] each trocomplishment of information .
    [4] i was n ' t an activist .
    [5] the battle for rights problem does n ' t have to only look at thewood sex .
    [6] in fact , on the contrary , it ' s true .
    [7] this was n ' t a bordy , which was said .
    [8] this is a family portrait of her .
    [9] ` ` ` ` ' ' we do n ' t have freedom , ' ' ' they said . ' '
    
    references:
    [0] ["that 's twice as long as humans have been on this planet ."]
    [1] ['now , where do you find such goldilocks conditions ?']
    [2] ['but of course , life is more than just exotic chemistry .']
    [3] ['each rung contains information .']
    [4] ['i was not an activist .']
    [5] ['the fight for equal rights is not just about gay marriage .']
    [6] ['in fact , on the contrary , it highlights them .']
    [7] ["this was n't a brothel , per se ."]
    [8] ['this is a family portrait .']
    [9] ["`` `` '' we have no freedom , '' '' they said . ''"]
    
    bleu_score:{'1-garm': 0.578121249939999, '2-garm': 0.4293865974117414, '3-garm': 0.3304440812067238, '4-garm': 0.2581370778046978}
    747/747 [==============================] - 42s 56ms/step - loss: 1.3287 - accuracy: 0.6750 - val_loss: 2.1797 - val_accuracy: 0.5850

        
    (6) 模型评价
    
    1.epoch=18 时的模型
    
    candidates:
    [0] that ' s twice as long as the existence of the men on this planet .
    [1] where are you find these glodys ?
    [2] but of course , life is more than iranstitrial chemistry .
    [3] every trocompanied information .
    [4] i was n ' t an activist .
    [5] the fight for rights does n ' t just see the wedding with the homosex .
    [6] in fact , unlikely , we ' re focusing .
    [7] this was not a borus , exactly said .
    [8] this is a family portrait of family .
    [9] we do n ' t have freedom , they said .
    
    references:
    [0] ["that 's twice as long as humans have been on this planet ."]
    [1] ['now , where do you find such goldilocks conditions ?']
    [2] ['but of course , life is more than just exotic chemistry .']
    [3] ['each rung contains information .']
    [4] ['i was not an activist .']
    [5] ['the fight for equal rights is not just about gay marriage .']
    [6] ['in fact , on the contrary , it highlights them .']
    [7] ["this was n't a brothel , per se ."]
    [8] ['this is a family portrait .']
    [9] ["`` `` '' we have no freedom , '' '' they said . ''"]
    
    bleu_score:{'1-garm': 0.5793677363814626, '2-garm': 0.43253117325590495, '3-garm': 0.33414948953613227, '4-garm': 0.2611155803480264}
        
        
    