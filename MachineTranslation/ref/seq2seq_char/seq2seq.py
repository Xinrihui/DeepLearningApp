from __future__ import print_function

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

from tensorflow.keras.utils import plot_model


import numpy as np

batch_size = 32  # 训练批次大小。
epochs = 100   # 训练迭代轮次。
latent_dim = 256  # 编码空间隐层维度。
num_samples = 10000  # 训练样本数。
# 磁盘数据文件路径。
data_path = '../dataset/anki/fra-eng/fra.txt'

# 向量化数据。
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text, _ = line.split('\t')
    # 我们使用 "tab" 作为 "起始序列" 字符，
    # 对于目标，使用 "\n" 作为 "终止序列" 字符。
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data 领先 decoder_input_data by 一个时间步。
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data 将提前一个时间步，并且将不包含开始字符。
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
    decoder_target_data[i, t:, target_token_index[' ']] = 1.
# 定义输入序列并处理它。
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# 我们抛弃 `encoder_outputs`，只保留状态。
encoder_states = [state_h, state_c]

# 使用 `encoder_states` 作为初始状态来设置解码器。
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# 我们将解码器设置为返回完整的输出序列，并返回内部状态。
# 我们不在训练模型中使用返回状态，但将在推理中使用它们。
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# 定义模型，将 `encoder_input_data` & `decoder_input_data` 转换为 `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 打印 模型(计算图) 的所有网络层
print(model.summary())

# 输出训练计算图的图片
plot_model(model, to_file='seq2seq_model.png', show_layer_names=True, show_shapes=True)

# 执行训练
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# 保存模型
model.save('s2s.h5')

# 接下来: 推理模式 (采样)。
# 这是演习：
# 1) 编码输入并检索初始解码器状态
# 2) 以该初始状态和 "序列开始" token 为目标运行解码器的一步。 输出将是下一个目标 token。
# 3) 重复当前目标 token 和当前状态

# 定义采样模型
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# 反向查询 token 索引可将序列解码回可读的内容。
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # 将输入编码为状态向量。
    states_value = encoder_model.predict(input_seq)

    # 生成长度为 1 的空目标序列。
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # 用起始字符填充目标序列的第一个字符。
    target_seq[0, 0, target_token_index['\t']] = 1.

    # 一批序列的采样循环
    # (为了简化，这里我们假设一批大小为 1)。
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # 采样一个 token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # 退出条件：达到最大长度或找到停止符。
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # 更新目标序列（长度为 1）。
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # 更新状态
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # 抽取一个序列（训练集的一部分）进行解码。
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)