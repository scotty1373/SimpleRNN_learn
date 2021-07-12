# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import datasets
from tensorflow.keras import preprocessing
import numpy as np

batch_size = 64
IMDB = datasets.imdb
max_word_token = 10000
maxlen = 100
word_embedding_size = 80
Training = True

# num_words从dataset中按照出现频率选取单词，这块如果不定义，则embedding层会报单词数量与向量数不匹配(常见单词)
# 相当于只对常见单词前10000个进行编码，而对其他生僻字都用一个编码来表示
(x_train, y_train), (x_test, y_test) = IMDB.load_data(num_words=max_word_token)

# 对句子长度控制
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# dataset establish
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(500).batch(batch_size, drop_remainder=True)     # 将最后一个不足batchsize的样本丢弃
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.shuffle(500).batch(batch_size, drop_remainder=True)

print(f'x_train shape: {x_train.shape}, {tf.reduce_max(x_train)}, {tf.reduce_min(x_train)}')
print(f'x_test shape: {x_test.shape}')


# 自定义网络层，从keras.Model继承
class MyRNN(keras.Model):
    def __init__(self, unit):
        super(MyRNN, self).__init__()
        # 初始memory状态
        self.state0 = [tf.zeros([batch_size, unit])]
        self.state1 = [tf.zeros([batch_size, unit])]
        self.state2 = [tf.zeros([batch_size, unit])]
        # [None, 100] -> [None, 100, 80] 将每次输入的100个单词转成[100, 80]的向量，一个单词用80长度的向量表示
        # transfrom text to embedding representation
        self.embedding = keras.layers.Embedding(max_word_token,
                                                word_embedding_size,
                                                input_length=maxlen)
        # [None, 100, 80] -> [None, unit] -> [None] 维度变换如下
        self.rnncell = keras.layers.SimpleRNNCell(units=unit, dropout=0.2)
        self.rnncell1 = keras.layers.SimpleRNNCell(units=unit, dropout=0.2)
        self.rnncell2 = keras.layers.SimpleRNNCell(units=unit, dropout=0.2)
        self.fc = keras.layers.Dense(1)

    # 通过training配置来决定是train还是test，对于RNNCell中的dropout
    # 如果是train则在训练时网络部断开连接，当test时为了不让dropout影响结果，则删除dropout在网络中的链接
    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.embedding(x)
        state0 = self.state0
        state1 = self.state1
        state2 = self.state2
        # x输入的word长度为时间轴rnn返回传播次数
        for word_ in tf.unstack(x, axis=1):      # [None, 0, 80] -> [None, 100, 80] iter
            # keras.layers.SimpleRNNCell train和test逻辑不同，给出training做针对性处理
            out0, state0 = self.rnncell(word_, state0, training)
            out1, state1 = self.rnncell1(out0, state1, training)      # 接上层rnn输出作为本层输入
            out2, state2 = self.rnncell1(out1, state2, training)
        x = self.fc(out2)
        prob = tf.sigmoid(x)
        return prob


if __name__ == '__main__':
    unit = 64
    epochs = 4
    model = MyRNN(unit)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['acc'])
    model.fit(db_train, epochs=epochs, validation_data=db_test)
    model.summary()



# def simple_rnn():
#     input_seq = keras.layers.Input(maxlen,)
#     common = keras.layers.Embedding(input_dim=max_word_token,
#                                     output_dim=word_embedding_size,
#                                     input_length=maxlen)(input_seq)
#     common = keras.layers.SimpleRNN(maxlen)(common)
#     # 对t=10的时间戳序列经过单个cell循环十次
#     common = keras.layers.Flatten()(common)
#     common_altitude = keras.layers.Dense(units=2, activation='sigmoid')(common)
#     model_struct = keras.Model(inputs=input_seq, outputs=common_altitude)
#
#     return model_struct


# model = simple_rnn()
# model.summary()
# model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=['acc'])
# y_train = tf.one_hot(y_train, depth=2)
# model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
#
# test_result = model.predict(x_test)
# test_result = tf.argmax(test_result, axis=1)
# currect_rate = tf.equal(test_result, y_test)
# accurancy_rate = tf.reduce_mean(tf.cast(currect_rate, tf.float32))      # bool to num
# print(f'test accuracy： {accurancy_rate}')
#
# samples = ['The cat sat on the mat.', 'The Tesla is the greatest company.']
#
#
# token_index = {}
# for sample in samples:
#     for word in sample.split():
#         if word not in token_index:
#             token_index[word] = len(token_index) + 1
#
# max_length = 10
#
# result = np.zeros(shape=(len(samples), max_length, max(token_index.values()) + 1))
#
# for i, sample in enumerate(samples):
#     for j, word in list(enumerate(sample.split()))[:max_length]:
#         index = token_index.get(word)
#         result[i, j, index] = 1
#
# print(token_index)
# print(result)

