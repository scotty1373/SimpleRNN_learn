# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import datasets
from tensorflow.keras import preprocessing
import numpy as np


IMDB = datasets.imdb
(x_train, y_train), (x_test, y_test) = IMDB.load_data()
max_word_token = 1000
maxlen = 100
word_embedding_size = 20

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)


def simple_rnn():
    input_seq = keras.layers.Input(shape=(maxlen,))
    common = keras.layers.Embedding(input_dim=max_word_token,
                                    output_dim=word_embedding_size)(input_seq)
    common = keras.layers.SimpleRNN(word_embedding_size)(common)
    # 对t=10的时间戳序列经过单个cell循环十次
    common = keras.layers.Flatten()(common)
    common_altitude = keras.layers.Dense(units=1, activation='softmax')(common)
    model_struct = keras.Model(inputs=input_seq, outputs=common_altitude)

    return model_struct


model = simple_rnn()
model.summary()
model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

samples = ['The cat sat on the mat.', 'The Tesla is the greatest company.']

token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1

max_length = 10

result = np.zeros(shape=(len(samples), max_length, max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        result[i, j, index] = 1

print(token_index)
print(result)

