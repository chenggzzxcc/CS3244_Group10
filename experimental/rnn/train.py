import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

data = pd.read_csv('../../data/fake_news/train.csv')
data = data.fillna(' ')
data.count()

# test_data = pd.read_csv('../../data/fake_news/test.csv')
# test_data = test_data.fillna(' ')
# test_data.count()

# print(data['text'].head())

# Tokenize text

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['text'])
word_index = tokenizer.word_index
vocab_size = len(word_index)
# print(vocab_size)

# Padding data, making to vectors of numbers

sequences = tokenizer.texts_to_sequences(data['text'])
padded = pad_sequences(sequences, maxlen=500, padding='post', truncating='post')


# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(test_data['text'])
# store_index = tokenizer.word_index
# vocabulary_size = len(store_index)
# # print(vocabulary_size)
#
# sequences = tokenizer.texts_to_sequences(test_data['text'])
# test_padded = pad_sequences(sequences, maxlen=500, padding='post', truncating='post')

split = 0.2
split_n = int(round(len(padded)*(1-split),0))

train_data = padded[:split_n]
train_labels = data['label'].values[:split_n]
test_data = padded[split_n:]
test_labels = data['label'].values[split_n:]

embeddings_index = {}
with open('glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
print(len(coefs))

embeddings_matrix = np.zeros((vocab_size+1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, 100, weights=[embeddings_matrix], trainable=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(20, return_sequences=True),
    tf.keras.layers.LSTM(20),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(train_data, train_labels, epochs=5, batch_size=100, validation_data=[test_data, test_labels])

print("Training Complete")


