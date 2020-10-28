import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from experimental.rnn.generic_utils import read_pairs
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau

# data = pd.read_csv('../../data/fake_news/train.csv')
# data = data.fillna(' ')
# data.count()

train_path = '../../data/fakenewskdd2020/train.csv'
train_raw_data = dict(read_pairs(train_path, cast=(str, int), offset=1))
train_data = {
    'text': pd.Index(list(train_raw_data.keys())),
    'label': pd.Index(list(train_raw_data.values()))
}

test_path = '../../data/fakenewskdd2020/test.csv'
test_raw_data = dict(read_pairs(test_path, cast=(int, str), offset=1))
test_data = {
    'id': pd.Index(list(test_raw_data.keys())),
    'text': pd.Index(list(test_raw_data.values()))
}

df = pd.DataFrame(test_data)

# test_data = pd.read_csv('../../data/fake_news/test.csv')
# test_data = test_data.fillna(' ')
# test_data.count()

# print(data['label'].__len__())
# print(sum(data['label']))

# Tokenize text

tokenizer = Tokenizer()
# tokenizer.fit_on_texts(data['text'])
tokenizer.fit_on_texts(train_data['text'])
tokenizer.fit_on_texts(test_data['text'])
word_index = tokenizer.word_index
vocab_size = len(word_index)
print(vocab_size)

# Padding data, making to vectors of numbers

# sequences = tokenizer.texts_to_sequences(data['text'])
train_sequences = tokenizer.texts_to_sequences(train_data['text'])
train_padded = pad_sequences(train_sequences, maxlen=2000, padding='post', truncating='post')

test_sequences = tokenizer.texts_to_sequences(test_data['text'])
test_padded = pad_sequences(test_sequences, maxlen=2000, padding='post', truncating='post')

# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(test_data['text'])
# store_index = tokenizer.word_index
# vocabulary_size = len(store_index)
# # print(vocabulary_size)
#
# sequences = tokenizer.texts_to_sequences(test_data['text'])
# test_padded = pad_sequences(sequences, maxlen=500, padding='post', truncating='post')

split = 0.2
split_n = int(round(len(train_padded)*(1-split),0))

train_train_data = train_padded[:split_n]
train_train_labels = train_data['label'].values[:split_n]
train_test_data = train_padded[split_n:]
train_test_labels = train_data['label'].values[split_n:]

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
    tf.keras.layers.Embedding(vocab_size+1, 100, weights=[embeddings_matrix], trainable=True),
    tf.keras.layers.Dropout(0.1),
    # tf.keras.layers.Conv1D(128, 7, activation='swish'),
    # tf.keras.layers.MaxPooling1D(pool_size=4),
    # tf.keras.layers.Conv1D(64, 3, activation='swish'),
    # tf.keras.layers.MaxPooling1D(pool_size=2),
    # tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16, return_sequences=True, recurrent_dropout=0.1)),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(256, activation='swish'),
    tf.keras.layers.Dropout(0.4),
    # tf.keras.layers.Dense(256),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

optimizer = Adam(lr=2e-3)
# scheduler = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
#     min_delta=0.0001, cooldown=0, min_lr=0, **kwargs
# )
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()
history = model.fit(train_train_data, train_train_labels, validation_split=0.1, epochs=5, batch_size=16, validation_data=[train_test_data, train_test_labels])
# for i in range(len(results)):
#     num = results[i]
#     results[i] = round(num)
# print(results)
df['label'] = model.predict(test_padded)
for i in range(len(df['label'])):
    if df['label'][i] > 0.5:
        df['label'][i] = 1
    else:
        df['label'][i] = 0

df['label'] = df['label'].astype("int")
df = df.drop(["text"], axis=1)
print(df)
df.to_csv('submission.csv', index=False)
# # print(results)
# my_submission = pd.DataFrame({'id': pd.Index(list(raw_data.keys())), 'label': results})
# print(my_submission)
print("Training Complete")
# score = 76%


