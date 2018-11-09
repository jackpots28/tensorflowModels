import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

import numpy as np

# print(tf.__version__)

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
# print(train_data[0])

word_Index = imdb.get_word_index()

word_Index = {k:(v+3) for k,v in word_Index.items()}
word_Index["<PAD>"] = 0
word_Index["<START>"] = 1
word_Index["<UNK>"] = 2  # unknown
word_Index["<UNUSED>"] = 3

reverse_word_Index = dict([(value, key) for (key, value) in word_Index.items()])


def decode_review(text):
    return ' '.join([reverse_word_Index.get(i, '?') for i in text])


# this was to show the first review -> print(decode_review(train_data[0]))


train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_Index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_Index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# this was to show the new padded length of each data "0"
# and "1" in data sets -> print(len(train_data[0]), len(train_data[1]))

# this was to show the new padded review -> print(train_data[0])

vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=19,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data, test_labels)
print(results)

history_dict = history.history
history_dict.keys()
# dict_keys(['loss', 'val_loss', 'acc', 'val_acc'])

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()





