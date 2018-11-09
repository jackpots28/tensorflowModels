import tensorflow as tf
tf.enable_eager_execution()

import unidecode

import numpy as np
import re
import random
import time

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/yashkatariya/shakespeare.txt')

text = unidecode.unidecode(open(path_to_file).read())
# print(len(text))

unique = sorted(set(text))

char2idx = {u: i for i, u in enumerate(unique)}
idx2char = {i: u for i, u in enumerate(unique)}

max_length = 100
vocab_size = len(unique)
embedding_dim = 256
units = 1024
BATCH_SIZE = 64
BUFFER_SIZE = 10000

input_text = []
target_text = []

for f in range(0, len(text) - max_length, max_length):
    inps = text[f: f+max_length]
    targ = text[f+1: f+max_length]

    input_text.append([char2idx[i] for i in inps])
    target_text.append([char2idx[t] for t in targ])


print(np.array(input_text).shape)
print(np.array(target_text).shape)

dataset = tf.data.Dataset.from_tensor_slices((input_text, target_text)).shuffle(BUFFER_SIZE)
dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))

class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, batch_size):
        super(Model, self).__init__()
        self.units = units
        self.batch_sz = batch_size

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        if tf.test.is_gpu_available():
            self.gru = tf.keras.layers.CuDNNGRU(self.units,
                                                return_sequences=True,
                                                return_state=True,
                                                recurrent_initializer='glorot_uniform')

        else:
            self.gru = tf.keras.layers.GRU(self.units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_activation='sigmoid',
                                           recurrent_initializer='glorot_uniform')

        self.fc = tf.keras.layers.Dense(vocab_size)

        def call(self, x, hidden):
            x = self.embedding(x)
            output, states = self.gru(x, initial_state=hidden)
            output = tf.reshape(output, (-1, output.shape[2]))
            x = self.fc(output)

            return x, states


model = Model(vocab_size, embedding_dim, units, BATCH_SIZE)
optimizer = tf.train.AdamOptimizer()


def loss_function(real, preds):
    return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)


EPOCHS = 30

for epoch in range(EPOCHS):
    start = time.time()
    hidden = model.reset_states()

    for (batch, (inp, target)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            predictions, hidden = model(inp, hidden)

            target = tf.reshape(target, (-1,))
            loss = loss_function(target, predictions)

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1,
                                                         batch,
                                                         loss))
    print('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))