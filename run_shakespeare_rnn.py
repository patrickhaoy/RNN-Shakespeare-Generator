from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os
import time

# Read the data
txt = open('shakespeare.txt', 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(txt))  # unique characters in the file

# Vectorize the text
char2idx = {u: c for c, u in enumerate(vocab)}  # character to index mapping of vocab
idx2char = np.array(vocab)
txt_as_int = np.array([char2idx[c] for c in txt])  # characters in text converted to integers

# Create inputs and targets for training
"""
Takes in a sequence and returns two variables: 
one without the last character (input), one without the first character (target)
"""
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


seq_length = 100  # each training input and corresponding target will be seq_length characters
examples_per_epoch = len(txt) // (seq_length + 1)  # number of (training input, target) pairs
char_dataset = tf.data.Dataset.from_tensor_slices(txt_as_int)  # Converts vector to stream of char indices
seq = char_dataset.batch(seq_length + 1, drop_remainder=True)  # Partition char to sequences of (seq_length+1) size
dataset = seq.map(split_input_target)  # Dataset contains sequences containing with input and target text

# Creating training batches
batch_size = 64
buffer_size = 10000  # maintains buffer of 10000 in which it shuffles elements
dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)  # shuffle data and pack into batches

# Building the model
"""
Create a sequential, or linear stack of layers. This sequential will be three layers:
1. Input layer (Embedding): numbers of each character mapped to vector with embedding_dim dimensions
2. GRU: Type of RNN with size rnn_units
3. Output layer (Dense): vocab_size units
"""
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


vocab_size = len(vocab)  # length of txt vocab
embedding_dim = 256  # embedding dimension
rnn_units = 1024  # number of rnn units
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)

# Training the model
"""
Cross Entropy loss function
"""
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer='adam', loss=loss)  # configure training procedure with Adam algorithm

# Configure checkpoints
checkpoint_dir = './training_checkpoints'  # directory checkpoints are saved to during training
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")  # Name of checkpt files
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(  # Saves model after each epoch
    filepath=checkpoint_prefix, save_weights_only=True
)

# Execute training
epochs = 10
history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])

# Restore latest checkpoint
tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)  # Batch size 1 to keep prediction simple
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()

# Prediction loop
"""
Generate text using the learned model
"""
def generate_text(model, start_string):
    num_generate = 1000  # number of characters to generate
    input_eval = tf.expand_dims([char2idx[s] for s in start_string], 0)  # Convert start string to numbers
    text_generated = []
    temperature = 1.0  # low temp -> more predictable, high temp -> more surprising; up for experimentation

    model.reset_states()  # batch_size == 1
    for i in range(num_generate):
        predictions = tf.squeeze(model(input_eval), 0) / temperature
        # Use categorical dist. to predict word returned
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)  # We pass predicted word as input into model
        text_generated.append(idx2char[predicted_id])

    return start_starting + ''.join(text_generated)


# Test model with input string of "ROMEO: "
print(generate_text(model, start_string=u"ROMEO: "))
