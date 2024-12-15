import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import os
import pandas as pd

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(key_dim)
        ])
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs):
        # self-attention with residual connection
        x_norm1 = self.layernorm1(inputs)
        attn_output = self.mha(x_norm1, x_norm1)
        attn_output = self.dropout1(attn_output)
        out1 = inputs + attn_output

        # feed-forward with residual connection
        x_norm2 = self.layernorm2(out1)
        ffn_output = self.ffn(x_norm2)
        ffn_output = self.dropout2(ffn_output)
        out2 = out1 + ffn_output

        return out2

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=sequence_length, output_dim=d_model
        )

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        position_embeddings = self.position_embeddings(positions)
        sum = inputs + position_embeddings
        return inputs + position_embeddings

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "sequence_length": self.position_embeddings.input_dim,
            "d_model": self.position_embeddings.output_dim,
        })
        return config

def build_model(input_shape, num_heads, key_dim, ff_dim, num_transformer_blocks, dropout_rate=0.1):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Masking(mask_value=0)(inputs)
    x = tf.keras.layers.Dense(key_dim)(x)
    x = PositionalEmbedding(input_shape[0], key_dim)(x)
    for _ in range(num_transformer_blocks):
        x = TransformerBlock(num_heads=num_heads, key_dim=key_dim, ff_dim=ff_dim, dropout_rate=dropout_rate)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    return model


all_x = np.load('x_61046.npy', allow_pickle=True)
all_y = np.load('y_61046.npy', allow_pickle=True)
x_all = [np.delete(arr, -1, axis=1) for arr in all_x] # removing length feature
input_vectors = np.array(x_all)
label_vectors = np.squeeze(np.array(all_y))
train_input_vectors, val_input_vectors, train_label_vectors, val_label_vectors = train_test_split(input_vectors, label_vectors, test_size=0.02, random_state=42)

x_encoded_path = 'x_encoded.npy'
x_val_encoded_path = 'x_val_encoded.npy'
encoded_outputs = np.load(x_encoded_path)
val_encoded_outputs = np.load(x_val_encoded_path)

for i in range(encoded_outputs.shape[0]):
  encoded_outputs[i] = (encoded_outputs[i] - np.mean(encoded_outputs[i])) / np.std(encoded_outputs[i])

for i in range(val_encoded_outputs.shape[0]):
  val_encoded_outputs[i] = (val_encoded_outputs[i] - np.mean(val_encoded_outputs[i])) / np.std(val_encoded_outputs[i])

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=4000,
    decay_rate=0.96,
    staircase=True
)

input_shape = (500, 28)
num_heads = 8
key_dim = 128
ff_dim = input_shape[0]
num_transformer_blocks = 8
dropout_rate = 0.1

model = build_model(input_shape, num_heads, key_dim, ff_dim, num_transformer_blocks, dropout_rate)
optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.summary()

epoch_n = 450
batch_size = 128
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
        encoded_outputs, train_label_vectors,
        validation_data=(val_encoded_outputs, val_label_vectors),
        epochs=epoch_n,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )
