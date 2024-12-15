from math import nan
import torch
import numpy as np
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

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
        x_norm1 = self.layernorm1(inputs)
        attn_output = self.mha(x_norm1, x_norm1)
        attn_output = self.dropout1(attn_output)
        out1 = inputs + attn_output

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
        return inputs + position_embeddings


def build_autoencoder(input_shape, num_heads, key_dim, ff_dim, num_transformer_blocks, dropout_rate=0.1):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Masking(mask_value=0)(inputs)
    x = tf.keras.layers.Dense(key_dim)(x)
    x = PositionalEmbedding(input_shape[0], key_dim)(x)

    for _ in range(num_transformer_blocks):
        x = TransformerBlock(num_heads=num_heads, key_dim=key_dim, ff_dim=ff_dim, dropout_rate=dropout_rate)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    encoded = tf.keras.layers.Dense(28, activation='relu', name='encoded')(x)

    # Decoder
    x = tf.keras.layers.Dense(64, activation='relu')(encoded)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    decoded = tf.keras.layers.Dense(input_shape[-1], activation='linear')(x)

    autoencoder = tf.keras.Model(inputs, decoded)
    encoder = tf.keras.Model(inputs, encoded)

    return autoencoder, encoder

input_shape = (500, 28)
num_heads = 2
key_dim = 28
ff_dim = input_shape[0]
num_transformer_blocks = 2
dropout_rate = 0.1

autoencoder, encoder = build_autoencoder(input_shape, num_heads, key_dim, ff_dim, num_transformer_blocks, dropout_rate)
optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
autoencoder.summary()

all_x = np.load('x_61046.npy', allow_pickle=True)
all_y = np.load('y_61046.npy', allow_pickle=True)
x_all = [np.delete(arr, -1, axis=1) for arr in all_x] # removing length feature

input_vectors = np.array(x_all)
label_vectors = np.squeeze(np.array(all_y))

train_input_vectors, val_input_vectors, train_label_vectors, val_label_vectors = train_test_split(input_vectors, label_vectors, test_size=0.02, random_state=42)


# Set seed
np.random.seed(5)
torch.manual_seed(5)
if torch.cuda.is_available():
    torch.cuda.manual_seed(5)

autoencoder.fit(
    train_input_vectors,
    train_input_vectors,
    validation_data=(val_input_vectors, val_input_vectors),
    epochs=10,
    batch_size=128,
    tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 25, restore_best_weights = True)]
)


encoded_outputs = encoder.predict(train_input_vectors)
val_encoded_outputs = encoder.predict(val_input_vectors)

x_encoded_path = 'x_train_encoded.npy'
x_val_encoded_path = 'x_val_encoded.npy'

np.save(x_encoded_path, encoded_outputs)
np.save(x_val_encoded_path, val_encoded_outputs)
