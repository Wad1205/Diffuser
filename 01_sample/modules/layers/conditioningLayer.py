# 20250820 release

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

# -------------------------------
# Timestep Embedding
# -------------------------------
class TimestepEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def call(self, t):
        # sinusoidal positional embedding
        half_dim = self.dim // 2
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = tf.cast(t[:, None], tf.float32) * emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        
        if self.dim % 2 == 1:
            emb = tf.pad(emb, [[0, 0], [0, 1]])

        return emb
    
# -------------------------------
# Condition Encoder
# -------------------------------
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, model_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.pos_encoding = self.positional_encoding(position, model_dim)

    def get_angles(self, pos, i, model_dim):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(model_dim))
        return pos * angle_rates

    def positional_encoding(self, position, model_dim):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], 
                                     np.arange(model_dim)[np.newaxis, :], 
                                     model_dim)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]
    
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, 
                                                      key_dim=model_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"), 
            tf.keras.layers.Dense(model_dim)
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training=False):
        attn = self.att(x, x)
        attn = self.dropout1(attn, training=training)
        out1 = self.norm1(x + attn)
        ffn = self.ffn(out1)
        ffn = self.dropout2(ffn, training=training)
        return self.norm2(out1 + ffn)
