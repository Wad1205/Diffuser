# 20250820 release

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

# -------------------------------
# AdaGN Conv Block
# -------------------------------
class AdaGNConvBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, groups=8, activation="swish", **kwargs):
        super().__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv1D(out_channels, 3, padding="same")
        self.norm1 = tfa.layers.GroupNormalization(groups=groups, axis=-1)

        self.conv2 = tf.keras.layers.Conv1D(out_channels, 3, padding="same")
        self.norm2 = tfa.layers.GroupNormalization(groups=groups, axis=-1)
        self.activation = tf.keras.activations.get(activation)
        self.out_channels = out_channels
        self.groups = groups

    def build(self, input_shape):
        cond_dim = input_shape[1][-1]
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(self.out_channels*2, activation="swish"),
            tf.keras.layers.Dense(self.out_channels*2)
        ])

    def call(self, inputs):
        x, cond = inputs  # x: (B, T, C), cond: (B, cond_dim)

        gamma_beta = self.mlp(cond)  # (B, 2*out_dim)
        gamma, beta = tf.split(gamma_beta, 2, axis=-1)  # (B, out_dim) each
        gamma = tf.expand_dims(gamma, 1)  # (B,1,C)
        beta = tf.expand_dims(beta, 1)

        # conv1 + GN + AdaModulation
        h = self.conv1(x)
        h = self.norm1(h)
        h = h * (1 + gamma) + beta
        h = self.activation(h)

        # conv2 + GN + AdaModulation
        h = self.conv2(h)
        h = self.norm2(h)
        h = h * (1 + gamma) + beta
        h = self.activation(h)

        return h

# -------------------------------
# Residual AdaGN Block
# -------------------------------
class ResidualAdaGNBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, groups=8, activation="swish", **kwargs):
        super().__init__(**kwargs)
        self.block = AdaGNConvBlock(out_channels, groups, activation)
        self.res_conv = tf.keras.layers.Conv1D(out_channels, 1) if in_channels != out_channels else tf.identity

    def call(self, inputs):
        x, cond = inputs
        h = self.block([x, cond])
        return self.res_conv(x) + h

# -------------------------------
# Down / Up sampling
# -------------------------------
class Downsample1D(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        # stride=2でダウンサンプリング
        self.conv = tf.keras.layers.Conv1D(dim, kernel_size=3, strides=2, padding='same')

    def call(self, x):
        return self.conv(x)

class Upsample1D(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        # stride=2の逆畳み込みでアップサンプリング
        self.conv = tf.keras.layers.Conv1DTranspose(dim, kernel_size=4, strides=2, padding='same')

    def call(self, x):
        return self.conv(x)
