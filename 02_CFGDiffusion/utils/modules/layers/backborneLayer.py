# 20250820 release

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

# -------------------------------
# Sampling Block (AdaGN Conv Block)
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
# Sampling Block (Residual AdaGN Block)
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
# TemporalUnet: Down / Up sampling
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

# -------------------------------
# TemporalUnet: Activation
# -------------------------------
class Mish(tf.keras.layers.Activation):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'mish'

def act_mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

tf.keras.utils.get_custom_objects().update({'mish': Mish(act_mish)})

# -------------------------------
# TemporalUnet: Sampling Block (Conv1dBlock)
# -------------------------------
class Conv1dBlock(tf.keras.layers.Layer):
    '''
        Conv1d --> GroupNorm --> Mish/SiLU
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, mish=True, n_groups=8, **kwargs):
        super().__init__(**kwargs)
        
        #act_fn = tf.keras.activations.mish if mish else tf.keras.activations.swish
        act_fn = Mish(act_mish) if mish else tf.keras.activations.swish

        self.block = tf.keras.Sequential([
            #tf.keras.layers.Conv1D(out_channels, kernel_size, padding=['valid', 'same'][kernel_size // 2]),
            tf.keras.layers.Conv1D(out_channels, kernel_size, padding='same'),
            # tfa.layers.GroupNormalizationは (..., channels) の形式を扱える
            tfa.layers.GroupNormalization(groups=n_groups, axis=-1),
            tf.keras.layers.Activation(act_fn),
        ])

    def call(self, x):
        return self.block(x)

# -------------------------------
# TemporalUnet: Sampling Block (ResidualTemporalBlock)
# -------------------------------
class ResidualTemporalBlock(tf.keras.layers.Layer):
    def __init__(self, inp_channels, out_channels, embed_dim, horizon, 
                 kernel_size=5, mish=True, **kwargs):
        super().__init__(**kwargs)
        
        #act_fn = tf.keras.activations.mish if mish else tf.keras.activations.swish
        act_fn = Mish(act_mish) if mish else tf.keras.activations.swish

        self.blocks = [
            Conv1dBlock(inp_channels, out_channels, kernel_size, mish=mish),
            Conv1dBlock(out_channels, out_channels, kernel_size, mish=mish),
        ]

        self.time_mlp = tf.keras.Sequential([
            tf.keras.layers.Activation(act_fn),
            tf.keras.layers.Dense(out_channels),
        ])

        self.residual_conv = tf.keras.layers.Conv1D(out_channels, 1) \
                                if inp_channels != out_channels else tf.identity

    def call(self, x, t):
        '''
            x : [ batch_size, horizon, inp_channels ]
            t : [ batch_size, embed_dim ]
            returns:
            out : [ batch_size, horizon, out_channels ]
        '''
        # 時間埋め込みを適用
        time_embed = self.time_mlp(t)
        # (batch, out_channels) -> (batch, 1, out_channels) にしてブロードキャスト可能に
        time_embed_expanded = time_embed[:, tf.newaxis, :]
        
        out = self.blocks[0](x) + time_embed_expanded
        out = self.blocks[1](out)
        
        return out + self.residual_conv(x)
