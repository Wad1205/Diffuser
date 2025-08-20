# 20250820

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
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=model_dim)
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

# -------------------------------
# Conditioning Module
# -------------------------------
class ConditioningModule(tf.keras.layers.Layer):
    def __init__(self, cond_len, cond_dim=128, num_heads=4, ff_dim=256, num_cond_blocks=2, 
                 **kwargs):
        super().__init__(**kwargs)
        self.cond_dim = cond_dim
        self.num_cond_blocks = num_cond_blocks

        # タイムステップ埋め込み
        self.t_embed = tf.keras.Sequential([
            TimestepEmbedding(cond_dim),
            tf.keras.layers.Dense(cond_dim, activation="swish"),
            tf.keras.layers.Dense(cond_dim),
        ])

        # Transformer Encoder block (シンプル版)
        self.c_embed = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(cond_dim),
                PositionalEncoding(cond_len, cond_dim),
            ] + [
                TransformerEncoder(cond_dim, num_heads, ff_dim) for _ in range(num_cond_blocks)
            ]
        )

        # クロスアテンションの追加
        self.cross_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=cond_dim)
        self.cross_attention_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # 出力プロジェクション
        self.norm = tf.keras.layers.GlobalAveragePooling1D()
        self.out_proj = tf.keras.layers.Dense(cond_dim)

    def call(self, timestep, cond_seq, training=False):
        """
        timestep: (B,) int32
        cond_seq: (B, T, C_cond)
        """
        # sinusoidal embedding for t
        t_emb = self.t_embed(timestep)  # (B, cond_dim)

        # transformer encoding of cond_seq
        cond_x = self.c_embed(cond_seq) # (B, cond_dim)
        
        # cond_xに注入する
        # t_embを(B, 1, cond_dim)に拡張してアテンションに適合させる
        t_emb_expanded = tf.expand_dims(t_emb, axis=1) # (B, 1, cond_dim)
        
        # Q: cond_x, K: t_emb, V: t_emb
        attn_output = self.cross_attention(query=cond_x, key=t_emb_expanded, value=t_emb_expanded)
        cond_x_with_t = self.cross_attention_norm(cond_x + attn_output) #残差接続と正規化        
        cond_vec = self.norm(cond_x_with_t) # (B, cond_dim)
        
        cond_vec = self.out_proj(cond_vec) # (B, cond_dim)

        return cond_vec

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

# -------------------------------
# UNet Backbone
# -------------------------------
class UNetBackbone(tf.keras.Model):
    def __init__(
            self, 
            transition_dim,
            cond_dim,
            dim=128,
            dim_mults=(1, 2, 4, 8),
            base_channels=64, 
            groups=8, 
            activation="swish",
            **kwargs
        ):
        super().__init__(**kwargs)

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        # Down-sampling
        self.downs = []
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append([
                ResidualAdaGNBlock(dim_in, dim_out, groups=groups, activation=activation),
                ResidualAdaGNBlock(dim_in, dim_out, groups=groups, activation=activation),
                Downsample1D(dim_out) if not is_last else tf.identity
            ])

        # Middle block
        mid_dim = dims[-1]
        self.mid_block1 = ResidualAdaGNBlock(mid_dim, mid_dim, groups=groups, activation=activation)
        self.mid_block2 = ResidualAdaGNBlock(mid_dim, mid_dim, groups=groups, activation=activation)

        # Up-sampling
        self.ups = []
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1) # Note: indexing adjustment for reversed list
            self.ups.append([
                #skip connにより2倍になる
                ResidualAdaGNBlock(dim_in * 2, dim_out, groups=groups, activation=activation),
                ResidualAdaGNBlock(dim_out, dim_out, groups=groups, activation=activation),
                Upsample1D(dim_out) if not is_last else tf.identity
            ])

        # Final
        act_fn = tf.keras.activations.get(activation)
        self.final_conv = tf.keras.Sequential([
            tf.keras.layers.Conv1D(dim, dim, padding='same'),
            tfa.layers.GroupNormalization(groups=groups, axis=-1),
            tf.keras.layers.Activation(act_fn),
            tf.keras.layers.Conv1D(transition_dim, 1)  # 出力はノイズ予測
        ])

    def call(self, x, cond_vec):
        
        skips = []

        # Down-sampling
        for resnet1, resnet2, downsample in self.downs:
            x = resnet1([x, cond_vec])
            x = resnet2([x, cond_vec])
            skips.append(x)
            x = downsample(x)

        # Middle block
        x = self.mid_block1([x, cond_vec])
        x = self.mid_block2([x, cond_vec])

        # Up-sampling path
        for resnet1, resnet2, upsample in self.ups:
            # スキップ接続を結合
            skip_connection = skips.pop()
            x = tf.concat([x, skip_connection], axis=-1) # channels-lastなのでaxis=-1
            
            x = resnet1([x, cond_vec])
            x = resnet2([x, cond_vec])
            x = upsample(x)

        out = self.final_conv(x)
        return out


# -------------------------------
# Full Denoise Model
# -------------------------------
class DenoiseModel(tf.keras.Model):
    def __init__(self, transition_dim, cond_len, cond_dim=128, base_channels=64, **kwargs):
        super().__init__(**kwargs)

        self.conditioning = ConditioningModule(
            cond_len, 
            cond_dim=cond_dim, 
            num_heads=4, 
            ff_dim=256, 
            num_cond_blocks=2
        )

        self.unet = UNetBackbone(
            transition_dim,
            cond_dim,
            base_channels=base_channels,
            dim=128,
            dim_mults=(1, 2, 4, 8),
            groups=8, 
            activation="swish",
        )

    def call(self, x_t, t, cond_seq, training=False):
        """
        x_t: noisy input (B, T, C)
        t: timesteps (B,)
        cond_seq: condition sequence (B, T_cond, C_cond)
        """
        cond_vec = self.conditioning(t, cond_seq)  # (B, cond_dim*2)
        out = self.unet(x_t, cond_vec)
        return out

# -------------------------------
# 動作チェック
# -------------------------------
if __name__ == "__main__":
    B, T, C = 4, 128, 1
    T_cond, C_cond = 128, 4

    x_t = tf.random.normal([B, T, C])
    t = tf.random.uniform([B], minval=0, maxval=1000, dtype=tf.int32)
    cond_seq = tf.random.normal([B, T_cond, C_cond])

    model = DenoiseModel(transition_dim=C, cond_len=T)
    y = model(x_t, t, cond_seq)
    print("output shape:", y.shape)  # (B, T, 1)

    model.summary()
