# 20250820 release

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from .layers.conditioningLayer import TimestepEmbedding, PositionalEncoding, TransformerEncoder
from .layers.backborneLayer import ResidualAdaGNBlock, Downsample1D, Upsample1D

# -------------------------------
# Conditioning Module: Cross Att
# -------------------------------
class ConditioningModule_CrossAtt(tf.keras.layers.Layer):
    def __init__(self, 
                 cond_len, 
                 cond_dim=128, 
                 num_heads=4, 
                 ff_dim=256, 
                 num_cond_blocks=2, 
                 cfg_dropout=0.1,
                 c_emb_dropout=0.1, 
                 **kwargs):
        super().__init__(**kwargs)
        
        self.cond_dim = cond_dim
        self.num_cond_blocks = num_cond_blocks
        self.cfg_dropout = cfg_dropout

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
                TransformerEncoder(cond_dim, num_heads, ff_dim, 
                                   rate=c_emb_dropout) for _ in range(num_cond_blocks)
            ]
        )

        # 無条件用の trainable embedding
        self.null_embedding = self.add_weight(
            shape=(cond_dim,),
            initializer="zeros",
            trainable=True,
            name="null_embedding"
        )
        
        # クロスアテンションの追加
        self.cross_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=cond_dim)
        self.cross_attention_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # 出力プロジェクション
        self.norm = tf.keras.layers.GlobalAveragePooling1D()
        self.out_proj = tf.keras.layers.Dense(cond_dim)

    def call(self, 
             timestep, 
             cond_seq, 
             use_dropout=True, 
             force_dropout=False, 
             training=None,
             **keywards):
        """
        timestep: (B,) int32
        cond_seq: (B, T, C_cond)
        """
        # sinusoidal embedding for t
        t_emb = self.t_embed(timestep)  # (B, cond_dim)

        # transformer encoding of cond_seq
        cond_x = self.c_embed(cond_seq) # (B, T, cond_dim)
        
        # dropout dor CFG
        if use_dropout:
            # Bernoulli分布のサンプリングを模倣
            mask_shape = (tf.shape(cond_x)[0], 1, 1)
            mask = tf.cast(tf.random.uniform(mask_shape) > self.cfg_dropout, tf.float32)
            cond_x = mask * cond_x + (1 - mask) * tf.broadcast_to(self.null_embedding, tf.shape(cond_x))
            
        if force_dropout:
            # 強制的に無条件にする
            cond_x = tf.broadcast_to(self.null_embedding, tf.shape(cond_x))
            
        # cond_xに注入する
        # Q: cond_x, K: t_emb, V: t_emb
        t_emb_expanded = tf.expand_dims(t_emb, axis=1) # (B, 1, cond_dim) att用
        attn_output = self.cross_attention(query=cond_x, key=t_emb_expanded, value=t_emb_expanded)
        cond_x_with_t = self.cross_attention_norm(cond_x + attn_output) #残差接続と正規化
        cond_vec = self.norm(cond_x_with_t) # (B, cond_dim)
        
        cond_vec = self.out_proj(cond_vec) # (B, cond_dim)

        return cond_vec

# -------------------------------
# Backbone Module: ResAdaGN
# -------------------------------
class UNetBackbone_ResAdaGN(tf.keras.Model):
    def __init__(
            self, 
            transition_dim,
            cond_dim,
            dim=128,
            dim_mults=(1, 2, 4),
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

    def call(self, x, cond_vec, training=None,
             **keywards):
        
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
# Denoise Model: AdaGNUnet
# -------------------------------
class DenoiseModel(tf.keras.Model):
    def __init__(self, 
                 transition_dim, 
                 horizon, 
                 cond_dim=128, 
                 num_heads=4, 
                 ff_dim=256, 
                 num_cond_blocks=2, 
                 cfg_dropout=0.1,
                 c_emb_dropout=0.1,
                 base_channels=64, 
                 dim=128, 
                 dim_mults=(1, 2, 4, 8), 
                 groups=8, 
                 activation="swish",
                 **kwargs
                ):
        super().__init__(name='AdaGNUnet', **kwargs)

        self.conditioning = ConditioningModule_CrossAtt(
            horizon, 
            cond_dim=cond_dim, 
            num_heads=num_heads, 
            ff_dim=ff_dim, 
            num_cond_blocks=num_cond_blocks,
            cfg_dropout=cfg_dropout,
            c_emb_dropout=c_emb_dropout, 
        )

        self.unet = UNetBackbone_ResAdaGN(
            transition_dim,
            cond_dim,
            base_channels=base_channels,
            dim=dim,
            dim_mults=dim_mults,
            groups=groups, 
            activation=activation,
        )

    def call(self, 
             x, 
             cond, 
             t, 
             s=None, 
             use_dropout=True, 
             force_dropout=False, 
             training=None,
             **keywards):
        """
        x_t: noisy input (B, T, C)
        t: timesteps (B,)
        s: condition sequence (B, T_cond, C_cond)
        """
        cond_vec = self.conditioning(t, s, 
                                     use_dropout=use_dropout, 
                                     force_dropout=force_dropout, 
                                     training=training)  # (B, cond_dim*2)
        out = self.unet(x, cond_vec)
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
    print("output shape:", y.shape)  # (B, T, C)

    model.summary()
