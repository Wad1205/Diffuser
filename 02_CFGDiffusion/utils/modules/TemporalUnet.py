# 20250820 release

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from .layers.backborneLayer import Downsample1D, Upsample1D
from .layers.backborneLayer import Mish, act_mish, ResidualTemporalBlock, Conv1dBlock
from .layers.conditioningLayer import SinusoidalPosEmb

# -------------------------------
# Conditioning Module: Temporal
# -------------------------------
class ConditioningModule_Temporal(tf.keras.layers.Layer):
    def __init__(
        self,
        dim=128,
        cfg_dropout=0.1,
        act_swish=False,
        use_sequence_cond=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        # act_fn = tf.keras.activations.mish if not act_swish else tf.keras.activations.swish
        act_fn = Mish(act_mish) if not act_swish else tf.keras.activations.swish

        self.cfg_dropout = cfg_dropout

        self.t_embed = tf.keras.Sequential([
            SinusoidalPosEmb(dim),
            tf.keras.layers.Dense(dim * 4),
            tf.keras.layers.Activation(act_fn),
            tf.keras.layers.Dense(dim),
        ])

        if not use_sequence_cond:
            self.c_embed = tf.keras.Sequential([
                tf.keras.layers.Dense(dim),
                tf.keras.layers.Activation(act_fn),
                tf.keras.layers.Dense(dim * 4),
                tf.keras.layers.Activation(act_fn),
                tf.keras.layers.Dense(dim),
            ])
        else:
            self.c_embed = tf.keras.Sequential([
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(dim)),
                tf.keras.layers.Dense(dim * 4),
                tf.keras.layers.Activation(act_fn),
                tf.keras.layers.Dense(dim),
            ])
            
        # 無条件用の trainable embedding
        self.null_embedding = self.add_weight(
            shape=(dim,),
            initializer="zeros",
            trainable=True,
            name="null_embedding"
        )

    def call(self, 
             timestep, 
             cond_seq, 
             use_dropout=True, 
             force_dropout=False, 
             training=None,
             **keywards):
        '''
            timestep : [ batch ]
            cond_seq : [ batch, (len), N ]
            dropout: CFG用
        '''
        # KerasのConv1Dは (batch, length, channels) を期待するため、
        t  = self.t_embed(timestep)
        c_ = self.c_embed(cond_seq)

        # dropout dor CFG
        if use_dropout:
            # Bernoulli分布のサンプリングを模倣
            mask_shape = (tf.shape(c_)[0], 1)
            mask = tf.cast(tf.random.uniform(mask_shape) > self.cfg_dropout, tf.float32)
            c_ = mask * c_ + (1 - mask) * tf.broadcast_to(self.null_embedding, tf.shape(c_))
            
        if force_dropout:
            # 強制的に無条件にする
            c_ = tf.broadcast_to(self.null_embedding, tf.shape(c_))
        
        return tf.concat([t, c_], axis=-1)

# -------------------------------
# Backbone Module: ResidualTemporal
# -------------------------------
class Backbone_ResidualTemporal(tf.keras.Model):
    def __init__(
        self,
        horizon, 
        transition_dim,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        act_swish=False,
        kernel_size=5,
        **kwargs
    ):
        super().__init__(**kwargs)

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        # horizon, embed_dim は未使用
        self.downs = []
        self.ups = []
        num_resolutions = len(in_out)
        embed_dim = 2 * dim

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append([
                ResidualTemporalBlock(dim_in, dim_out, 
                                      embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, 
                                      mish=not act_swish),
                ResidualTemporalBlock(dim_out, dim_out, 
                                      embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, 
                                      mish=not act_swish),
                Downsample1D(dim_out) if not is_last else tf.identity
            ])
            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, 
                                                embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, 
                                                mish=not act_swish)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, 
                                                embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, 
                                                mish=not act_swish)

        # オリジナルのPyTorchコードでは in_out[1:] の reversed
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1) # Note: indexing adjustment for reversed list
            self.ups.append([
                ResidualTemporalBlock(dim_out * 2, dim_in, 
                                      embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, 
                                      mish=not act_swish),
                ResidualTemporalBlock(dim_in, dim_in, 
                                      embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, 
                                      mish=not act_swish),
                Upsample1D(dim_in) if not is_last else tf.identity
            ])
            if not is_last:
                horizon = horizon * 2

        self.final_conv = tf.keras.Sequential([
            Conv1dBlock(dim, dim, kernel_size=kernel_size, mish=not act_swish),
            tf.keras.layers.Conv1D(transition_dim, 1),
        ])

    def call(self, x, cond_vec, training=None,
             **keywards):
        '''
            x : [ batch, horizon, transition_dim ]
            cond_vec : [ batch, N ]
            dropout: CFG用
        '''

        h = []

        # Down-sampling path
        for resnet1, resnet2, downsample in self.downs:
            x = resnet1(x, cond_vec)
            x = resnet2(x, cond_vec)
            h.append(x)
            x = downsample(x)

        # Middle block
        x = self.mid_block1(x, cond_vec)
        x = self.mid_block2(x, cond_vec)

        # Up-sampling path
        for resnet1, resnet2, upsample in self.ups:
            # スキップ接続を結合
            skip_connection = h.pop()
            x = tf.concat([x, skip_connection], axis=-1) # channels-lastなのでaxis=-1
            
            x = resnet1(x, cond_vec)
            x = resnet2(x, cond_vec)
            x = upsample(x)

        x = self.final_conv(x)

        return x

# -------------------------------
# Denoise Model: TemporalUnet
# -------------------------------
class DenoiseModel(tf.keras.Model):
    def __init__(
        self,
        transition_dim,
        horizon, 
        cond_dim,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        cfg_dropout=0.1,
        act_swish=False,
        kernel_size=5,
        use_sequence_cond=False,
        **kwargs
    ):
        super().__init__(name='TemporalUnet', **kwargs)

        self.conditioning = ConditioningModule_Temporal(
            dim=dim,
            cfg_dropout=cfg_dropout,
            act_swish=act_swish,
            use_sequence_cond=use_sequence_cond,
        )
        
        self.unet = Backbone_ResidualTemporal(
            horizon, 
            transition_dim,
            dim=dim,
            dim_mults=dim_mults,
            act_swish=act_swish,
            kernel_size=kernel_size,
            **kwargs
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
        '''
            x : [ batch, horizon, transition_dim ]
            cond: 未使用
            t : [ batch ]
            s : [ batch, (len), N ]
            dropout: CFG用
        '''

        cond_vec = self.conditioning(t, cond_seq=s, 
                                     use_dropout=use_dropout, 
                                     force_dropout=force_dropout, 
                                     training=training)

        self.unet(x, cond_vec)

        return x

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
