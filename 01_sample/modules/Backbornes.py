# 20250820 release

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from .layers.backborneLayer import ResidualAdaGNBlock, Downsample1D, Upsample1D

# -------------------------------
# UNet Backbone
# -------------------------------
class UNetBackbone_ResAdaGN(tf.keras.Model):
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
