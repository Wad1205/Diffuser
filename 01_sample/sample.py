# 20250820 release

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from modules.Backbornes import UNetBackbone_ResAdaGN
from modules.Conditionings import ConditioningModule_CrossAtt

# -------------------------------
# Full Denoise Model
# -------------------------------
class DenoiseModel(tf.keras.Model):
    def __init__(self, transition_dim, cond_len, cond_dim=128, base_channels=64, **kwargs):
        super().__init__(**kwargs)

        self.conditioning = ConditioningModule_CrossAtt(
            cond_len, 
            cond_dim=cond_dim, 
            num_heads=4, 
            ff_dim=256, 
            num_cond_blocks=2
        )

        self.unet = UNetBackbone_ResAdaGN(
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
    print("output shape:", y.shape)  # (B, T, C)

    model.summary()
