# 20250820 release

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from .layers.conditioningLayer import TimestepEmbedding, PositionalEncoding, TransformerEncoder

# -------------------------------
# Conditioning Module
# -------------------------------
class ConditioningModule_CrossAtt(tf.keras.layers.Layer):
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
        cond_x = self.c_embed(cond_seq) # (B, T, cond_dim)
        
        # cond_xに注入する
        # t_embを(B, 1, cond_dim)に拡張してアテンションに適合させる
        t_emb_expanded = tf.expand_dims(t_emb, axis=1) # (B, 1, cond_dim)
        
        # Q: cond_x, K: t_emb, V: t_emb
        attn_output = self.cross_attention(query=cond_x, key=t_emb_expanded, value=t_emb_expanded)
        cond_x_with_t = self.cross_attention_norm(cond_x + attn_output) #残差接続と正規化        
        cond_vec = self.norm(cond_x_with_t) # (B, cond_dim)
        
        cond_vec = self.out_proj(cond_vec) # (B, cond_dim)

        return cond_vec
