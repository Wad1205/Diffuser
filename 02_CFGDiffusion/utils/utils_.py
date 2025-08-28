
import os
import socket
import struct
import copy
import sys
import subprocess
import gc
import random
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class CustomScaler(StandardScaler):
    def __init__(self, *, copy=True, with_mean=True, with_std=True, 
                 org_label=[], mode_dict={}, 
                 pre_process={'gain':{}, 'min':{}, 'max':{}}):
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)

        self.setModeParams(org_label=org_label, mode_dict=mode_dict, 
                           pre_process=pre_process)

    def setModeParams(self, org_label=[], mode_dict={}, 
                      pre_process={'gain':{}, 'min':{}, 'max':{}}):
        
        self.org_label = org_label
        
        self.mode_dict = {lb: mode_dict[lb] for lb in self.org_label 
                          if lb in list(mode_dict.keys())} #{ラベル名 : モード数}
        
        def get_preprocess(pp_={}):
            target = [k for k in list(pp_.keys()) if k in self.org_label]
            id = [self.org_label.index(k) for k in target]
            v  = [pp_[k] for k in target]
            return id, np.array(v)

        self.pp_gain_id, self.pp_gain_v = get_preprocess(pre_process['gain'])
        self.pp_min_id, self.pp_min_v   = get_preprocess(pre_process['min'])
        self.pp_max_id, self.pp_max_v   = get_preprocess(pre_process['max'])

        # one-hot表現使用時の処理
        self.one_hot_keys   = list(self.mode_dict.keys())
        self.one_hot_values = list(self.mode_dict.values())
        self.not_oh_keys    = [lb for lb in self.org_label if not lb in self.one_hot_keys]
        self.one_hot_eye    = [np.eye(v) for v in self.one_hot_values]

        # 正規化後のラベル
        self.normalized_label = copy.deepcopy(self.not_oh_keys)
        for k, v in zip(self.one_hot_keys, self.one_hot_values):
            for i in range(v):
                self.normalized_label += [k] if i==0 else [k+f'_{i}']

        # 各ID
        self.orgID_not_oh   = [self.org_label.index(lb) for lb in self.not_oh_keys]
        self.orgID_oh       = [self.org_label.index(lb) for lb in self.one_hot_keys]

        self.normID_not_oh  = [self.normalized_label.index(lb) for lb in self.not_oh_keys]
        self.normID_oh      = [self.normalized_label.index(lb) for lb in self.one_hot_keys]

        self.norm2orgID     = [self.normalized_label.index(lb) for lb in self.org_label]

        # 正規化を戻す対応
        self.of2org = [self.org_label.index(lb) for lb in self.one_hot_keys]
        
    def pre_process(self, x, addMinMax=False):
        """データの前処理 (最大クリップ, 最小クリップ, 乗算)
        """
        x_ref = x.copy()

        if len(self.pp_gain_id)>0: x_ref[:, self.pp_gain_id] *= self.pp_gain_v
        if len(self.pp_min_id)>0:  x_ref[:, self.pp_min_id] = np.maximum(x_ref[:, self.pp_min_id], 
                                                                         self.pp_min_v)
        if len(self.pp_max_id)>0:  x_ref[:, self.pp_max_id] = np.minimum(x_ref[:, self.pp_max_id], 
                                                                         self.pp_max_v)

        if addMinMax:
            X_min = np.copy(x_ref[:1])
            if len(self.pp_min_id)>0:  X_min[:, self.pp_min_id] = self.pp_min_v

            X_max = np.copy(x_ref[:1])
            if len(self.pp_max_id)>0:  X_max[:, self.pp_max_id] = self.pp_max_v

            x_ref = np.concatenate([x_ref, X_min, X_max], axis=0)
            
        return x_ref
    
    def fit(self, x):
        X = self.pre_process(x, addMinMax=True)
        super().fit(X)

        # -1~1になるように
        self.scale_ = np.max(np.abs(X - self.mean_), axis=0)
        self.var_ = self.scale_**2
    
    def transform(self, x, copy=None):
        # 通常のscaling
        X = self.pre_process(x)
        X_ = super().transform(X, copy=copy)
        X_ = X_[:, self.orgID_not_oh]

        # one-hot表現適用
        if len(self.one_hot_keys)>0:

            # 各データのone-hotベクトル
            Xtrg = X[:, self.orgID_oh].astype(np.int16)
            X_oh = [e[Xtrg[:,i]] for i, e in enumerate(self.one_hot_eye)]
            X_oh = np.concatenate(X_oh, axis=-1)

            # one-hoto表現外のパラメータとone-hotベクトルを結合
            X_ = np.concatenate([X_, X_oh], axis=1)

        return X_

    def inverse_transform(self, X, copy=None):
        
        # デフォルトのscalerで戻す
        X_ = X[:, self.norm2orgID]
        X_ = super().inverse_transform(X_, copy=copy)

        # one-hot表現を戻す
        if len(self.one_hot_keys)>0:
            X_oh = [np.argmax(X[:, i:i+v], axis=1).reshape(-1, 1) 
                        for i, v in zip(self.normID_oh, self.one_hot_values)]
            X_oh = np.concatenate(X_oh, axis=-1)
            X_[:, self.of2org] = X_oh
        return X_
