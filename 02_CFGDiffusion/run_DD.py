
# %%
import os
os.chdir(os.path.dirname(__file__))
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" #TFのGPUメモリ制限
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #TFの警告非表示

import zipfile
import io
import copy
import yaml
import json
import glob
import shutil
import random
import time
import pickle
import collections
import gc
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

import tensorflow as tf

from utils.train_DD import TrainDD, TestDD
from utils.diffusion import Diffusion
from utils.modules.AdaGNUnet import DenoiseModel as AdaGNUnet
from utils.modules.TemporalUnet import DenoiseModel as TemporalUnet
from utils.utils_ import CustomScaler

def printConfig(cfg={}, head='Cfg', save_buf=None):
    print(f'{head}:\n ', '\n  '.join(f'{k}: {v}' for k, v in cfg.items()), file=save_buf)
    return save_buf
    
def getDataset(path_lst, label_info={}, scalers=[], horizon=16, 
               shuffle=False, repeat=False, batch_size=None, max_data=0, 
               augmentation=False, ow_cond={}):
    
    csv_files = [p for p in path_lst if '.csv' in p]
    
    gen_label = label_info['generates']
    cond_label = label_info['conditions']

    # csv毎のデータ格納
    gen_x = []
    cond_s = []
    for fi, f in enumerate(csv_files):
        print(f' {fi+1}/{len(csv_files)} load csv: {os.path.basename(f)}')
        df = pd.read_csv(f)
        gen_x.append(df[gen_label].to_numpy())
        cond_s.append(df[cond_label].to_numpy())
    
    # 条件の上書き
    for s in cond_s:    
        for k, v in ow_cond.items():
            s[..., cond_label.index(k)] = v
    
    # ==================== Scalers ====================
    if len(scalers)>0:
        scaler_gen, scaler_cnd = scalers
    else:
        scaler_gen = CustomScaler(org_label=gen_label,
                                  mode_dict=label_info['mode_dict'],
                                  pre_process={k:label_info[k] for k in ['gain', 'min', 'max']})
        scaler_gen.fit(np.concatenate(gen_x, axis=0)) 
        
        scaler_cnd = CustomScaler(org_label=cond_label,
                                   mode_dict=label_info['mode_dict'],
                                   pre_process={k:label_info[k] for k in ['gain', 'min', 'max']})
        scaler_cnd.fit(np.concatenate(cond_s, axis=0)) 
    
    # ==================== データ ====================
    max_horizon = np.min([x.shape[0] for x in gen_x])
    mabiki = 1 if max_data<1 else int(max_horizon/max_data+1)
    make_horizon = lambda x: np.array([x[i:i+horizon] for i in range(0, x.shape[0] - horizon, mabiki)])
    
    if not augmentation:
        # データ生成モードではないとき
        gen_x_ = [scaler_gen.transform(x) for x in gen_x]
        gen_x_ = np.concatenate([make_horizon(x_) for x_ in gen_x_]).astype(np.float32)
        
        cond_s_ = [scaler_cnd.transform(s) for s in cond_s]
        cond_s_ = np.concatenate([make_horizon(s_) for s_ in cond_s_]).astype(np.float32)
    else:
        # データ生成モードの時
        gen_x_ = np.array([scaler_gen.transform(x[-horizon:]) for x in gen_x]).astype(np.float32)
        cond_s_ = np.array([scaler_cnd.transform(s[-horizon:]) for s in cond_s]).astype(np.float32)

    gen_x_ = gen_x_.reshape(-1, horizon, gen_x_.shape[-1])
    cond_s_ = cond_s_.reshape(-1, horizon, cond_s_.shape[-1])
        
    # ==================== TF Dataset の作成 ====================
    n_data = gen_x_.shape[0]
    if batch_size is None: batch_size = n_data
    else: batch_size = min(batch_size, n_data)
    
    if not label_info['use_sequence_cond']:
        # 時系列条件でないとき
        tfDataset = tf.data.Dataset.from_tensor_slices(
                                (gen_x_, np.mean(cond_s_, axis=1)),
                            )
    else:
        # 時系列条件の時
        tfDataset = tf.data.Dataset.from_tensor_slices(
                                (gen_x_, cond_s_),
                            )
    
    # バッチ化
    if shuffle: tfDataset = tfDataset.shuffle(buffer_size=n_data, #毎stepランダム取得
                                              reshuffle_each_iteration=True) #毎spoch順序変更
    if repeat: 
        tfDataset = tfDataset.repeat() #毎step取得可能データに設定, forでのリピート数
    
    tfDataset = tfDataset.batch(batch_size,
                                drop_remainder=True,) #バッチサイズ指定
    tfDataset = tfDataset.prefetch(tf.data.AUTOTUNE) #バックグラウンドで次のデータを用意
        
    # 情報
    ret_cfg = {'name': 'TFdataset',
               'mabiki': mabiki,
               'generate shape': gen_x_.shape,
               'conditoins shape': cond_s_.shape,
               'batch_size': batch_size}
    ret_cfg = ret_cfg | {'n_data': len(csv_files)} | {f' -data_{fi:03d}': f for fi, f in enumerate(csv_files)}
    
    return tfDataset, [scaler_gen, scaler_cnd], ret_cfg

def summary(model, diffusion_model):
    
    # 構造の保存
    buf = io.StringIO()

    print('\n========================================', file=buf)
    model.summary(print_fn=lambda x: buf.write(x + '\n'))

    print('\n========================================', file=buf)
    diffusion_model.summary(print_fn=lambda x: buf.write(x + '\n'))

    return buf

def make_diffusion_model(config, batch):
    
    observation_dim = tf.shape(batch[0])[-1].numpy()
    upd_dict = {'transition_dim': int(observation_dim),
                'horizon': int(config['horizon'])}
    # model = TemporalUnet(
    #             horizon=config['horizon'],
    #             transition_dim=observation_dim, 
    #             cond_dim=config['cond_dim'],
    #             dim=config['dim'],
    #             dim_mults=config['dim_mults'],
    #             condition_dropout=config['condition_dropout'],
    #             act_swish=config['act_swish'],
    #             kernel_size=config['kernel_size'],
    #             use_sequence_cond=config['use_sequence_cond'],
    #         )
    if config['denoise_model'] == 'AdaGNUnet':
        config['denoiser']['AdaGNUnet'].update(upd_dict)
        model = AdaGNUnet(
            **config['denoiser']['AdaGNUnet']
        )
    
    diffusion_model = Diffusion(
                            model=model, 
                            horizon=config['horizon'], 
                            observation_dim=observation_dim, 
                            beta_schedule_name='cosine', 
                            n_timesteps=config['n_timesteps'],
                            loss_type=config['loss_type'], 
                            clip_denoised=config['clip_denoised'], 
                            predict_epsilon=config['predict_epsilon'],
                            noise_ratio=config['noise_ratio'], 
                            loss_discount=config['loss_discount'], 
                            use_class_free_guidance=config['use_class_free_guidance'],
                            condition_guidance_w=config['condition_guidance_w'], 
                            name=config['name']
                        )
    
    # ==================== Build ====================
    # ビルドしないと初期化されない
    # Create dummy input data
    batch_size_example = min(8, tf.shape(batch[0])[0])
    dummy_x = batch[0][:batch_size_example, ...]
    dummy_s = batch[1][:batch_size_example, ...]
    dummy_s = tf.clip_by_value(dummy_s, 0., 1.)
    dummy_t = tf.zeros((batch_size_example))
    cond = [[0, dummy_x[:,0,:]]]
    
    print("Building Models...")
    _ = model(dummy_x, cond, t=dummy_t, s=dummy_s, training=False)

    generated_x = diffusion_model(cond=cond, training=False, 
                                    s=dummy_s, horizon=config['horizon'], n_timesteps=8)

    ret_cfg = {'org shape': dummy_x.shape, 'generated shape': generated_x.shape}
    
    return model, diffusion_model, ret_cfg

def flatten(lst):
    for item in lst:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item
            
if __name__ == '__main__':
    
    save_buf = io.StringIO()
        
    # ==================== 設定の読み込み ====================
    # 実行設定
    with open('config\\config_train.yaml', 'r', encoding='utf-8') as f:
        config_train = yaml.safe_load(f)
    
    # 実行モード
    mode = config_train['mode']
    loadFLG = not mode == 'train'
    trainFLG = mode in ['train', 'retrain']
    augFLG = mode == 'aug' #テスト時にデータ増強用のモードになる
    
    # datasetのcsvリスト ※サブdirにも含まれるcsvファイルをすべて取る
    dataset_lst_fnc = lambda f: [dataset_lst_fnc(f_) for f_ in glob.glob(os.path.join(f, '*'))] if not os.path.isfile(f) else f
    train_set = list(flatten([dataset_lst_fnc(f) for f in config_train['dataset_path']['train']]))
    valid_set = list(flatten([dataset_lst_fnc(f) for f in config_train['dataset_path']['valid']]))
    test_set = list(flatten([dataset_lst_fnc(f) for f in config_train['dataset_path']['test' if not augFLG else 'aug']]))
    
    # configの読み込み モデル設定
    config_f = 'config\\config.yaml' if not loadFLG else config_train['load_path']['config']
    with open(config_f, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # テスト時の生成と保存条件
    if not augFLG:
        test_cond_idx = [0]
        test_split_horizon = [0] + list(range(1, config['horizon']-1, 8)) + [config['horizon']-1]
        test_OWcond = {}
    else:
        test_cond_idx = list(range(0, 8, 1))
        test_split_horizon = [config['horizon']-1] #未使用
        test_OWcond = {'reward': 1.0}
    
    # scalerの読み込み
    scaler_f = None if not loadFLG else config_train['load_path']['scaler']
    if not scaler_f is None:
        with open(scaler_f, 'rb') as f:
            scalers = pickle.load(f)
    else:
        scalers = []
    
    # ==================== ディレクトリの作成 ====================
    # tempディレクトリ
    temp_dir = 'temp'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    # 保存ディレクトリ
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9), 'JST'))
    now = now.strftime('%Y%m%d_%H_%M_%S')
    save_dir = f'dd_{now}_{mode}'
    save_dir = os.path.join('save', save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # %%
    # ==================== Datasetの作成 ====================
    print('**************** Create Dataset ****************')
    if trainFLG:
        print(' TrainDs...')
        TrainDs, scalers, trainDS_cfg = getDataset(
                train_set, 
                label_info=config, 
                scalers=scalers,
                horizon=config['horizon'],
                shuffle=True, 
                repeat=True,
                batch_size=config_train['batch_size'],
            )
        
        print(' ValidDs...')
        ValidDs, _, validDS_cfg = getDataset(
                valid_set,
                label_info=config, 
                scalers=scalers,
                horizon=config['horizon'],
                shuffle=False,  
                repeat=True,
                batch_size=None,
                max_data=256
            )
        
        # 設定の表示    
        save_buf = printConfig(cfg=trainDS_cfg, head='Train DatasetN', save_buf=save_buf)
        save_buf = printConfig(cfg=validDS_cfg, head='Validation DatasetN', save_buf=save_buf)
    
    print(' TestDs...')
    TestDs, _, TestDS_cfg = getDataset(
            test_set,
            label_info=config, 
            scalers=scalers,
            horizon=config['horizon'],
            shuffle=False,  
            repeat=False,
            batch_size=256,
            augmentation=augFLG,
            ow_cond=test_OWcond,
        )
    
    # 設定の表示
    save_buf = printConfig(cfg=TestDS_cfg, head='Test DatasetN', save_buf=save_buf)

    # %%
    # ==================== モデルの構築 ====================
    print('\n**************** Build Model ****************', file=save_buf)
    batch = next(iter(TestDs))
    model, diffusion_model, ret_cfg = make_diffusion_model(config, batch)
    _, target_model, _ = make_diffusion_model(config, batch)
    print('\n'.join(f'{k}: {v}' for k, v in ret_cfg.items()), file=save_buf)    
        
    # summary
    print('\n**************** Summarise Models ****************', file=save_buf)
    buf = summary(model, diffusion_model)
    print(buf.getvalue(), file=save_buf)
    
    # %%
    # ==================== 保存 ====================
    print(save_buf.getvalue())
    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        f.write(save_buf.getvalue())
    
    # config
    os.makedirs(os.path.join(save_dir, 'config'), exist_ok=True)
    for k, v in {'config_train': config_train, 'config': config}.items():
        with open(os.path.join(save_dir, 'config', f'{k}.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump(v, f, indent=2, sort_keys=False, allow_unicode=True)
        
    # scalers
    with open(os.path.join(save_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump(scalers, f)

    # %%
    # ==================== 学習 ====================
    dd_keywords = {
        'diffusion_model': diffusion_model,
        'target_model': target_model,
        'tau': config['tau'],
        'learning_rate': config['learning_rate'],
        'gradient_accumulate_every': config['gradient_accumulate_every'],
        'step_start_ema': config['step_start_ema'],
        'update_ema_every': 5,
    }
    
    if trainFLG:
        # 学習
        print('**************** Train Decision Diffuser ****************')
        model_path = TrainDD(
                iter(TrainDs), 
                ValidDsIter=iter(ValidDs),
                epochs=config_train['n_epochs'], 
                n_steps_per_epoch=config_train['n_steps_per_epoch'],
                save_freq=config_train['save_freq'], 
                load_flg=loadFLG, 
                load_path={'models':config_train['load_path']['models']}, 
                temp_dir='.\\temp',
                save_dir=save_dir,
                dd_keywords=dd_keywords,
            )
        config_train['load_path']['models'] = model_path

    # %%
    # ==================== テスト ====================
    print('**************** Test Decision Diffuser ****************')
    ground_truth, generated, conditions = TestDD(
            TestDsIter=iter(TestDs), 
            cond_idx=test_cond_idx,
            load_path={'models':config_train['load_path']['models']}, 
            save_dir=save_dir,
            dd_keywords=dd_keywords
        )
    
    # %%
    # ==================== テスト結果の保存 ====================
    gen_label = config['generates']
    cond_label = config['conditions']
    scaler_gen = scalers[0]
    scaler_cnd = scalers[1]
    
    save_test_dir = os.path.join(save_dir, 'test_csv')
    os.makedirs(save_test_dir, exist_ok=True)
    
    if not augFLG:
        # 指定したstep先 (t_) の生成データと正解を保存
        for t_ in test_split_horizon:
            t = min(ground_truth.shape[1]-1, t_)
            gt = scaler_gen.inverse_transform(ground_truth[:, t, :])
            gen = scaler_gen.inverse_transform(generated[:, t, :])
            
            pd.DataFrame(gt, columns=gen_label).to_csv(os.path.join(save_test_dir, f'step{t}_ground_truth.csv'), index_label='step')
            pd.DataFrame(gen, columns=gen_label).to_csv(os.path.join(save_test_dir, f'step{t}_generated.csv'), index_label='step')
    else:
        target_files = [v for k, v in TestDS_cfg.items() if '-data_' in k]
        for fi, f in enumerate(target_files):
            df_gt = pd.read_csv(f, encoding='utf-8')
            df_gen = copy.deepcopy(df_gt)
            
            gen = df_gt[gen_label].to_numpy()
            gen[-config['horizon']:, :] = scaler_gen.inverse_transform(generated[fi, :, :])
            df_gen[gen_label] = gen
            
            cnd = df_gt[cond_label].to_numpy()
            if config['use_sequence_cond']:
                cnd[-config['horizon']:, :] = scaler_cnd.inverse_transform(conditions[fi, :, :])
            else:
                cnd[-config['horizon']:, :] = scaler_cnd.inverse_transform(conditions[fi:fi+1, :])
            df_gen[['reward']] = cnd[:, [cond_label.index('reward')]]
            
            df_gt.to_csv(os.path.join(save_test_dir, f'{os.path.basename(f)}'.replace('.csv', '_grt.csv')), 
                         index=False, encoding='utf-8')
            df_gen.to_csv(os.path.join(save_test_dir, f'{os.path.basename(f)}'.replace('.csv', '_gen.csv')), 
                          index=False, encoding='utf-8')

# %%
