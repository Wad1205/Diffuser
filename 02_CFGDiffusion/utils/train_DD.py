
import os
import pickle
import shutil
import random
import time
import tracemalloc
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

import tensorflow as tf
        
import tensorflow_probability as tfp
tfd = tfp.distributions #tensorflowの確率分布クラス
tfb = tfp.bijectors

#tf.keras.backend.set_floatx('float64')

class DecisionDiffuser:
    def __init__(
            self,
            diffusion_model,
            target_model,
            tau=0.005,
            learning_rate=2e-5,
            gradient_accumulate_every=2,
            step_start_ema=2000,
            update_ema_every=10,
        ):
        
        self.tau = tau
        self.gradient_accumulate_every = gradient_accumulate_every

        # double network
        self.step_start_ema = step_start_ema
        self.update_ema_every = update_ema_every
        self.step = 0
        
        # モデル作成
        self.make_models(diffusion_model, target_model, learning_rate=learning_rate)
    
    def make_models(self, model, target_model, learning_rate=1e-4):
        """モデルの作成"""

        # model
        self.model = model
        self.target_model = target_model

        # Define optimizers
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

        # モデルのコンパイル
        self.model.compile(optimizer=self.optimizer)
        
    def load_model(self, load_dir):
        
        """モデルの読み込み"""
        print(' Load Models...')
        self.model.load_weights(os.path.join(load_dir, 'dd_model', 'weights'))
        self.target_model.load_weights(os.path.join(load_dir, 'dd_target', 'weights'))
        
    def save_models(self, save_dir):
        """モデルの保存"""
        print('save models...')
        os.makedirs(save_dir, exist_ok=True)

        self.model.save_weights(os.path.join(save_dir, 'dd_model', 'weights'))
        try:
            self.target_model.save_weights(os.path.join(save_dir, 'dd_target', 'weights'))
        except:
            pass
        print('save END', save_dir)
   
    def update_target_weights(self, model, target_model, tau=0.005):
        """targetモデルの更新 強化学習の安定のためtargetを学習modelに徐々に近づける"""
        weights = model.get_weights()
        target_weights = target_model.get_weights()
        for i in range(len(target_weights)):  # set tau% of target model to be new weights
            target_weights[i] = tau*weights[i] + (1 - tau)*target_weights[i]
        target_model.set_weights(target_weights)
    
    def train(self, batchs, **args):
        """学習メソッド: 引数は全部tensorであること
        """
        
        # lossの計算
        with tf.GradientTape(persistent=True) as tape:
            for i in range(self.gradient_accumulate_every):
                x = batchs[i][0]
                cond = [[0, batchs[i][0][:,0,:]]]
                s = batchs[i][1]
                loss = self.model.loss_fnc(x, cond, s=s)
                loss = tf.reduce_mean(loss) / self.gradient_accumulate_every

        # apply
        grad = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

        # targetの更新
        if self.step_start_ema<self.step:
            if self.step % self.update_ema_every == 0:
                self.update_target_weights(self.model, self.target_model, self.tau)

        self.step += 1

        return args | {'Loss train': loss}
    
    def validation(self, gt_gen, s, cond_idx=[0], ret_loss=True):
        
        generated = self.model(cond=[[id, gt_gen[:,id,:]] for id in cond_idx], 
                               training=False, 
                               s=s)
        if ret_loss:
            loss = tf.reduce_mean((generated-gt_gen)**2)
        else:
            loss = None

        return {'val_loss': loss}, generated

# 履歴の保存クラス
class History:
    def __init__(self, proc_freq, log_freq, save_dir):
        
        # 履歴用
        self.proc_freq = proc_freq
        self.proc_hist_csv = os.path.join(save_dir, 'history.csv') #リセット時にhist_valsを保存
        self.reset_hist()
        
        # log用 メモリ使用量や実行速度
        self.log_freq = log_freq
        self.log_txt = os.path.join(save_dir, f'log.txt') #ログの保存ファイル
        self.reset_log()
        with open(self.log_txt, 'w') as f:
            f.write(f'LOG Freq={log_freq}\n')
        
    def reset_hist(self):
        self.proc_hist = []
    
    def save_hist(self, epoch, train_loss={}):
        
        # 値をセット
        if len(train_loss)>0:
            self.proc_hist.append([epoch]+list(train_loss.values()))
        else:
            pass
        
        # csvに保存
        if epoch%self.proc_freq==0:
            print('Save History csv ..')
            
            hist_label = ['epoch'] + list(train_loss.keys())
            df_hist = pd.DataFrame(self.proc_hist, columns=hist_label)
            df_hist.set_index("epoch", inplace=True)
            if os.path.exists(self.proc_hist_csv):
                df = pd.read_csv(self.proc_hist_csv, index_col='epoch', encoding='cp932')                
                df = pd.concat([df, df_hist], axis=0).ffill().bfill().fillna(0)
            else:
                df = df_hist.ffill().bfill().fillna(0)
            df.astype(np.float32).to_csv(self.proc_hist_csv, index_label='epoch', encoding='cp932')
            
            self.reset_hist()

    # logファイル
    def reset_log(self):
        self.log_dict = {'epoch': 0, 'train_time':0.,}
        
    def save_log(self, epoch=0, train_time=0.):
        
        self.log_dict['epoch']      = epoch
        self.log_dict['train_time'] += train_time / self.log_freq
        
        if epoch%self.log_freq==0:
            print('Save Log file ...')
    
            # メモリ使用量をセット
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            for i, stat in enumerate(top_stats[:5]):
                self.log_dict[f'alloc_top {i}'] = stat

            # 書き出し
            save_buf = StringIO()
            now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9), 'JST'))
            print('Log: ' + now.strftime('%Y%m%d_%H_%M_%S'), file=save_buf)

            for k, v in self.log_dict.items():
                print(f' {k}:{v}', file=save_buf)

            # 追記
            with open(self.log_txt, 'a') as f:
                f.write(save_buf.getvalue())
            
            self.reset_log()

def TrainDD(
        TrainDsIter, 
        ValidDsIter=None,
        epochs=100, 
        n_steps_per_epoch=10000,
        save_freq=1000, 
        load_flg=False, 
        load_path={'models':None}, 
        temp_dir='.\\temp',
        save_dir='.\\',
        dd_keywords={}
    ):

    # tensorboardでの確認用
    tb_train_writer = tf.summary.create_file_writer(os.path.join(temp_dir, 'logs', 'train'))
    tb_test_writer = tf.summary.create_file_writer(os.path.join(temp_dir, 'logs', 'test'))
    
    """ DDクラス作成
    """
    dd_model = DecisionDiffuser(**dd_keywords)
    
    # モデルのload
    if load_flg:
        dd_model.load_model(load_path['models'])
    
    # 初期モデルの保存
    save_model_dir = os.path.join(save_dir, 'models')
    save_trained_dir = os.path.join(save_model_dir, 'trained_models')
    
    dd_model.save_models(os.path.join(save_model_dir, 'init_models'))
    shutil.copytree(os.path.join(save_model_dir, 'init_models'), save_trained_dir)
    
    """ 学習時の履歴クラス
    """
    history = History(proc_freq=1000, log_freq=1000, save_dir=save_dir)
    
    """ 学習
    """
    for step in range(1, epochs+1):
        
        step_tf = tf.constant(step)
        proc_time = time.time()
        
        # メモリ使用追跡用
        tracemalloc.start()

        """ 学習 per epoch
        """
        train_time = time.time()
        for step_ in range(1, n_steps_per_epoch+1):
            
            # 学習
            step_train_time = time.time()
            batchs = [next(TrainDsIter) for _ in range(dd_model.gradient_accumulate_every)]
            train_ret = dd_model.train(batchs, epoch=step_tf)
            step_train_time = time.time() - step_train_time

            # 表示
            if step_%10==0:
                print(f'{step}/{epochs} epoch ({step_}/{n_steps_per_epoch} step); 10 step_ train time:', 
                      10*step_train_time)
                
                for k, v in train_ret.items():
                    print(f' {k}: {v.numpy()}')
                    
            # モデルの保存
            if step_%save_freq==0:
                print('Save Model ...')
                dd_model.save_models(save_trained_dir)
                
                with open(os.path.join(save_trained_dir, 'memo.txt'), 'w') as f:
                    f.write(f'epoch; {step}/{epochs}\n step; {step_}/{n_steps_per_epoch}')
            
            # Train Tensorbordに記録
            if step_%10==0:
                with tb_train_writer.as_default():
                    for k, v in train_ret.items():
                        tf.summary.scalar(k, v, step_)
                    
        # tensorをnumpyに変換
        train_ret_np = {k: v.numpy() for k, v in train_ret.items()}
        
        train_time = time.time() - train_time
        
        """ 評価
        """
        valid_time = time.time()
        if not ValidDsIter is None:
            print('Valid ...')
            val_x, val_s = next(ValidDsIter)
            valid_ret, _ = dd_model.validation(val_x, val_s)

            # Test TensorBoard に記録
            with tb_test_writer.as_default():
                for k, v in valid_ret.items():
                    tf.summary.scalar(k, v, step)
                    
        valid_time = time.time() - valid_time
        
        """ 他処理
        """
        # historyの保存とメモリ使用量
        history.save_hist(step, train_ret_np)
        history.save_log(step, train_time)
        shutil.copytree(temp_dir, os.path.join(save_dir, 'temp'), dirs_exist_ok=True) #tensorboardログを保存
        
        # ログ出力
        print('***********************************************************')
        print(f'Epoch End: {step}/{epochs}')
        print(f"    proc time: {round(time.time()-proc_time, 3)} sec")
        print(f"   train time: {round(train_time, 3)} sec")
        print(f"   valid time: {round(valid_time, 3)} sec")
        print(f'   train loss: {round(train_ret_np["Loss train"], 4)}')
        print('***********************************************************')
    
    return save_trained_dir

def TestDD(
        TestDsIter,
        load_path={'models':None}, 
        cond_idx=[0],
        save_dir='.\\',
        dd_keywords={},
    ):
    
    """ DDクラス作成
    """
    dd_model = DecisionDiffuser(**dd_keywords)
    
    # モデルのload
    dd_model.load_model(load_path['models'])
    
    # モデルの保存
    save_model_dir = os.path.join(save_dir, 'models')
    dd_model.save_models(os.path.join(save_model_dir, 'test_models'))
    
    """ テスト
    """
    print('Test TestDsIter...')
    ground_truth    = []
    generated       = []
    conditions      = []
    for i, (gt_gen, s) in enumerate(TestDsIter):
        _, gen = dd_model.validation(gt_gen, s, cond_idx=cond_idx, ret_loss=False)
        ground_truth.append(gt_gen.numpy())
        generated.append(gen.numpy())
        conditions.append(s.numpy())
        if (i+1)%10==0:
            print(f' test step {i+1}')
    
    ground_truth    = np.concatenate(ground_truth, axis=0)
    generated       = np.concatenate(generated, axis=0)
    conditions      = np.concatenate(conditions, axis=0)
    
    return ground_truth, generated, conditions