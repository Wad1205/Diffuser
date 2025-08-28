import tensorflow as tf
import numpy as np
# import copy # Not directly used in the translated TensorFlow code

# Beta linear スケジューラー
def linear_beta_schedule(n_timesteps, beta_start=1e-4, beta_end=0.02):
    """Generates a linear beta schedule."""
    return np.linspace(beta_start, beta_end, n_timesteps)

# Beta cosine スケジューラー
def cosine_beta_schedule(n_timesteps, s=0.008):
    """
    Generates a cosine beta schedule as proposed in
    https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = n_timesteps + 1
    x = np.linspace(0.0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

# Beta vp スケジューラー
def vp_beta_schedule(n_timesteps, beta_min=0.1, beta_max=10.0):
    """
    Generates a Variance Preserving (VP) beta schedule.
    """
    t = np.arange(1, n_timesteps+1)
    T = n_timesteps
    alpha = np.exp(-beta_min/T - 0.5 * (beta_max - beta_min) * (2*t - 1) / T**2)
    betas = 1. - alpha
    return betas

# a から indices t で抽出したものを x_shape 数にreshapeして返す
def extract(a, t, x_shape):
    """
    Extracts values from tensor `a` at indices `t` and reshapes for broadcasting.
    `a`: 1D tensor (e.g., alphas_cumprod).
    `t`: Batch of indices (timesteps).
    `x_shape`: Shape of the tensor `x` (e.g., (batch_size, feature_dim)) to which
               the extracted values will be broadcasted.
    """
    batch_size = tf.shape(t)[0]
    out = tf.gather(a, t) # インデックスtで取得する
    # Reshape 'out' to (batch_size, 1, 1, ...) to match the rank of x_shape for broadcasting
    num_dims_to_add = len(x_shape) - 1
    out_resh = tf.reshape(out, [batch_size] + [1] * num_dims_to_add)
    return out_resh

# --- Placeholder for the Denoiser Model (User needs to define their actual model) ---

def apply_conditioning(x, cond, str_dim=0):
    batch_size = tf.shape(x)[0]
    updated_x = tf.identity(x)

    for t, val in cond:

        # 更新用のインデックスを作る
        b = tf.range(batch_size)  # shape=(batch_size,)
        t = tf.fill([batch_size], t)  # shape=(batch_size,)
        indices = tf.stack([b, t], axis=1)  # shape=(batch_size, 2)

        # scatter update
        updated_x = tf.tensor_scatter_nd_update(updated_x, indices=indices, updates=val)
    return updated_x

# --- Diffusion Model (TensorFlow Version) ---
class Diffusion(tf.keras.Model):
    def __init__(self, 
                 model, 
                 horizon, 
                 observation_dim, 
                 beta_schedule_name='cosine', 
                 n_timesteps=1000,
                 loss_type='l2', 
                 clip_denoised=True, 
                 predict_epsilon=True,
                 noise_ratio=1., 
                 loss_discount=1.0, 
                 use_class_free_guidance=False,
                 condition_guidance_w=0.1, 
                 **args):
        super(Diffusion, self).__init__(**args)

        self.model = model #denoiseモデル
        self.horizon = horizon #時系列長さ
        self.observation_dim = observation_dim #生成する変数のdim
        self.use_class_free_guidance = use_class_free_guidance #推論時CFG補正の有無
        self.condition_guidance_w = condition_guidance_w #推論時のCFG補正係数

        # Made noise_ratio a tf.Variable for dynamic updates (e.g., during eval)
        self.max_noise_ratio = tf.constant(noise_ratio, dtype=tf.float32)
        self.noise_ratio_var = tf.Variable(noise_ratio, trainable=False, 
                                           dtype=tf.float32, name="current_noise_ratio")

        if beta_schedule_name == 'linear':
            betas_np = linear_beta_schedule(n_timesteps)
        elif beta_schedule_name == 'cosine':
            betas_np = cosine_beta_schedule(n_timesteps)
        elif beta_schedule_name == 'vp':
            betas_np = vp_beta_schedule(n_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule_name}")

        betas = tf.constant(betas_np, dtype=tf.float32)

        alphas = 1. - betas
        alphas_cumprod = tf.math.cumprod(alphas, axis=0)
        alphas_cumprod_prev = tf.concat([tf.ones(1, dtype=tf.float32), alphas_cumprod[:-1]], axis=0)

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        # Store constants as tf.Variable (trainable=False) or tf.constant
        # Using tf.Variable ensures they are part of the model's state if saved/loaded
        self.betas = tf.Variable(betas, trainable=False, name="betas")

        self.alphas_cumprod      = tf.Variable(alphas_cumprod, trainable=False, name="alphas_cumprod")
        self.alphas_cumprod_prev = tf.Variable(alphas_cumprod_prev, trainable=False, name="alphas_cumprod_prev")

        self.sqrt_alphas_cumprod = tf.Variable(tf.sqrt(alphas_cumprod), 
                                               trainable=False, name="sqrt_alphas_cumprod")
        self.sqrt_one_minus_alphas_cumprod = tf.Variable(tf.sqrt(1. - alphas_cumprod), 
                                                         trainable=False, name="sqrt_one_minus_alphas_cumprod")
        self.log_one_minus_alphas_cumprod  = tf.Variable(tf.math.log(1. - alphas_cumprod), 
                                                         trainable=False, name="log_one_minus_alphas_cumprod")
        self.sqrt_recip_alphas_cumprod     = tf.Variable(tf.sqrt(1. / alphas_cumprod), 
                                                         trainable=False, name="sqrt_recip_alphas_cumprod")
        self.sqrt_recipm1_alphas_cumprod   = tf.Variable(tf.sqrt(1. / alphas_cumprod - 1.), 
                                                         trainable=False, name="sqrt_recipm1_alphas_cumprod")

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_variance = tf.Variable(posterior_variance, 
                                              trainable=False, name="posterior_variance")

        # Clamping posterior_variance before log to avoid log(0)
        # tf.maximum is used instead of tf.clip_by_value for min only
        clipped_posterior_variance = tf.maximum(posterior_variance, 1e-20)
        self.posterior_log_variance_clipped = tf.Variable(
            tf.math.log(clipped_posterior_variance),
            trainable=False, name="posterior_log_variance_clipped"
        )
        self.posterior_mean_coef1 = tf.Variable(
            betas * tf.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod),
            trainable=False, name="posterior_mean_coef1"
        )
        self.posterior_mean_coef2 = tf.Variable(
            (1. - alphas_cumprod_prev) * tf.sqrt(alphas) / (1. - alphas_cumprod),
            trainable=False, name="posterior_mean_coef2"
        )

        #self.weights = self.get_loss_weights(loss_discount)
        self.loss_fn_instance = tf.keras.losses.MeanSquaredError()
    
    def set_noise_variance(self, noise_var=0.):
        # 除去過程で付与するノイズの分散を設定
        self.noise_ratio_var.assign(noise_var)

    def get_loss_weights(self, discount):
        ''' sets loss coefficients for trajectory
        '''
        self.action_weight = 1
        dim_weights = tf.ones(self.observation_dim, dtype=tf.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** tf.arange(self.horizon, dtype=tf.float32)
        discounts = discounts / discounts.mean()
        loss_weights = tf.einsum('h,t->ht', discounts, dim_weights)
        # Cause things are conditioned on t=0
        if self.predict_epsilon:
            loss_weights[0, :] = 0

        return loss_weights
    
    # ------------------------------------------ sampling ------------------------------------------#
    def predict_start_from_noise(self, x_t, t, noise_pred):
        """
        Predicts x0 from noisy image x_t and predicted noise.
        If self.predict_epsilon is False, noise_pred is actually x0_pred.
        """
        if self.predict_epsilon:
            # √[barAl_t] * x_t - √[1/barAl_t-1] * ε
            term1 = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            term2 = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise_pred
            return term1 - term2
        else:
            return noise_pred # Model directly predicts x0

    def q_posterior(self, x_start, x_t, t):
        """Computes the mean and variance of the posterior q(x_{t-1} | x_t, x_0)."""

        # B_t*√[barAl_t-1]/(1-barAl_t) * x_start + (1-barAl_t-1)*√[Al_t]/(1-barAl_t)
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # B_t*(1-barAl_t-1)/(1-barAl_t)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        
        # Log[B_t*(1-barAl_t-1)/(1-barAl_t)]
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, s=None): # `s` is  conditioning
        """Computes the mean and variance of p_theta(x_{t-1} | x_t)."""

        # Predict
        # √[barAl_t] * x_t - √[1/barAl_t-1] * ε(x_t, t, s)

        # CFG
        if self.use_class_free_guidance:
            # Dropout により条件の削除を行う
            epsilon_cond = self.model(x, cond, t, s=s, use_dropout=False, force_dropout=False)
            epsilon_uncond = self.model(x, cond, t, s=s, use_dropout=False, force_dropout=True)
            model_output = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            model_output = self.model(x, cond, t, use_dropout=False, force_dropout=False)

        # Denoise
        x_recon_from_model = self.predict_start_from_noise(x, t=t, 
                                                           noise_pred=tf.cast(model_output, dtype=x.dtype))

        if self.clip_denoised:
            x_recon_from_model = tf.clip_by_value(x_recon_from_model, -1., 1.)
        else:
            assert self.clip_denoised, "RuntimeError: clip_denoised is False, expecting True."

        # Sampling
        model_mean, posterior_var, posterior_log_var = self.q_posterior(
            x_start=x_recon_from_model, x_t=x, t=t
        )
        return model_mean, posterior_var, posterior_log_var

    def p_sample(self, x, cond, t, s=None):
        """
         Samples x_{t-1} from p_theta(x_{t-1} | x_t).
         by using; Algorithm 4: Diffusion Policy (A Backward Version [Ho et al., 2020]
        """
        
        batch_size = tf.shape(x)[0]
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, s=s)

        noise = 0.5*tf.random.normal(shape=tf.shape(x), dtype=x.dtype)

        # When t == 0, mask noise
        is_zero_t = tf.cast(tf.equal(t, 0), dtype=x.dtype) # (batch_size,)
        num_dims_to_add = len(x.shape) - 1 # x.shape is TensorShape, len() gives rank
        nonzero_mask = tf.reshape(1.0 - is_zero_t, [batch_size] + [1] * num_dims_to_add)

        # sample = model_mean + σ*z
        return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise * self.noise_ratio_var

    def p_sample_loop(self, shape, cond, s=None, n_timesteps=None):
        """Generates samples from the diffusion model through reverse process."""
        
        batch_size = shape[0]

        # Start from pure noise xT
        x = 0.5*tf.random.normal(shape, dtype=tf.float32)
        x = apply_conditioning(x, cond, 0)
        
        if n_timesteps is None:
            n_timesteps = self.n_timesteps
            
        for i in reversed(range(0, n_timesteps)):
            # reverse process
            timesteps = tf.fill((batch_size,), i) # current denoise step
            timesteps = tf.cast(timesteps, dtype=tf.int64) # cast
            x = self.p_sample(x, cond, timesteps, s=s)

            #*********************************************************
            x = apply_conditioning(x, cond, 0)
            #*********************************************************

        return x

    def conditional_sample(self, cond, s=None, horizon=None, *args, **kwargs):
        ''' 
            cond : [ (time, state), ... ]
            s : (batch, n_condition)
        '''

        # 生成データのshape
        batch_size = tf.shape(cond[0][1])[0]
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.observation_dim)

        # reverse 過程で行動のサンプリング
        return self.p_sample_loop(shape, cond, s=s, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#
    def q_sample(self, x_start, t, noise=None):
        """Diffuses the data (x_start) to timestep t (forward process)."""
        if noise is None:
            noise = tf.random.normal(shape=tf.shape(x_start), dtype=x_start.dtype)

        # sample = sqrt(alphas_cumprod_t) * x_start + sqrt(1 - alphas_cumprod_t) * noise
        term1 = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        term2 = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        return term1 + term2

    def p_losses(self, x_start, s, cond, t, weights=1.0):
        """Computes the loss for training the diffusion model."""
        # 初期ノイズ
        noise = tf.random.normal(shape=tf.shape(x_start), dtype=x_start.dtype)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, 0)

        x_recon_pred = self.model(x_noisy, cond, t, s, 
                                  use_dropout=True, force_dropout=False, training=True)

        if not self.predict_epsilon:
            x_recon_pred = apply_conditioning(x_recon_pred, cond, 0)

        tf.debugging.assert_equal(tf.shape(noise), tf.shape(x_recon_pred),
                                  message="Shape mismatch between noise and reconstruction.")

        if self.predict_epsilon:
            target = noise
        else:
            target = x_start

        # This will be mean over batch if default reduction
        loss_value = self.loss_fn_instance(target, x_recon_pred)
        loss_value = loss_value * tf.cast(weights, dtype=loss_value.dtype)

        return loss_value

    @tf.function()
    def loss_fnc(self, x, cond, s=None, weights=1.0):
        """Calculates the training loss for a batch of data
            Note: CFGに関して, 前処理で入力条件condをカットしておく
        """
        self.set_noise_variance(self.max_noise_ratio)

        batch_size = tf.shape(x)[0]
        # Sample random timesteps for each item in the batch
        t = tf.random.uniform(shape=[batch_size], minval=0, 
                              maxval=self.n_timesteps, dtype=tf.int64)
        return self.p_losses(x, s, cond, t, weights=weights)

    # ------------------------------------------ call ------------------------------------------#
    @tf.function()
    def call(self, cond, training=None, *args, **kwargs):
        """
            cond: [[step, value]] denoise step毎の入力の上書き値
            s: 条件
            horizon: 時系列step数 動的な時系列に対応
        """
        self.set_noise_variance(0.)
        return self.conditional_sample(cond, *args, **kwargs)
