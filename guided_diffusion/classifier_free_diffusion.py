import jittor as jt
import numpy as np
import jittor.nn as nn
from tqdm import tqdm

class GaussianDiffusion(nn.Module):
    def __init__(self, model, betas:np.ndarray, w:float, v:float):
        super().__init__()
        self.model = model
        self.betas = jt.array(betas)
        self.w = w
        self.v = v
        self.T = len(betas)
        self.alphas = 1 - self.betas
        self.log_alphas = jt.log(self.alphas)
        
        self.log_alphas_bar = jt.cumsum(self.log_alphas, dim = 0)
        self.alphas_bar = jt.exp(self.log_alphas_bar)
        # self.alphas_bar = jt.cumprod(self.alphas, dim = 0)
        
        self.log_alphas_bar_prev = nn.pad(self.log_alphas_bar[:-1],[1,0],'constant', 0)
        self.alphas_bar_prev = jt.exp(self.log_alphas_bar_prev)
        self.log_one_minus_alphas_bar_prev = jt.log(1.0 - self.alphas_bar_prev)
        # self.alphas_bar_prev = F.pad(self.alphas_bar[:-1],[1,0],'constant',1)

        # calculate parameters for q(x_t|x_{t-1})
        self.log_sqrt_alphas = 0.5 * self.log_alphas
        self.sqrt_alphas = jt.exp(self.log_sqrt_alphas)
        # self.sqrt_alphas = jt.sqrt(self.alphas)

        # calculate parameters for q(x_t|x_0)
        self.log_sqrt_alphas_bar = 0.5 * self.log_alphas_bar
        self.sqrt_alphas_bar = jt.exp(self.log_sqrt_alphas_bar)
        # self.sqrt_alphas_bar = jt.sqrt(self.alphas_bar)
        self.log_one_minus_alphas_bar = jt.log(1.0 - self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = jt.exp(0.5 * self.log_one_minus_alphas_bar)
        
        # calculate parameters for q(x_{t-1}|x_t,x_0)
        # log calculation clipped because the \tilde{\beta} = 0 at the beginning
        self.tilde_betas = self.betas * jt.exp(self.log_one_minus_alphas_bar_prev - self.log_one_minus_alphas_bar)
        self.log_tilde_betas_clipped = jt.log(jt.concat((self.tilde_betas[1].view(-1), self.tilde_betas[1:]), 0))
        self.mu_coef_x0 = self.betas * jt.exp(0.5 * self.log_alphas_bar_prev - self.log_one_minus_alphas_bar)
        self.mu_coef_xt = jt.exp(0.5 * self.log_alphas + self.log_one_minus_alphas_bar_prev - self.log_one_minus_alphas_bar)
        self.vars = jt.concat((self.tilde_betas[1:2],self.betas[1:]), 0)
        self.coef1 = jt.exp(-self.log_sqrt_alphas)
        self.coef2 = self.coef1 * self.betas / self.sqrt_one_minus_alphas_bar
        # calculate parameters for predicted x_0
        self.sqrt_recip_alphas_bar = jt.exp(-self.log_sqrt_alphas_bar)
        # self.sqrt_recip_alphas_bar = jt.sqrt(1.0 / self.alphas_bar)
        self.sqrt_recipm1_alphas_bar = jt.exp(self.log_one_minus_alphas_bar - self.log_sqrt_alphas_bar)
        # self.sqrt_recipm1_alphas_bar = jt.sqrt(1.0 / self.alphas_bar - 1)
    @staticmethod
    def _extract(coef:jt.Var, t:jt.Var, x_shape:tuple) -> jt.array:
        """
        input:

        coef : an array
        t : timestep
        x_shape : the shape of array x that has K dims(the value of first dim is batch size)

        output:

        a array of shape [batchsize,1,...] where the length has K dims.
        """
        assert t.shape[0] == x_shape[0]

        neo_shape = jt.ones_like(np.array(x_shape)).unary(jt.int32)
        neo_shape[0] = x_shape[0]
        neo_shape = neo_shape.tolist()
        chosen = coef[t]
        chosen = chosen
        return chosen.reshape(neo_shape)

    def q_mean_variance(self, x_0:jt.Var, t:jt.Var) -> tuple[jt.Var, jt.Var]:
        """
        calculate the parameters of q(x_t|x_0)
        """
        mean = self._extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0
        var = self._extract(1.0 - self.sqrt_alphas_bar, t, x_0.shape)
        return mean, var
    
    def q_sample(self, x_0:jt.Var, t:jt.Var) -> tuple[jt.Var, jt.Var]:
        """
        sample from q(x_t|x_0)
        """
        eps = jt.randn_like(x_0)
        return self._extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 \
            + self._extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * eps, eps
    
    def q_posterior_mean_variance(self, x_0:jt.Var, x_t:jt.Var, t:jt.Var) -> tuple[jt.Var, jt.Var, jt.Var]:
        """
        calculate the parameters of q(x_{t-1}|x_t,x_0)
        """
        posterior_mean = self._extract(self.mu_coef_x0, t, x_0.shape) * x_0 \
            + self._extract(self.mu_coef_xt, t, x_t.shape) * x_t
        posterior_var_max = self._extract(self.tilde_betas, t, x_t.shape)
        log_posterior_var_min = self._extract(self.log_tilde_betas_clipped, t, x_t.shape)
        log_posterior_var_max = self._extract(jt.log(self.betas), t, x_t.shape)
        log_posterior_var = self.v * log_posterior_var_max + (1 - self.v) * log_posterior_var_min
        neo_posterior_var = jt.exp(log_posterior_var)
        
        return posterior_mean, posterior_var_max, neo_posterior_var
    def p_mean_variance(self, x_t:jt.Var, t:jt.Var, **model_kwargs) -> tuple[jt.Var, jt.Var]:
        """
        calculate the parameters of p_{theta}(x_{t-1}|x_t)
        """
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        cemb_shape = model_kwargs['cemb'].shape
        pred_eps_cond = self.model(x_t, t, **model_kwargs)
        model_kwargs['cemb'] = jt.zeros(cemb_shape)
        pred_eps_uncond = self.model(x_t, t, **model_kwargs)
        pred_eps = (1 + self.w) * pred_eps_cond - self.w * pred_eps_uncond
        
        assert jt.isnan(x_t).int().sum() == 0, f"nan in array x_t when t = {t[0]}"
        assert jt.isnan(t).int().sum() == 0, f"nan in array t when t = {t[0]}"
        assert jt.isnan(pred_eps).int().sum() == 0, f"nan in array pred_eps when t = {t[0]}"
        p_mean = self._predict_xt_prev_mean_from_eps(x_t, t.long(), pred_eps)
        p_var = self._extract(self.vars, t.long(), x_t.shape)
        return p_mean, p_var

    def _predict_x0_from_eps(self, x_t:jt.Var, t:jt.Var, eps:jt.Var) -> jt.Var:
        return self._extract(coef = self.sqrt_recip_alphas_bar, t = t, x_shape = x_t.shape) \
            * x_t - self._extract(coef = self.sqrt_one_minus_alphas_bar, t = t, x_shape = x_t.shape) * eps

    def _predict_xt_prev_mean_from_eps(self, x_t:jt.Var, t:jt.Var, eps:jt.Var) -> jt.Var:
        return self._extract(coef = self.coef1, t = t, x_shape = x_t.shape) * x_t - \
            self._extract(coef = self.coef2, t = t, x_shape = x_t.shape) * eps

    def p_sample(self, x_t:jt.Var, t:jt.Var, **model_kwargs) -> jt.Var:
        """
        sample x_{t-1} from p_{theta}(x_{t-1}|x_t)
        """
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,), f"size of t is not batch size {B}"
        mean, var = self.p_mean_variance(x_t , t, **model_kwargs)
        assert jt.isnan(mean).int().sum() == 0, f"nan in array mean when t = {t[0]}"
        assert jt.isnan(var).int().sum() == 0, f"nan in array var when t = {t[0]}"
        noise = jt.randn_like(x_t)
        noise[t <= 0] = 0 
        return mean + jt.sqrt(var) * noise
    
    def sample(self, shape:tuple, **model_kwargs) -> jt.Var:
        """
        sample images from p_{theta}
        """
        print('Start generating...')
        if model_kwargs == None:
            model_kwargs = {}
        x_t = jt.randn(shape)
        tlist = jt.ones([x_t.shape[0]]) * self.T
        for _ in tqdm(range(self.T),dynamic_ncols=True):
            tlist -= 1
            with jt.no_grad():
                x_t = self.p_sample(x_t, tlist, **model_kwargs)
        x_t = jt.clamp(x_t, -1, 1)
        print('Ending sampling process...')
        return x_t
    
    def ddim_p_mean_variance(self, x_t:jt.Var, t:jt.Var, prevt:jt.Var, eta:float, **model_kwargs) -> jt.Var:
        """
        calculate the parameters of p_{theta}(x_{t-1}|x_t)
        """
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        cemb_shape = model_kwargs['cemb'].shape
        pred_eps_cond = self.model(x_t, t, **model_kwargs)
        model_kwargs['cemb'] = jt.zeros(cemb_shape)
        pred_eps_uncond = self.model(x_t, t, **model_kwargs)
        pred_eps = (1 + self.w) * pred_eps_cond - self.w * pred_eps_uncond
        
        assert jt.isnan(x_t).int().sum() == 0, f"nan in array x_t when t = {t[0]}"
        assert jt.isnan(t).int().sum() == 0, f"nan in array t when t = {t[0]}"
        assert jt.isnan(pred_eps).int().sum() == 0, f"nan in array pred_eps when t = {t[0]}"

        alphas_bar_t = self._extract(coef = self.alphas_bar, t = t, x_shape = x_t.shape)
        alphas_bar_prev = self._extract(coef = self.alphas_bar_prev, t = prevt + 1, x_shape = x_t.shape)
        sigma = eta * jt.sqrt((1 - alphas_bar_prev) / (1 - alphas_bar_t) * (1 - alphas_bar_t / alphas_bar_prev))
        p_var = sigma ** 2
        coef_eps = 1 - alphas_bar_prev - p_var
        coef_eps[coef_eps < 0] = 0
        coef_eps = jt.sqrt(coef_eps)
        p_mean = jt.sqrt(alphas_bar_prev) * (x_t - jt.sqrt(1 - alphas_bar_t) * pred_eps) / jt.sqrt(alphas_bar_t) + \
            coef_eps * pred_eps
        return p_mean, p_var
    
    def ddim_p_sample(self, x_t:jt.Var, t:jt.Var, prevt:jt.Var, eta:float, **model_kwargs) -> jt.Var: 
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,), f"size of t is not batch size {B}"
        mean, var = self.ddim_p_mean_variance(x_t , t.long(), prevt.long(), eta, **model_kwargs)
        assert jt.isnan(mean).int().sum() == 0, f"nan in array mean when t = {t[0]}"
        assert jt.isnan(var).int().sum() == 0, f"nan in array var when t = {t[0]}"
        noise = jt.randn_like(x_t)
        noise[t <= 0] = 0 
        return mean + jt.sqrt(var) * noise
    
    def ddim_sample(self, shape:tuple, num_steps:int, eta:float, select:str, **model_kwargs) -> jt.Var:
        print('Start generating(ddim)...')
        if model_kwargs == None:
            model_kwargs = {}
        # a subsequence of range(0,1000)
        if select == 'linear':
            tseq = list(np.linspace(0, self.T-1, num_steps).astype(int))
        elif select == 'quadratic':
            tseq = list((np.linspace(0, np.sqrt(self.T), num_steps-1)**2).astype(int))
            tseq.insert(0, 0)
            tseq[-1] = self.T - 1
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{select}"')
        
        x_t = jt.randn(shape)
        tlist = jt.zeros([x_t.shape[0]])
        for i in tqdm(range(num_steps),dynamic_ncols=True):
            with jt.no_grad():
                tlist = tlist * 0 + tseq[-1-i]
                if i != num_steps - 1:
                    prevt = jt.ones_like(tlist) * tseq[-2-i]
                else:
                    prevt = - jt.ones_like(tlist) 
                x_t = self.ddim_p_sample(x_t, tlist, prevt, eta, **model_kwargs)
                jt.cuda.empty_cache()
        x_t = jt.clamp(x_t, -1, 1)
        print('Ending sampling process(ddim)...')
        return x_t
    
    def trainloss(self, x_0:jt.Var, **model_kwargs) -> jt.Var:
        """
        calculate the loss of denoising diffusion probabilistic model
        """
        if model_kwargs == None:
            model_kwargs = {}
        t = jt.randint(self.T, shape=(x_0.shape[0],))
        x_t, eps = self.q_sample(x_0, t)
        pred_eps = self.model(x_t, t, **model_kwargs)
        loss = nn.mse_loss(pred_eps, eps, reduction='mean')
        return loss