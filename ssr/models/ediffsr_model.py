import logging
from collections import OrderedDict

import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from basicsr.models.sr_model import SRModel
from basicsr.utils import imwrite, tensor2img

from ssr.metrics import calculate_metric
from ssr.utils.model_utils import build_network
import math
import torch
import abc
from tqdm import tqdm
import torchvision.utils as tvutils
import os
from scipy import integrate


class SDE(abc.ABC):
    def __init__(self, T, device=None):
        self.T = T
        self.dt = 1 / T
        self.device = device

    @abc.abstractmethod
    def drift(self, x, t):
        pass

    @abc.abstractmethod
    def dispersion(self, x, t):
        pass

    @abc.abstractmethod
    def sde_reverse_drift(self, x, score, t):
        pass

    @abc.abstractmethod
    def ode_reverse_drift(self, x, score, t):
        pass

    @abc.abstractmethod
    def score_fn(self, x, t):
        pass

    ################################################################################

    def forward_step(self, x, t):
        return x + self.drift(x, t) + self.dispersion(x, t)

    def reverse_sde_step_mean(self, x, score, t):
        return x - self.sde_reverse_drift(x, score, t)

    def reverse_sde_step(self, x, score, t):
        return x - self.sde_reverse_drift(x, score, t) - self.dispersion(x, t)

    def reverse_ode_step(self, x, score, t):
        return x - self.ode_reverse_drift(x, score, t)

    def forward(self, x0, T=-1):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

        return x

    def reverse_sde(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_sde_step(x, score, t)

        return x

    def reverse_ode(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_ode_step(x, score, t)

        return x


#############################################################################


class IRSDE(SDE):
    '''
    Let timestep t start from 1 to T, state t=0 is never used
    '''

    def __init__(self, max_sigma, T=100, schedule='cosine', eps=0.01, device=None):
        super().__init__(T, device)
        self.max_sigma = max_sigma / 255 if max_sigma >= 1 else max_sigma
        self._initialize(self.max_sigma, T, schedule, eps)

    def _initialize(self, max_sigma, T, schedule, eps=0.01):

        def constant_theta_schedule(timesteps, v=1.):
            """
            constant schedule
            """
            print('constant schedule')
            timesteps = timesteps + 1  # T from 1 to 100
            return torch.ones(timesteps, dtype=torch.float32)

        def linear_theta_schedule(timesteps):
            """
            linear schedule
            """
            print('linear schedule')
            timesteps = timesteps + 1  # T from 1 to 100
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        def cosine_theta_schedule(timesteps, s=0.008):
            """
            cosine schedule
            """
            print('cosine schedule')
            timesteps = timesteps + 2  # for truncating from 1 to -1
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:-1]
            return betas

        def get_thetas_cumsum(thetas):
            return torch.cumsum(thetas, dim=0)

        def get_sigmas(thetas):
            return torch.sqrt(max_sigma ** 2 * 2 * thetas)

        def get_sigma_bars(thetas_cumsum):
            return torch.sqrt(max_sigma ** 2 * (1 - torch.exp(-2 * thetas_cumsum * self.dt)))

        if schedule == 'cosine':
            thetas = cosine_theta_schedule(T)
        elif schedule == 'linear':
            thetas = linear_theta_schedule(T)
        elif schedule == 'constant':
            thetas = constant_theta_schedule(T)
        else:
            print('Not implemented such schedule yet!!!')

        sigmas = get_sigmas(thetas)
        thetas_cumsum = get_thetas_cumsum(thetas) - thetas[0]  # for that thetas[0] is not 0
        self.dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = get_sigma_bars(thetas_cumsum)

        self.thetas = thetas.to(self.device)
        self.sigmas = sigmas.to(self.device)
        self.thetas_cumsum = thetas_cumsum.to(self.device)
        self.sigma_bars = sigma_bars.to(self.device)

        self.mu = 0.
        self.model = None

    #####################################

    # set mu for different cases
    def set_mu(self, mu):
        self.mu = mu

    # set score model for reverse process
    def set_model(self, model):
        self.model = model

    #####################################

    def mu_bar(self, x0, t):
        return self.mu + (x0 - self.mu) * torch.exp(-self.thetas_cumsum[t] * self.dt)

    def sigma_bar(self, t):
        return self.sigma_bars[t]

    def drift(self, x, t):
        return self.thetas[t] * (self.mu - x) * self.dt

    def sde_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (self.mu - x) - self.sigmas[t] ** 2 * score) * self.dt

    def ode_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (self.mu - x) - 0.5 * self.sigmas[t] ** 2 * score) * self.dt

    def dispersion(self, x, t):
        return self.sigmas[t] * (torch.randn_like(x) * math.sqrt(self.dt)).to(self.device)

    def get_score_from_noise(self, noise, t):
        return -noise / self.sigma_bar(t)

    def score_fn(self, x, t, **kwargs):
        # need to pre-set mu and score_model
        noise = self.model(x, self.mu, t, **kwargs)
        return self.get_score_from_noise(noise, t)

    def noise_fn(self, x, t, **kwargs):
        # need to pre-set mu and score_model
        return self.model(x, self.mu, t, **kwargs)

    # optimum x_{t-1}
    def reverse_optimum_step(self, xt, x0, t):
        A = torch.exp(-self.thetas[t] * self.dt)
        B = torch.exp(-self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-self.thetas_cumsum[t - 1] * self.dt)

        term1 = A * (1 - C ** 2) / (1 - B ** 2)
        term2 = C * (1 - A ** 2) / (1 - B ** 2)

        return term1 * (xt - self.mu) + term2 * (x0 - self.mu) + self.mu

    def sigma(self, t):
        return self.sigmas[t]

    def theta(self, t):
        return self.thetas[t]

    def get_real_noise(self, xt, x0, t):
        return (xt - self.mu_bar(x0, t)) / self.sigma_bar(t)

    def get_real_score(self, xt, x0, t):
        return -(xt - self.mu_bar(x0, t)) / self.sigma_bar(t) ** 2

    # forward process to get x(T) from x(0)
    def forward(self, x0, T=-1, save_dir='forward_state'):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{t}.png', normalize=False)
        return x

    def reverse_sde(self, xt, T=-1, save_states=False, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t, **kwargs)
            x = self.reverse_sde_step(x, score, t)

            if save_states:  # only consider to save 100 images
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x

    def reverse_ode(self, xt, T=-1, save_states=False, save_dir='ode_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t, **kwargs)
            x = self.reverse_ode_step(x, score, t)

            if save_states:  # only consider to save 100 images
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x

    # sample ode using Black-box ODE solver (not used)
    def ode_sampler(self, xt, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3, ):
        shape = xt.shape

        def to_flattened_numpy(x):
            """Flatten a torch tensor `x` and convert it to numpy."""
            return x.detach().cpu().numpy().reshape((-1,))

        def from_flattened_numpy(x, shape):
            """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
            return torch.from_numpy(x.reshape(shape))

        def ode_func(t, x):
            t = int(t)
            x = from_flattened_numpy(x, shape).to(self.device).type(torch.float32)
            score = self.score_fn(x, t)
            drift = self.ode_reverse_drift(x, score, t)
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(ode_func, (self.T, eps), to_flattened_numpy(xt),
                                       rtol=rtol, atol=atol, method=method)

        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(self.device).type(torch.float32)

        return x

    def optimal_reverse(self, xt, x0, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            x = self.reverse_optimum_step(x, x0, t)

        return x

    ################################################################

    def weights(self, t):
        return torch.exp(-self.thetas_cumsum[t] * self.dt)

    # sample states for training
    def generate_random_states(self, x0, mu):
        x0 = x0.to(self.device)
        mu = mu.to(self.device)

        self.set_mu(mu)

        batch = x0.shape[0]

        timesteps = torch.randint(1, self.T + 1, (batch, 1, 1, 1)).long()

        state_mean = self.mu_bar(x0, timesteps)
        noises = torch.randn_like(state_mean)
        noise_level = self.sigma_bar(timesteps)
        noisy_states = noises * noise_level + state_mean

        return timesteps, noisy_states.to(torch.float32)

    def noise_state(self, tensor):
        return tensor + torch.randn_like(tensor) * self.max_sigma


################################################################################
################################################################################
############################ Denoising SDE ##################################
################################################################################
################################################################################


class DenoisingSDE(SDE):
    '''
    Let timestep t start from 1 to T, state t=0 is never used
    '''

    def __init__(self, max_sigma, T, schedule='cosine', device=None):
        super().__init__(T, device)
        self.max_sigma = max_sigma / 255 if max_sigma > 1 else max_sigma
        self._initialize(self.max_sigma, T, schedule)

    def _initialize(self, max_sigma, T, schedule, eps=0.04):

        def linear_beta_schedule(timesteps):
            timesteps = timesteps + 1
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        def cosine_beta_schedule(timesteps, s=0.008):
            """
            cosine schedule
            as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
            """
            timesteps = timesteps + 2
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            # betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = 1 - alphas_cumprod[1:-1]
            return betas

        def get_thetas_cumsum(thetas):
            return torch.cumsum(thetas, dim=0)

        def get_sigmas(thetas):
            return torch.sqrt(max_sigma ** 2 * 2 * thetas)

        def get_sigma_bars(thetas_cumsum):
            return torch.sqrt(max_sigma ** 2 * (1 - torch.exp(-2 * thetas_cumsum * self.dt)))

        if schedule == 'cosine':
            thetas = cosine_beta_schedule(T)
        else:
            thetas = linear_beta_schedule(T)
        sigmas = get_sigmas(thetas)
        thetas_cumsum = get_thetas_cumsum(thetas) - thetas[0]
        self.dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = get_sigma_bars(thetas_cumsum)

        self.thetas = thetas.to(self.device)
        self.sigmas = sigmas.to(self.device)
        self.thetas_cumsum = thetas_cumsum.to(self.device)
        self.sigma_bars = sigma_bars.to(self.device)

        self.mu = 0.
        self.model = None

    # set noise model for reverse process
    def set_model(self, model):
        self.model = model

    def sigma(self, t):
        return self.sigmas[t]

    def theta(self, t):
        return self.thetas[t]

    def mu_bar(self, x0, t):
        return x0

    def sigma_bar(self, t):
        return self.sigma_bars[t]

    def drift(self, x, x0, t):
        return self.thetas[t] * (x0 - x) * self.dt

    def sde_reverse_drift(self, x, score, t):
        A = torch.exp(-2 * self.thetas_cumsum[t] * self.dt)
        return -0.5 * self.sigmas[t] ** 2 * (1 + A) * score * self.dt

    def ode_reverse_drift(self, x, score, t):
        A = torch.exp(-2 * self.thetas_cumsum[t] * self.dt)
        return -0.5 * self.sigmas[t] ** 2 * A * score * self.dt

    def dispersion(self, x, t):
        return self.sigmas[t] * (torch.randn_like(x) * math.sqrt(self.dt)).to(self.device)

    def get_score_from_noise(self, noise, t):
        return -noise / self.sigma_bar(t)

    def get_init_state_from_noise(self, x, noise, t):
        return x - self.sigma_bar(t) * noise

    def get_init_state_from_score(self, x, score, t):
        return x + self.sigma_bar(t) ** 2 * score

    def score_fn(self, x, t):
        # need to preset the score_model
        noise = self.model(x, t)
        return self.get_score_from_noise(noise, t)

    ############### reverse sampling ################

    def get_real_noise(self, xt, x0, t):
        return (xt - self.mu_bar(x0, t)) / self.sigma_bar(t)

    def get_real_score(self, xt, x0, t):
        return -(xt - self.mu_bar(x0, t)) / self.sigma_bar(t) ** 2

    def reverse_sde(self, xt, x0=None, T=-1, save_states=False, save_dir='sde_state'):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            if x0 is not None:
                score = self.get_real_score(x, x0, t)
            else:
                score = self.score_fn(x, t)
            x = self.reverse_sde_step(x, score, t)

            if save_states:
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x

    def reverse_ode(self, xt, x0=None, T=-1, save_states=False, save_dir='ode_state'):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            if x0 is not None:
                real_score = self.get_real_score(x, x0, t)

            score = self.score_fn(x, t)
            x = self.reverse_ode_step(x, score, t)

            if save_states:
                interval = self.T // 100
                if t % interval == 0:
                    state = x.clone()
                    if x0 is not None:
                        state = torch.cat([x, score, real_score], dim=0)
                    os.makedirs(save_dir, exist_ok=True)
                    idx = t // interval
                    tvutils.save_image(state.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x

    def ode_sampler(self, xt, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3, ):
        shape = xt.shape

        def to_flattened_numpy(x):
            """Flatten a torch tensor `x` and convert it to numpy."""
            return x.detach().cpu().numpy().reshape((-1,))

        def from_flattened_numpy(x, shape):
            """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
            return torch.from_numpy(x.reshape(shape))

        def ode_func(t, x):
            t = int(t)
            x = from_flattened_numpy(x, shape).to(self.device).type(torch.float32)
            score = self.score_fn(x, t)
            drift = self.ode_reverse_drift(x, score, t)
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(ode_func, (self.T, eps), to_flattened_numpy(xt),
                                       rtol=rtol, atol=atol, method=method)

        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(self.device).type(torch.float32)

        return x

    def get_optimal_timestep(self, sigma, eps=1e-6):
        sigma = sigma / 255 if sigma > 1 else sigma
        thetas_cumsum_hat = -1 / (2 * self.dt) * math.log(1 - sigma ** 2 / self.max_sigma ** 2 + eps)
        T = torch.argmin((self.thetas_cumsum - thetas_cumsum_hat).abs())
        return T

    ##########################################################
    ########## below functions are used for training #########
    ##########################################################

    def reverse_optimum_step(self, xt, x0, t):
        A = torch.exp(-self.thetas[t] * self.dt)
        B = torch.exp(-self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-self.thetas_cumsum[t - 1] * self.dt)

        term1 = A * (1 - C ** 2) / (1 - B ** 2)
        term2 = C * (1 - A ** 2) / (1 - B ** 2)

        return term1 * (xt - x0) + term2 * (x0 - x0) + x0

    def optimal_reverse(self, xt, x0, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            x = self.reverse_optimum_step(x, x0, t)

        return x

    def weights(self, t):
        # return 0.1 + torch.exp(-self.thetas_cumsum[t] * self.dt)
        return self.sigmas[t] ** 2

    def generate_random_states(self, x0):
        x0 = x0.to(self.device)

        batch = x0.shape[0]
        timesteps = torch.randint(1, self.T + 1, (batch, 1, 1, 1)).long()

        noises = torch.randn_like(x0, dtype=torch.float32)
        noise_level = self.sigma_bar(timesteps)
        noisy_states = noises * noise_level + x0

        return timesteps, noisy_states


EMA = None
lr_scheduler = None
Lion = None
MatchingLoss = None

from basicsr.utils.registry import MODEL_REGISTRY


logger = logging.getLogger("base")

@MODEL_REGISTRY.register()
class DenoisingModel(SRModel):
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.is_train = False
        # super(DenoisingModel, self).__init__(opt)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        self.opt['scale'] = 4
        self.opt['n_lr_images'] = 1

        # define network and load pretrained models
        self.model = build_network(self.opt).to(self.device)
        if opt["dist"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[torch.cuda.current_device()]
            )
        else:
            # self.model = DataParallel(self.model)
            ...
        # print network
        # self.print_network()
        self.load()
        self.sde = IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=self.device)
        self.sde.set_model(self.model)

        # if self.is_train:
        #     self.model.train()
        #
        #     is_weighted = opt['train']['is_weighted']
        #     loss_type = opt['train']['loss_type']
        #     self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
        #     self.weight = opt['train']['weight']
        #
        #     # optimizers
        #     wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
        #     optim_params = []
        #     for (
        #         k,
        #         v,
        #     ) in self.model.named_parameters():  # can optimize for a part of the model
        #         if v.requires_grad:
        #             optim_params.append(v)
        #         else:
        #             if self.rank <= 0:
        #                 logger.warning("Params [{:s}] will not optimize.".format(k))
        #
        #     if train_opt['optimizer'] == 'Adam':
        #         self.optimizer = torch.optim.Adam(
        #             optim_params,
        #             lr=train_opt["lr_G"],
        #             weight_decay=wd_G,
        #             betas=(train_opt["beta1"], train_opt["beta2"]),
        #         )
        #     elif train_opt['optimizer'] == 'AdamW':
        #         self.optimizer = torch.optim.AdamW(
        #             optim_params,
        #             lr=train_opt["lr_G"],
        #             weight_decay=wd_G,
        #             betas=(train_opt["beta1"], train_opt["beta2"]),
        #         )
        #     elif train_opt['optimizer'] == 'Lion':
        #         self.optimizer = Lion(
        #             optim_params,
        #             lr=train_opt["lr_G"],
        #             weight_decay=wd_G,
        #             betas=(train_opt["beta1"], train_opt["beta2"]),
        #         )
        #     else:
        #         print('Not implemented optimizer, default using Adam!')
        #
        #     self.optimizers.append(self.optimizer)
        #
        #     # schedulers
        #     if train_opt["lr_scheme"] == "MultiStepLR":
        #         for optimizer in self.optimizers:
        #             self.schedulers.append(
        #                 lr_scheduler.MultiStepLR_Restart(
        #                     optimizer,
        #                     train_opt["lr_steps"],
        #                     restarts=train_opt["restarts"],
        #                     weights=train_opt["restart_weights"],
        #                     gamma=train_opt["lr_gamma"],
        #                     clear_state=train_opt["clear_state"],
        #                 )
        #             )
        #     elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
        #         for optimizer in self.optimizers:
        #             self.schedulers.append(
        #                 torch.optim.lr_scheduler.CosineAnnealingLR(
        #                     optimizer,
        #                     T_max=train_opt["niter"],
        #                     eta_min=train_opt["eta_min"])
        #             )
        #     else:
        #         raise NotImplementedError("MultiStepLR learning rate scheme is enough.")
        #
        #     self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
        #     self.log_dict = OrderedDict()

    def feed_data(self, data):
        upsample = nn.Upsample(scale_factor=4, mode='nearest')
        self.lr = LQ = upsample(data["lr"].to(self.device).float()/255)
        self.gt = GT = data.get("hr")
        if GT is not None:
            GT = GT.to(self.device).float()/255
        state = self.sde.noise_state(LQ)
        self.state = state.to(self.device)    # noisy_state
        self.condition = LQ.to(self.device)  # LQ
        if GT is not None:
            self.state_0 = GT.to(self.device)  # GT

    def optimize_parameters(self, step, timesteps, sde=None):
        sde.set_mu(self.condition)

        self.optimizer.zero_grad()

        timesteps = timesteps.to(self.device)

        # Get noise and score
        noise = sde.noise_fn(self.state, timesteps.squeeze())
        score = sde.get_score_from_noise(noise, timesteps)

        # Learning the maximum likelihood objective for state x_{t-1}
        xt_1_expection = sde.reverse_sde_step_mean(self.state, score, timesteps)
        xt_1_optimum = sde.reverse_optimum_step(self.state, self.state_0, timesteps)
        loss = self.weight * self.loss_fn(xt_1_expection, xt_1_optimum)

        loss.backward()
        self.optimizer.step()
        self.ema.update()

        # set log
        self.log_dict["loss"] = loss.item()

    def test(self, sde=None, save_states=False):
        self.sde.set_mu(self.condition)

        self.model.eval()
        with torch.no_grad():
            self.output = self.sde.reverse_sde(self.state, save_states=save_states)

        self.model.eval()

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.condition.detach()[0].float().cpu()
        out_dict["result"] = self.output.detach()[0].float().cpu()
        if need_GT:
            out_dict["gt"] = self.state_0.detach()[0].float().cpu()
        return out_dict

    # def print_network(self, *args, **kwargs):
    #     s, n = self.get_network_description(self.model)
    #     if isinstance(self.model, nn.DataParallel) or isinstance(
    #         self.model, DistributedDataParallel
    #     ):
    #         net_struc_str = "{} - {}".format(
    #             self.model.__class__.__name__, self.model.module.__class__.__name__
    #         )
    #     else:
    #         net_struc_str = "{}".format(self.model.__class__.__name__)
    #     if self.rank <= 0:
    #         logger.info(
    #             "Network G structure: {}, with parameters: {:,d}".format(
    #                 net_struc_str, n
    #             )
    #         )
    #         logger.info(s)

    def load(self):
        load_path_G = self.opt["path"]["pretrain_network_g"]
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.model, self.opt["path"]["strict_load_g"])

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(
                network, DistributedDataParallel
        ):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith("module."):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v

        network.load_state_dict(load_net_clean, strict=strict)

    def save(self, iter_label):
        self.save_network(self.model, "G", iter_label)
        self.save_network(self.ema.ema_model, "EMA", 'lastest')

    def _initialize_best_metric_results(self, dataset_name, metrics2run):
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
            return
        elif not hasattr(self, 'best_metric_results'):
            self.best_metric_results = dict()

        # add a dataset record
        record = dict()
        for metric, content in metrics2run.items():
            better = content.get('better', 'higher')
            init_val = float('-inf') if better == 'higher' else float('inf')
            record[metric] = dict(better=better, val=init_val, iter=-1)
        self.best_metric_results[dataset_name] = record

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']

        with_metrics = False
        if dataset_name == 'test':
            with_metrics = self.opt['test'].get('metrics') is not None
            if with_metrics:
                metrics2run = self.opt['test']['metrics']
        else:
            with_metrics = self.opt['val'].get('metrics') is not None
            if with_metrics:
                metrics2run = self.opt['val']['metrics']

        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in metrics2run.keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name, metrics2run)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            # TODO: the savename logic below does not work for val batch size > 1
            img_name = str(idx)

            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lr
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = os.path.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    save_img_path = os.path.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in metrics2run.items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)