from typing import List, Dict, Any, Optional, Tuple, Union
from collections import namedtuple
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from ding.torch_utils import Adam, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample, \
    qrdqn_nstep_td_data, qrdqn_nstep_td_error, get_nstep_return_data
from ding.policy import Policy
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY, DatasetNormalizer
from ding.utils.data import default_collate, default_decollate
from .common_utils import default_preprocess_learn

@POLICY_REGISTRY.register('dd')
class DDPolicy(Policy):
    config = dict(
        type='dd',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool type) priority: Determine whether to use priority in buffer sample.
        # Default False in SAC.
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (int) Number of training samples(randomly collected) in replay buffer when training starts.
        # Default 10000 in SAC.
        random_collect_size=10000,
        nstep=1,
        # normalizer type
        normalizer='CDFNormalizer',
        model=dict(
            # the type of model
            model='TemporalUnet',
            # config of model
            model_cfg=dict(
                # model dim, In GaussianInvDynDiffusion, it is obs_dim. In others, it is obs_dim + action_dim
                transition_dim=17,
                dim=128,
                dim_mults=[1, 2, 4, 8],
                # whether use return as a condition
                returns_condition=False,
                condition_dropout=0.1,
                # whether use calc energy
                calc_energy=False,
                kernel_size=5,
            ),
            # horizon of tarjectory which generated by model
            horizon=80,
            # timesteps of diffusion
            n_timesteps=1000,
            # hidden dim of action model
            hidden_dim=256,
            returns_condition=False,
            # whether use inverse action learning
            ar_inv=False,
            # whether train inverse action model only
            train_only_inv=False,
            # Whether predict epsilon
            predict_epsilon=True,
            # weight of condition guidance
            condition_guidance_w=0.1,
            # discount of loss
            loss_discount=1.0,
            # whether clip denoise
            clip_denoised=False,
        ),
        learn=dict(

            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=1,
            # (int) Minibatch size for gradient descent.
            batch_size=100,

            # (float type) learning_rate_q: Learning rate for model.
            # Default to 3e-4.
            # Please set to 1e-3, when model.value_network is True.
            learning_rate=3e-4,
            # (bool) Whether ignore done(usually for max step termination env. e.g. pendulum)
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with done is False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,

            # (float type) target_theta: Used for soft update of the target network,
            # aka. Interpolation factor in polyak averaging for target networks.
            # Default to 0.005.
            target_theta=0.005,
            # (float) discount factor for the discounted sum of rewards, aka. gamma.
            discount_factor=0.99,

            # (float) Weight uniform initialization range in the last output layer
            init_w=3e-3,
        ),
        collect=dict(
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        eval=dict(
            # return to go when evaluation
            test_ret=0.9,
            ),
        other=dict(
            replay_buffer=dict(
                # (int type) replay_buffer_size: Max size of replay buffer.
                replay_buffer_size=1000000,
                # (int type) max_use: Max use times of one data in the buffer.
                # Data will be removed once used for too many times.
                # Default to infinite.
                # max_use=256,
            ),
        ),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        return 'dd', ['ding.model.template.diffusion']
    
    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init q, value and policy's optimizers, algorithm config, main and target models.
        """
        # Init
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self.action_dim = self._cfg.model.action_dim
        self.obs_dim = self._cfg.model.obs_dim
        self.ar_inv = self._cfg.model.ar_inv
        self.n_timesteps = self._cfg.model.n_timesteps
        obs = np.random.rand(1, self.obs_dim)
        acs = np.random.rand(1, self.action_dim)
        sets = {'observations': np.array([obs]), 'actions': np.array([acs])}
        self.normalizer = DatasetNormalizer(sets, self._cfg.normalizer, [1])

        # Optimizers
        self._optimizer = Adam(
            self._model.parameters(),
            lr=self._cfg.learn.learning_rate,
        )

        # Algorithm config
        self._gamma = self._cfg.learn.discount_factor

        # Main and target models
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_theta}
        )
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()
        self._target_model.reset()

        self._forward_learn_cnt = 0

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        loss_dict = {}

        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=False
        )
        
        conds = {}
        val = data['condition_val']
        id = data['condition_id']
        for i in range(len(val)):
            if id[i].item() in conds:
                conds[id[i].item()].append(val[i])
            else:
                conds[id[i].item()] = [val[i]]
        for key in conds:
            conds[key] = torch.stack(conds[key])
        data['conditions'] = conds
        data['returns'] = data['returns'].unsqueeze(-1)
        if self._cuda:
            data = to_device(data, self._device)

        x = data['trajectories']
        if self._cfg.model.train_only_inv:
            x_t = x[:, :-1, self.action_dim:]
            a_t = x[:, :-1, :self.action_dim]
            x_t_1 = x[:, 1:, self.action_dim:]
            x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            x_comb_t = x_comb_t.reshape(-1, 2 * self.obs_dim)
            a_t = a_t.reshape(-1, self.action_dim)
            if self.ar_inv:
                loss_dict['policy_loss'] = self._model.inv_model.calc_loss(x_comb_t, a_t)
            else:
                pred_a_t = self._model.inv_model(x_comb_t)
                loss_dict['policy_loss'] = F.mse_loss(pred_a_t, a_t)
            total_loss = loss_dict['policy_loss']
        else:
            batch_size = len(x)
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
            cond = data['conditions']
            returns = data['returns']
            loss_dict['diffuse_loss'] = self._model.p_losses(x[:, :, self.action_dim:], cond, t, returns)
            x_t = x[:, :-1, self.action_dim:]
            a_t = x[:, :-1, :self.action_dim]
            x_t_1 = x[:, 1:, self.action_dim:]
            x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            x_comb_t = x_comb_t.reshape(-1, 2 * self.obs_dim)
            a_t = a_t.reshape(-1, self.action_dim)
            if self.ar_inv:
                loss_dict['policy_loss'] = self._model.inv_model.calc_loss(x_comb_t, a_t)
            else:
                pred_a_t = self._model.inv_model(x_comb_t)
                loss_dict['policy_loss'] = F.mse_loss(pred_a_t, a_t)
            total_loss = (1 / 2) * (loss_dict['diffuse_loss'] + loss_dict['policy_loss'])
        
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        self._forward_learn_cnt += 1
        self._target_model.update(self._learn_model.state_dict())
        return loss_dict
    
    def _monitor_vars_learn(self) -> List[str]:
        return [
            'diffuse_loss', 'policy_loss'
        ]
    
    def _state_dict_learn(self) -> Dict[str, Any]:
        ret = {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }
        return ret
    
    def _init_eval(self):
        self._eval_model = model_wrap(self._target_model, wrapper_name='base')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> Dict[str, Any]:
        data_id = list(data.keys())
        data = default_collate(list(data.values()))

        self._eval_model.eval()
        obs = self.normalizer.normalize(data, 'observations')
        obs = torch.tensor(obs)
        if self._cuda:
            obs = to_device(obs, self._device)
        returns = self._cfg.eval.test_ret * torch.ones(obs.shape[0], 1).to(self._device)
        conditions = {0: obs}
        with torch.no_grad():
            samples = self._eval_model.conditional_sample(conditions, returns)
            obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
            obs_comb = obs_comb.reshape(-1, 2 * self.obs_dim)
            action = self._eval_model.inv_model(obs_comb)
            if self._cuda:
                action = to_device(action, 'cpu')
            action = self.normalizer.unnormalize(action, 'actions')
        action = torch.tensor(action).to('cpu')
        output = {'action': action}
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _init_collect(self) -> None:
        self._unroll_len = self._cfg.collect.unroll_len
        self._gamma = self._cfg.discount_factor  # necessary for parallel
        self._nstep = self._cfg.nstep  # necessary for parallel
        self._collect_model = model_wrap(self._model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: dict, **kwargs) -> dict:
        pass

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        pass

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        r"""
            Overview:
                Get the trajectory and the n step return data, then sample from the n_step return data
            Arguments:
                - data (:obj:`list`): The trajectory's cache
            Returns:
                - samples (:obj:`dict`): The training samples generated
            """
        data = get_nstep_return_data(data, self._nstep, gamma=self._gamma)
        return get_train_sample(data, self._unroll_len)
