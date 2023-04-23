from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple
import torch
import torch.nn.functional as F
import copy

from ding.torch_utils import RMSprop, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import timestep_collate, default_collate, default_decollate
from .base_policy import Policy


@POLICY_REGISTRY.register('icq')
class ICQPolicy(Policy):
    """
    Overview:
        Policy class of ICQ algorithm. ICQ is a multi model reinforcement learning algorithm, \
            you can view the paper in the following link https://arxiv.org/abs/1803.11485
    Interface:
        _init_learn, _data_preprocess_learn, _forward_learn, _reset_learn, _state_dict_learn, _load_state_dict_learn \
            _init_collect, _forward_collect, _reset_collect, _process_transition, _init_eval, _forward_eval \
            _reset_eval, _get_train_sample, default_model
    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      qmix           | RL policy register name, refer to      | this arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     True           | Whether to use cuda for network        | this arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     False          | Whether the RL algorithm is on-policy
                                                        | or off-policy
        4. ``priority``         bool     False          | Whether use priority(PER)              | priority sample,
                                                                                                 | update priority
        5  | ``priority_``      bool     False          | Whether use Importance Sampling        | IS weight
           | ``IS_weight``                              | Weight to correct biased update.
        6  | ``learn.update_``  int      20             | How many updates(iterations) to train  | this args can be vary
           | ``per_collect``                            | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
        7  | ``learn.target_``   float    0.001         | Target network update momentum         | between[0,1]
           | ``update_theta``                           | parameter.
        8  | ``learn.discount`` float    0.99           | Reward's future discount factor, aka.  | may be 1 when sparse
           | ``_factor``                                | gamma                                  | reward env
        == ==================== ======== ============== ======================================== =======================
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='icq',
        # (bool) Whether to use cuda for network.
        cuda=True,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            update_per_collect=20,
            batch_size=32,
            learning_rate=0.0005,
            clip_value=20,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) Target network update momentum parameter.
            # in [0, 1].
            target_update_theta=0.008,
            # (float) The discount factor for future rewards,
            # in [0, 1].
            discount_factor=0.99,
            td_lambda=0.75,
        ),
        collect=dict(
            # (str) path of dataset.
            data_path='/mnt/nfs/zhaochen/ICQ/ICQ-MA/3s_vs_3z.h5'
        ),
        eval=dict(),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        return 'icq', ['ding.model.template.icq']

    def _init_learn(self) -> None:
        """
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the learner model of ICQPolicy
        Arguments:
            .. note::

                The _init_learn method takes the argument from the self._cfg.learn in the config file

            - learning_rate (:obj:`float`): The learning rate fo the optimizer
            - gamma (:obj:`float`): The discount factor
            - agent_num (:obj:`int`): Since this is a multi-agent algorithm, we need to input the agent num.
            - batch_size (:obj:`int`): Need batch size info to init hidden_state plugins
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._optimizer_critic = RMSprop(
            params=self._model.critic.parameters(),
            lr=self._cfg.learn.learning_rate_critic,
            alpha=0.99,
            eps=0.00001,
        )
        self._optimizer_mixer = RMSprop(
            params=self._model.mixer.parameters(),
            lr=self._cfg.learn.learning_rate_critic,
            alpha=0.99,
            eps=0.00001,
        )
        self._optimizer_actor = RMSprop(
            params=self._model.actor.parameters(),
            lr=self._cfg.learn.learning_rate_policy,
            alpha=0.99,
            eps=0.00001,
        )
        self._c_param = list(self._model.critic.parameters()) + list(self._model.mixer.parameters())
        self._gamma = self._cfg.learn.discount_factor
        self._td_lambda = self._cfg.learn.td_lambda

        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_update_theta}
        )
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='hidden_state',
            state_num=self._cfg.learn.batch_size,
            init_fn=lambda: [None for _ in range(self._cfg.model.agent_num)]
        )
        self._learn_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.learn.batch_size,
            init_fn=lambda: [None for _ in range(self._cfg.model.agent_num)]
        )
        self._learn_model.reset()
        self._target_model.reset()

    def _data_preprocess_learn(self, data: List[Any]) -> dict:
        r"""
        Overview:
            Preprocess the data to fit the required data format for learning

        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): the data collected from collect function, the Dict
                in data should contain keys including at least ['obs', 'action', 'reward']

        Returns:
            - data (:obj:`Dict[str, Any]`): the processed data, including at least \
                ['obs', 'action', 'reward', 'done']
        """
        # data preprocess
        assert set(data.keys()) > set(['obs', 'actions', 'reward'])
        if self._cuda:
            data = to_device(data, self._device)
        data['done'] = data['terminated'].float()
        for key in data.keys():
            data[key] = data[key].squeeze(1)
        return data

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, a batch of data for training, values are torch.Tensor or \
                np.ndarray or dict/list combinations.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Dict type data, a info dict indicated training result, which will be \
                recorded in text log and tensorboard, values are python scalar or a list of scalars.
        ArgumentsKeys:
            - necessary: ``obs``, ``next_obs``, ``action``, ``reward``, ``done``, ``filled``, ``avail_action``
        ReturnsKeys:
            - necessary: ``cur_lr``, ``total_loss``
                - cur_lr (:obj:`float`): Current learning rate
                - total_loss (:obj:`float`): The calculated loss
        """
        batch_size = data['actions'].shape[0]
        loss_dict = {}
        data = self._data_preprocess_learn(data)
        # ====================
        # ICQ_critic forward
        # ====================
        self._learn_model.train()
        self._target_model.train()
        self._learn_model.reset()
        self._target_model.reset()
        rewards = data['reward'][:, :-1]
        done = data['done'][:, :-1].float()
        mask = data['filled'][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - done[:, :-1])
        mask_td = mask.repeat(1, 1, self._cfg.model.agent_num).view(batch_size, -1, self._cfg.model.agent_num)
        max_t = data['filled'].sum(1).max(0)[0].item()
        with torch.no_grad():
            target_list = self._target_model.forward(data, mode='compute_critic')
            target_q_IS, target_q_values = target_list['total_q'], target_list['q_value']
            beta = 1000
            advantage_Q = F.softmax(target_q_IS / beta, dim=0)
            target_taken = self._target_model.mixer(
                torch.gather(target_q_values, dim=-1, index=data['actions']).squeeze(-1), data['state']
            ).unsqueeze(-1)
            target_taken = len(advantage_Q) * advantage_Q * target_taken
            target_q_ = target_taken.new_zeros(*target_taken.shape)
            target_q_[:, -1] = target_taken[:, -1] * (1 - torch.sum(done, dim=1))
            for t in range(target_q_.shape[1] - 2, -1, -1):
                target_q_[:, t] = self._td_lambda * self._gamma * target_q_[:, t + 1] + mask[:, t] * \
                    (rewards[:, t] + (1 - self._td_lambda) * self._gamma * target_taken[:,t + 1] * (1 - done[:, t]))
            target_q = target_q_[:, 0:-1]

        crit_loss = []
        total_q = []
        for t in range(max_t - 1):
            mask_t = mask[:, t:t + 1]
            if mask_t.sum() < 0.5:
                continue
            inputs = {}
            for key in data.keys():
                if not isinstance(data[key], list):
                    inputs[key] = data[key][:, t:t + 1]
            q_vals = self._learn_model.forward(inputs, mode='compute_critic')['total_q']
            total_q.append(q_vals)
            q_error = (q_vals - target_q[:, t:t + 1]) * mask_t
            critic_loss = (q_error ** 2).sum() / mask_t.sum()
            crit_loss.append(critic_loss.unsqueeze(0))
            # ====================
            # ICQ_critic update
            # ====================
            self._optimizer_critic.zero_grad()
            self._optimizer_mixer.zero_grad()
            critic_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self._c_param, self._cfg.learn.clip_value)
            self._optimizer_critic.step()
            self._optimizer_mixer.step()
        total_q = torch.cat(total_q, dim=1)
        critic_loss = torch.cat(crit_loss, dim=0)
        loss_dict['critic_loss'] = critic_loss.mean().item()
        # ====================
        # ICQ_actor update
        #=====================
        actions = data['actions'][:, :-1]
        avail_actions = data['avail_actions'][:, :-1]
        mask = mask.repeat(1, 1, self._cfg.model.agent_num).view(-1)
        states = data['state'][:, :-1]

        inputs['avail_actions'] = avail_actions
        inputs['obs'] = data['obs'][:, :-1]
        inputs['state'] = data['state'][:, :-1]
        inputs['actions'] = data['actions'][:, :-1]

        self._learn_model.actor.init_state()
        logits = self._learn_model.forward(inputs, mode='compute_actor')['logit']  # [B,T,A,N]
        logits[avail_actions == 0.0] = 0

        dist = torch.distributions.categorical.Categorical(logits=logits)
        log_p = dist.log_prob(actions.squeeze(-1)).view(batch_size, -1, self._cfg.model.agent_num)

        q_value = self._learn_model.forward(inputs, mode='compute_critic')['q_value'].detach()
        q_taken = torch.gather(q_value, -1, index=actions).squeeze(-1).view(batch_size, -1, self._cfg.model.agent_num)
        baseline = (torch.softmax(logits, dim=-1) * q_value).sum(-1).view(batch_size, -1, self._cfg.model.agent_num).detach()
        adv = (q_taken - baseline)
        beta = 0.1
        adv = F.softmax(adv / beta, dim=0).detach()
        coe = self._learn_model.mixer.k(states).view(batch_size, -1, self._cfg.model.agent_num)
        loss_dict['policy_loss'] = -(coe * (len(adv) * adv * log_p) * mask_td).sum() / mask_td.sum()
        self._optimizer_actor.zero_grad()
        loss_dict['policy_loss'].backward()
        self._optimizer_actor.step()
        # =============
        # after update
        # =============
        self._target_model.update(self._learn_model.state_dict())
        return {
            'total_q': total_q.mean().item() / self._cfg.model.agent_num,
            'target_total_q': target_q.mean().item() / self._cfg.model.agent_num,
            'grad_norm': grad_norm,
            **loss_dict
        }

    def _monitor_vars_learn(self) -> List[str]:
        return [
            'total_q', 'target_total_q', 'grad_norm', 'critic_loss', 'policy_loss',
        ]

    def _state_dict_learn(self) -> Dict[str, Any]:
        ret = {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer_critic': self._optimizer_critic.state_dict(),
            'optimizer_actor': self._optimizer_actor.state_dict(),
            'optimizer_mixer': self._optimizer_mixer.state_dict(),
        }
        return ret

    def _init_eval(self):
        self._eval_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.eval.env_num,
            save_prev_state=False,
            init_fn=lambda: [None for _ in range(self._cfg.model.agent_num)]
        )
        self._eval_model = model_wrap(self._eval_model, wrapper_name='argmax_sample')
        self._eval_model.actor.init_state()
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        ReturnsKeys
            - necessary: ``action``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        data = {'obs': data}
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, data_id=data_id, mode='compute_actor')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _init_collect(self) -> None:
        pass

    def _forward_collect(self, data: dict, **kwargs) -> dict:
        pass

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        pass

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        pass
