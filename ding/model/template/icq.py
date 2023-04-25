from typing import Union, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from ding.utils import list_split, MODEL_REGISTRY, squeeze, SequenceType
from ding.torch_utils import fc_block, MLP
from .q_learning import DRQN
from ding.model.template.qmix import Mixer
from ding.model.template.coma import COMACriticNetwork


class ICQActorNetwork(nn.Module):
    """
    Overview:
        Decentralized actor network in ICQ.
    Interface:
        __init__, forward, init_state
    """
    def __init__(
        self,
        obs_shape: int,
        action_shape: int,
        hidden_size_list: SequenceType = [128, 128, 64],
    ):
        """
        Overview:
            initialize ICQ actor network
        Arguments:
            - obs_shape (:obj:`int`): the dimension of each agent's observation state
            - action_shape (:obj:`int`): the dimension of action shape
            - hidden_size_list (:obj:`list`): the list of hidden size, default to [128, 128, 64]
        """
        super(ICQActorNetwork, self).__init__()
        self.main = DRQN(obs_shape, action_shape, hidden_size_list, lstm_type='gru')
        self.rnn_hidden_size = hidden_size_list[-1]

    def init_state(self):
        self.hidden_state = torch.zeros(1, self.rnn_hidden_size)

    def forward(self, inputs: Dict) -> Dict:
        """
        ArgumentsKeys:
            - necessary: ``obs``, ``avail_actions``, 
        ReturnsKeys:
            - necessary: ``logit``, ``action_mask``
        """
        agent_state = inputs['obs']
        if len(agent_state.shape) == 3:  # B, A, N
            agent_state = agent_state.unsqueeze(0)
            unsqueeze_flag = True
        else:
            unsqueeze_flag = False
        T, B, A = agent_state.shape[:3]
        agent_state = agent_state.reshape(T, -1, *agent_state.shape[3:])
        prev_state = self.hidden_state
        output = self.main({'obs': agent_state, 'prev_state': prev_state, 'enable_fast_timestep': True})
        logit, next_state = output['logit'], output['next_state']
        next_state, _ = list_split(next_state, step=A)
        self.hidden_state = next_state
        logit = logit.reshape(T, B, A, -1)
        if unsqueeze_flag:
            logit = logit.squeeze(0)
        return {'logit': logit.transpose(0, 1), 'action_mask': inputs['avail_actions']}


@MODEL_REGISTRY.register('icq')
class ICQ(nn.Module):
    """
    Overview:
        ICQ network
    Interface:
        __init__, forward, _setup_global_encoder
    """
    mode = ['compute_actor', 'compute_critic', 'compute_eval']

    def __init__(
            self,
            agent_num: int,
            obs_shape: int,
            action_shape: Union[int, SequenceType],
            actor_hidden_size_list: SequenceType,
            global_obs_shape: int,
            embedding_size: int,
            critic_hidden_size: int,
            activation: nn.Module = nn.ReLU(),
    ) -> None:
        """
        Overview:
            Initialize QMIX neural network, i.e. agent Q network and mixer.
        Arguments:
            - agent_num (:obj:`int`): the number of agent
            - obs_shape (:obj:`int`): the dimension of each agent's observation state
            - global_obs_shape (:obj:`int`): the dimension of global observation state
            - action_shape (:obj:`int`): the dimension of action shape
            - actor_hidden_size_list (:obj:`list`): the list of hidden size of actor
            - critic_hidden_size (:obj:'init'): the hidden size of critic
            - embedding_size (:obj:'int'): the embedding size of mixer
            - mixer (:obj:`bool`): use mixer net or not, default to True
            - lstm_type (:obj:`str`): use lstm or gru, default to gru
            - activation (:obj:`nn.Module`): Activation function in network, defaults to nn.ReLU().
            - dueling (:obj:`bool`): use dueling head or not, default to False.
        """
        super(ICQ, self).__init__()
        action_shape = squeeze(action_shape)
        actor_input_size = squeeze(obs_shape)
        critic_input_size = squeeze(obs_shape) + squeeze(global_obs_shape) + \
            agent_num * action_shape + (agent_num - 1) * action_shape
        self.actor = ICQActorNetwork(actor_input_size, action_shape, actor_hidden_size_list)
        self.critic = COMACriticNetwork(critic_input_size, action_shape, critic_hidden_size)
        self.mixer = Mixer(agent_num, global_obs_shape, embedding_size, activation=activation)
        self._global_state_encoder = nn.Identity()

    def forward(self, inputs: Dict, mode: str) -> Dict:
        """
        ArgumentsKeys:
            - necessary: ``obs`` { ``agent_state``, ``global_state``, ``action_mask`` }, ``action``
        ReturnsKeys:
            - necessary:
                - compute_critic: ``q_value``
                - compute_actor: ``logit``, ``next_state``, ``action_mask``
        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        if mode != 'compute_eval':
            x = {}
            for key in inputs.keys():
                if torch.is_tensor(inputs[key]):
                    x[key] = inputs[key].transpose(0, 1)
        else:
            x = {}
            x['obs'] = inputs['obs']['agent_state']
            x['avail_actions'] = inputs['obs']['action_mask']
        if mode == 'compute_actor':
            return self.actor(x)
        elif mode == 'compute_critic':
            global_state = x['state'].unsqueeze(0)
            global_state_embedding = self._global_state_encoder(global_state)
            data = {}
            data['action'] = x['actions']
            data['obs'] = {}
            data['obs']['agent_state'] = x['obs']
            data['obs']['global_state'] = x['state']
            q_values = self.critic(data)['q_value']
            q_taken = torch.gather(q_values, dim=-1, index=x['actions']).squeeze(3)
            total_q = self.mixer(q_taken, global_state_embedding)
            return {'total_q': total_q.transpose(0, 1).unsqueeze(-1), 'q_value': q_values.transpose(0, 1)}
        elif mode == 'compute_eval':
            return self.actor(x)
