from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.utils import MODEL_REGISTRY
from ding.torch_utils.network.nn_module import MLP

class RelationAggregator(nn.Module):
    def __init__(
            self,
            state_len: int,
            relation_len: int,
    ) -> None:
        super(RelationAggregator, self).__init__()
        self._state_encoder = nn.Sequential(
            nn.Linear(state_len + relation_len, state_len),
            nn.ReLU(inplace=True),
        )

    def forward(self, state, relation, alive_mask):
        relation_avr, relation_max = relation.chunk(2, dim=-1)
        relation_avr = (relation_avr * alive_mask.unsqueeze(1).unsqueeze(-1)).mean(-2)
        relation_max = (relation_max * alive_mask.unsqueeze(1).unsqueeze(-1)).max(-2).values
        state = self._state_encoder(torch.cat([state, relation_avr, relation_max], dim=-1))
        return state
    
class Unify_input(nn.Module):
    def __init__(self, agent_num: int,
                  embed_num: int,
                  state_len: int,
                  hidden_len: int,
                  relation_len: int = None,
                  ) -> None:
        super().__init__()
        self.agent_num = agent_num
        self._action_encoder = MLP(2 * hidden_len + hidden_len + hidden_len, hidden_len, 2 * hidden_len, 2,
                                   activation=nn.ReLU(inplace=True))
        self._state_encoder = MLP(state_len, hidden_len, hidden_len, 2, activation=nn.ReLU(inplace=True))
        self.relation = relation_len is not None
        if self.relation:
            self._relation_encoder = MLP(hidden_len + relation_len, hidden_len, 2 * hidden_len, 2,
                                        activation=nn.ReLU(inplace=True))
            self._relation_aggregator = RelationAggregator(hidden_len, 2 * hidden_len)
        self.embed_num = embed_num
        self._action_embed = nn.Parameter(
            torch.zeros(1, 1, embed_num, hidden_len))  # [batch, agent, no_attack_action_num, hidden]
        nn.init.kaiming_normal_(self._action_embed, mode='fan_out')

    def encode_state(self, obs) -> dict:
        state = obs['states']  # [batch, entity_num, state_len]
        alive_mask = obs['alive_mask']  # [batch, entity_num]
        state = self._state_encoder(state)

        if self.relation:
            relation = obs['relations']  # [batch, entity_num, relation_len]
            relation = self._relation_encoder(
                torch.cat([relation, state.unsqueeze(1).expand(-1, relation.shape[1], -1, -1)], dim=-1))
            state = self._relation_aggregator(state, relation, alive_mask)
            action_embed = self._action_encoder(torch.cat(
                [relation, state.unsqueeze(1).expand(-1, relation.shape[1], -1, -1),
                state.unsqueeze(2).expand(-1, -1, relation.shape[1], -1)], dim=-1))
        else:
            action_embed = self._action_encoder(torch.cat(
                [state.unsqueeze(1).expand(-1, relation.shape[1], -1, -1),
                state.unsqueeze(2).expand(-1, -1, relation.shape[1], -1)], dim=-1))
        return state, action_embed
    
    def forward(self, obs, action) -> dict:
        alive_mask = obs['alive_mask']  # [batch, entity_num]
        action_mask = obs['action_mask']  # [batch, agent_num, action_len]
        state, action_embed = self.encode_state(obs)
        batch_size, agent_num, hidden_len, embed_num, device = state.shape[0], state.shape[1], \
                                                               state.shape[2], self._action_embed.shape[
                                                                   2], state.device
        
        agent_id = torch.LongTensor([i]).to(device)
        active_self_embed = self._action_embed.expand(batch_size, -1, -1, -1)
        passive_none_embed = torch.zeros(batch_size, agent_num, 1, hidden_len).to(device)
        passive_map = torch.cat(
            [torch.zeros(agent_num, embed_num).long().to(device),
             torch.diag(torch.ones(agent_num)).long().to(device)],
            dim=1).view(1, -1, embed_num + agent_num, 1).expand(batch_size, -1, -1,
                                                                hidden_len*2)  # [batch, agent, action_num, hidden]
        # get action embed
        active_embed, passive_embed = action_embed.chunk(2, dim=-1)  # [batch, agent, hidden], [batch, agent, hidden]
        active_embed_ = torch.cat([active_self_embed, active_embed.unsqueeze(1)],
                                  dim=2)  # [batch, 1, active_action, hidden] embed for active agent
        active_embed = active_embed_.scatter_add(2,
                                                 embed_num + agent_id.view(1, 1, 1, 1).expand(batch_size, 1, 1, hidden_len),
                                                 passive_embed.unsqueeze(1).index_select(2, agent_id))
        passive_embed = torch.cat([passive_none_embed, passive_embed.unsqueeze(2)],
                                  dim=2)  # [batch, agent, passive_action(2), hidden] embed for passive agent
        
        # get update state
        active_embed = active_embed_.gather(2, action.view(-1, 1, 1, 1).expand(-1, -1, -1, hidden_len))
        state = state.scatter_add(1, agent_id.view(1, 1, 1).expand(batch_size, 1, hidden_len), active_embed.squeeze(2))
        passive_map = passive_map.gather(2, action.view(-1, 1, 1, 1).expand(-1, agent_num, 1,
                                                                            hidden_len))  # [batch, agent, 1, hidden]
        passive_embed = passive_embed.gather(2, passive_map).squeeze(2)
        state = state + passive_embed
        return state, active_embed, passive_embed