import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


def discrete_autoregreesive_act(decoder, obs_rep, batch_size, n_agent, action_dim, tpdv,
                                available_actions=None, deterministic=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv) # (batch, n_agent, action_dim)
    output_action = torch.zeros((batch_size, n_agent, action_dim), dtype=torch.long) # (batch, n_agent, action_dim)

    for i in range(n_agent):
        logit, log_prob = decoder(shifted_action, obs_rep)[:, i, :]
        output_action[:, i, :] = log_prob
        if i + 1 < n_agent:
            shifted_action[:, i + 1, :] = log_prob
    return output_action
    # output_action：(batch, n_agent, action_dim)


def discrete_parallel_act(decoder, obs_rep, prob, batch_size, n_agent, action_dim, tpdv,
                          available_actions=None):
    # obs_rep：(batch, n_agent, n_emb)
    # prob：(batch, n_agent, action_dim)
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv) # (batch, n_agent, action_dim)
    shifted_action[:, 1:, :] = prob[:, :-1, :]
    logit, log_prob = decoder(shifted_action, obs_rep)

    distri = Categorical(logits=logit)
    entropy = distri.entropy().unsqueeze(-1)
    return log_prob, entropy
    # log_prob：(batch, n_agent, action_dim)

class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                             .view(1, 1, n_agent + 1, n_agent + 1))

        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        #计算批量中所有头部的查询、键和值，并将头部向前计算为批量的维度
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        # 因果注意力 ：(B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side 并排重新组装所有头部输出

        # output projection # (batch, n_agent, embedded)
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # self.attn = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x)) # (batch, n_agent, emb)
        return x


class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x


class Encoder(nn.Module):

    def __init__(self, obs_dim, n_block, n_embd, n_head, n_agent):
        super(Encoder, self).__init__()

        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))

        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1)))

    def forward(self, obs):
        # obs: (batch, n_agent, obs_dim)
        obs_embeddings = self.obs_encoder(obs)
        # obs_embeddings: (batch, n_agent, n_embd)
        x = obs_embeddings

        rep = self.blocks(self.ln(x)) # (batch, n_agent, n_embd)
        v_loc = self.head(rep)  # (batch, n_agent, 1)

        return v_loc, rep


class Decoder(nn.Module):

    def __init__(self, action_dim, n_block, n_embd, n_head, n_agent):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd

        # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))
        self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd, bias=False), activate=True),
                                                nn.GELU())
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, action_dim)))

    # obs, action, and return
    def forward(self, action, obs_rep):
        # action: (batch, n_agent, action_dim),logits
        # obs_rep: (batch, n_agent, n_embd)
        action_embeddings = self.action_encoder(action)
        x = self.ln(action_embeddings)
        for block in self.blocks:
            x = block(x, obs_rep)
        logit = self.head(x) # (batch, n_agent, action_dim)
        log_prob = F.log_softmax(logit)

        return logit, log_prob


class MultiAgentTransformer(nn.Module):

    def __init__(self, obs_dim, action_dim, n_agent,
                 n_block, n_embd, n_head, device=torch.device("cpu")):
        super(MultiAgentTransformer, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device

        self.encoder = Encoder(obs_dim, n_block, n_embd, n_head, n_agent)
        self.decoder = Decoder(action_dim, n_block, n_embd, n_head, n_agent)
        self.to(device)

    def forward(self, obs, prob, available_actions=None):
        # obs: (batch, n_agent, obs_dim)
        # prob: (batch, n_agent, act_dim)
        # available_actions: (batch, n_agent, act_dim)

        obs = check(obs).to(**self.tpdv)
        prob = check(prob).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(obs)[0]
        v_loc, obs_rep = self.encoder(obs)
        prob = prob.long()
        prob_log, entropy = discrete_parallel_act(self.decoder, obs_rep, prob, batch_size,
                                                  self.n_agent, self.action_dim, self.tpdv)

        return prob_log, v_loc, entropy
        # prob_log:(batch, n_agent, action_dim)
        # v_loc:(batch, n_agent, 1)

    def get_actions(self, obs, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(obs)[0]
        v_loc, obs_rep = self.encoder(obs)
        output_action = discrete_autoregreesive_act(self.decoder, obs_rep, batch_size,
                                                    self.n_agent, self.action_dim, self.tpdv,
                                                    deterministic)

        return output_action, v_loc
        # output_action:(batch, n_agent, action_dim)
        # v_loc:(batch, n_agent, 1)

    def get_values(self, obs, available_actions=None):

        obs = check(obs).to(**self.tpdv)
        v_tot, obs_rep = self.encoder(obs)
        return v_tot
        # v_tot:(batch, n_agent, 1)


