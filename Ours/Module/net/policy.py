import torch
import numpy as np
from .transformer import MultiAgentTransformer as MAT
from torch.autograd import Variable
from torch.nn import functional as F

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class TransformerPolicy:
    """
    MAT Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param use_gpu: whether to use gpu.
    """

    def __init__(self, n_block, n_embd, n_head, num_agents,
                 model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        if use_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.l2_const = 1e-4
        self.action_type = 'Discrete'

        self.obs_dim = 16
        if self.action_type == 'Discrete':
            self.act_dim = 25

        print("obs_dim: ", self.obs_dim)
        print("act_dim: ", self.act_dim)

        self.num_agents = num_agents
        self.tpdv = dict(dtype=torch.float32, device=self.device)

        self.transformer = MAT(self.obs_dim, self.act_dim, num_agents,
                               n_block=n_block, n_embd=n_embd, n_head=n_head,
                               device=self.device)

        self.optimizer = torch.optim.Adam(self.transformer.parameters(), weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.transformer.load_state_dict(net_params)


    def get_actions(self, obs, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        """

        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

        actions, values = self.transformer.get_actions(obs,available_actions,deterministic)

        actions = actions.view(-1, self.act_dim)
        values = values.view(-1)
        return actions, values

    def get_values(self, obs, available_actions=None):
        """
        Get value function predictions.

        :return values: (torch.Tensor) value function predictions.
        """
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

        values = self.transformer.get_values(obs, available_actions)

        values = values.view(-1, 1)

        return values

    def evaluate_actions(self, obs, actions, available_actions=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param actions: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        actions = actions.reshape(-1, self.num_agents, self.act_dim)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

        action_log_probs, values, entropy = self.transformer(obs, actions, available_actions)

        action_log_probs = action_log_probs.view(-1, self.act_dim)
        values = values.view(-1)
        entropy = entropy.view(-1, self.act_dim)
        entropy = entropy.mean()

        return values, action_log_probs, entropy

    def save(self, save_dir, episode):
        torch.save(self.transformer.state_dict(), str(save_dir) + "/transformer_" + str(episode) + ".pt")

    def restore(self, model_dir):
        transformer_state_dict = torch.load(model_dir)
        self.transformer.load_state_dict(transformer_state_dict)

    def train_step(self, obs_batch, mcts_probs, value_batch, lr):
        # obs_batch:# (batch, n_agent, obs_dim)
        # mcts_probs:# (batch, n_agent, action_dim)
        # value_batch:# (batch, n_agent, 1)
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            obs_batch = Variable(torch.FloatTensor(np.array(obs_batch)).cuda())
            mcts_probs = Variable(torch.FloatTensor(np.array(mcts_probs)).cuda())
            value_batch = Variable(torch.FloatTensor(np.array(value_batch)).cuda())
        else:
            obs_batch = Variable(torch.FloatTensor(np.array(obs_batch)))
            mcts_probs = Variable(torch.FloatTensor(np.array(mcts_probs)))
            value_batch = Variable(torch.FloatTensor(np.array(value_batch)))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value, _ = self.transformer(obs_batch, mcts_probs)
        # log_act_probs:# (batch, n_agent, action_dim)
        # value:# (batch, n_agent, 1)
        value = value.reshape(-1) # value:# (batch)
        log_act_probs = log_act_probs.reshape(-1,self.act_dim)# log_act_probs:(batch, action_dim)
        value_batch = value_batch.reshape(-1) # value_batch:# (batch)
        mcts_probs = mcts_probs.reshape(-1,self.act_dim)# log_act_probs:(batch, action_dim)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value, value_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        return loss.item(), entropy.item()
