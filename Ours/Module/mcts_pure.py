# -*- coding: utf-8 -*-
"""
A pure implementation of the Monte Carlo Tree Search (MCTS)

@author: Liuhan Yin
"""

import numpy as np
import copy
from operator import itemgetter
import sys
import carla
from utils.utils import AgentState,array2transform,transform2array


def rollout_policy_fn(TreeEnv):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly
    action_probs = np.random.rand(len(TreeEnv.availables))
    return zip(TreeEnv.availables, action_probs)


def policy_value_fn(TreeEnv):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(TreeEnv.availables))/len(TreeEnv.availables)
    return zip(TreeEnv.availables, action_probs), 0


class TreeNode(object):

    def __init__(self, TreeEnv, parent, prior_p):
        self.TreeEnv = TreeEnv
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 1   #avoid 0 to in the denominator
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self.TreeEnv, self, prob)

    def select(self, cput):
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(cput))

    def update(self, leaf_value):
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        # update all ancesstor nodes
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        #self._u = (c_puct * self._P *
        #           np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        self._u = c_puct * self._P * np.sqrt(2 * np.log(self._parent._n_visits) / self._n_visits)
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    def __init__(self, TreeEnv, policy_value_fn, c_puct=5, n_playout=100):
        self.TreeEnv = TreeEnv
        self._root = TreeNode(self.TreeEnv, None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.state_temp = AgentState()

    def _playout(self, state):
        node = self._root
        while(1):
            if node.is_leaf():

                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            print(action)
            self.TreeEnv.step(action)

        action_probs, _ = self._policy(self.TreeEnv)
        # Check for end of game
        end = self.TreeEnv._terminal()
        if not end:
            node.expand(action_probs)
            # Evaluate the leaf node by random rollout
            # take time rollouts and average to get leaf value
        leaf_value = self._evaluate_rollout(state)
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        for i in range(limit):
            end = self.TreeEnv._terminal()
            if end:
                break
            action_probs = rollout_policy_fn(self.TreeEnv)
            max_action = max(action_probs, key=itemgetter(1))[0]
            obs, reward, end, success, info = self.TreeEnv.step(max_action)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        print(f"reward:{reward}")
        if success:
            return 10
        return reward

    def get_move(self, state):
        self.state_temp = copy.deepcopy(state)
        for n in range(self._n_playout):      
            self._playout(self.state_temp)
            print(f"playout:{n}")
            self.TreeEnv.reload(self.state_temp)
          #  self.TreeEnv.reset()
            
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(self.TreeEnv, None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    def __init__(self, TreeEnv, c_puct=5, n_playout=100):
        self.TreeEnv = TreeEnv
        self.mcts = MCTS(self.TreeEnv, policy_value_fn, c_puct, n_playout)
        self.state = AgentState()

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self):
        self.state.pos = self.TreeEnv.ego_state.pos
        pos = transform2array(self.state.pos)
        move = self.mcts.get_move(pos)
        self.mcts.update_with_move(-1)
        #self.TreeEnv.reload(self.state)
        return move

    def __str__(self):
        return "MCTS {}".format(self.player)
