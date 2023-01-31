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
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, TreeEnv, parent, prior_p):
        self.TreeEnv = TreeEnv
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 1
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self.TreeEnv, self, prob)

    def select(self, cput):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(cput))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
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
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        #self._u = (c_puct * self._P *
        #           np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        self._u = c_puct * self._P * np.sqrt(2 * np.log(self._parent._n_visits) / self._n_visits)
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, TreeEnv, policy_value_fn, c_puct=5, n_playout=100):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self.TreeEnv = TreeEnv
        self._root = TreeNode(self.TreeEnv, None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.state_temp = AgentState()

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
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
        """Use the rollout policy to play until the end of the game,
        returning +1 if the game success,-1 if false
        """
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
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state
        state includes:a img ,a vector of speed,

        Return: the selected action
        """

        self.state_temp = copy.deepcopy(state)
        for n in range(self._n_playout):      
            self._playout(self.state_temp)
            print(f"playout:{n}")
            self.TreeEnv.reload(self.state_temp)
          #  self.TreeEnv.reset()
            
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move is None:
            self._root = TreeNode(self.TreeEnv, None, 1.0)
            return self._root
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(self.TreeEnv, None, 1.0)
        return self._root

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""
    def __init__(self, TreeEnv, c_puct=5, n_playout=100):
        self.TreeEnv = TreeEnv
        self.mcts = MCTS(self.TreeEnv, policy_value_fn, c_puct, n_playout)
        self.state = AgentState()

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(None)

    def get_action(self):
        self.state.pos = self.TreeEnv.ego_state.pos
        pos = transform2array(self.state.pos)
        move = self.mcts.get_move(pos)
        self.mcts.update_with_move(None)
        #self.TreeEnv.reload(self.state)
        return move

    def __str__(self):
        return "MCTS {}".format(self.player)
