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
import time as t
import itertools
def int2action(int):
    availables = list(itertools.product(np.array([2.5, 5, 7.5, 10]), np.array([-0.4,-0.2, 0, 0.2, 0.4])))
    availables.insert(0,(0, 0))
    return availables[int]

def action2int(action):
    if action[0]:
        line = 10 * action[0] / 25
        row = 100 * action[1] / 20 + 2
        move = line * 5 + row - 4
    else:
        move = 0
    return int(move)


def rollout_policy_fn(TreeEnv):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly
    action_probs = np.random.rand(len(TreeEnv.availables))
    return zip(TreeEnv.availables, action_probs)


def policy_value_fn(TreeEnv):
    """a function that takes in a obs and outputs a list of (action, probability)
    tuples and a score for the obs"""
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(TreeEnv.availables))/len(TreeEnv.availables)
    return zip(TreeEnv.availables, action_probs), 0


class TreeNode(object):

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0.0000001 #avoid 0 to in the denominator
        self._Q = np.finfo(float).eps
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, cput):
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(cput))

    def update(self, leaf_value):
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += leaf_value

    def update_recursive(self, leaf_value):
        # update all ancesstor nodes
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        #self._u = (c_puct * self._P *
        #           np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        self._u = c_puct * self._P * np.sqrt(2 * np.log(self._parent._n_visits) / self._n_visits)
        return self._Q / self._n_visits + self._u

    def is_leaf(self):
        return self._children == {}

    def leaf(self):
        return self._n_visits < 2

    def is_root(self):
        return self._parent is None

    def print(self,cnt):
        cnt += 1
        for action,node in self._children.items():
            print(f"action:{action}   time:{node._n_visits}   value:{node._Q / node._n_visits}    current layer:{cnt}")
            #print(node.leaf())
            if node.leaf():
                continue
            node.print(cnt)


class MCTS(object):
    def __init__(self, policy_value_fn, c_puct=5, n_playout=100):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state, rank):
        node = self._root
        while(1):
            if node.is_leaf():

                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            action = int2action(action)
            state.do_move(action, rank)

        action_probs, _ = self._policy(state)
        # Check for end of game
        node.expand(action_probs)
        # Evaluate the leaf node by random rollout
        # take time rollouts and average to get leaf value
        #s = t.clock()
        leaf_value = state.run_episode(rank)
        #e = t.clock()
        #print(f"time:{e - s}")

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(leaf_value)

    def get_move(self, state, rank):
        for n in range(self._n_playout):
            state.reset()
            self._playout(state, rank)
            #print(f"play out:{n}")
        #for action,node in self._root._children.items():
        #    print(f"action:{action}   time:{node._n_visits}   value:{node._Q / node._n_visits}")
        #if rank == 0:
        #    self._root.print(0)


        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    def __init__(self, c_puct=5, n_playout=100):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, state, rank):
        #s = t.clock()
        move = self.mcts.get_move(state, rank)
        move = int2action(move)
        #state.change_init(move,rank)
        #e = t.clock()#
        #print(f"time:{e - s}")
        self.mcts.update_with_move(-1)
        return move

    def game_move(self, state):
        move_rew = []
        move_fowllower = []
        move_leader = []
        for i in range(state.availables_len):
            state.reset()
            action = int2action(i)
            state.do_move(action, 0)
            action_fol = self.get_action(state,1)
            move_fowllower.append(action_fol)
            state.reset()
            state.do_move(action_fol, 1)
            action_lea = self.get_action(state,0)
            move_leader.append(action_lea)
            move_rew.append(state.run_episode(0))

        index = np.argmax(np.array(move_rew))
        index_f = action2int(move_leader[index])
        move = [move_leader[index], move_fowllower[index_f]]

        return move


    def __str__(self):
        return "MCTS {}".format(self.player)
