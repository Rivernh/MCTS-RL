# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search which uses a policy-value network to 
guide the tree search and evaluate the leaf nodes

@author: Liuhan Yin
"""

import numpy as np
import copy
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

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = np.finfo(float).eps
        self._Q = np.finfo(float).eps
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors, rank=0):
        for action, prob in action_priors:
          #  print(action,prob)
            if rank:
                if action >=21:
                    action = action % 21
                    if action not in self._children:
                        self._children[action] = TreeNode(self, prob)
            else:
                if action < 21:
                    if action not in self._children:
                        self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += leaf_value

    def update_recursive(self, leaf_value):

        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self._u = c_puct * self._P * np.sqrt(2 * np.log(self._parent._n_visits) / self._n_visits)
        self.value = self._Q / self._n_visits + self._u
        return self.value

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


def limit(x,a=0,b=1):
    if x > b:
        return b
    if x < a:
        return a
    return round(x, 3)


class MCTS(object):
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
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
        state_v = np.array([state.UGV_info[0][0], state.UGV_info[0][1], state.UGV_info[0][2], state.speed_temp[0],
                          state.wpt[0][2][0], state.wpt[0][2][1], 0] + [state.UGV_info[1][0], state.UGV_info[1][1],
                                                           state.UGV_info[1][2], state.speed_temp[1],
                                                           state.wpt[1][2][0], state.wpt[1][2][1],1])
        action_probs, leaf_value = self._policy(state_v)
        leaf_value = leaf_value[rank]
        # Check for end of game.
        node.expand(action_probs,rank=rank)
        #leaf_value = state.run_episode(rank)

        # Update value and visit count of nodes in this traversal.
      #  node.update_recursive(leaf_value[rank])
     #   print(leaf_value)
        node.update_recursive(leaf_value)

    def get_move_probs(self, state, rank, temp=1e-3):
        for n in range(self._n_playout):
            state.reset()
            self._playout(state,rank)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        #act_probs = softmax(100 * np.array(visits) / np.sum(np.array(visits)))
        #print(act_probs)
        #print(visits)

        p = [limit(c) for c in act_probs]
        s = 1 - sum(p)
        for i in range(len(p)):
            if (p[i] + s) > 0:
                p[i] += s
                break

        return acts, p

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"

class MCTSPlayer(object):
    def __init__(self,policy_value_fn,
                 c_puct=2, n_playout=200, is_selfplay=0):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, state, rank, temp=1e-3, return_prob=1):
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(state.availables_len)
        acts, probs = self.mcts.get_move_probs(state,rank, temp)
        move_probs[list(acts)] = probs
        if self._is_selfplay:

            move = np.random.choice(
                acts,
                p=0.75*np.array(probs) + 0.25*np.random.dirichlet(0.03*np.ones(len(probs)))
            )
            # update the root node and reuse the search tree
            self.mcts.update_with_move(-1)
        else:

            move = np.random.choice(acts, p=probs)
            # reset the root node
            self.mcts.update_with_move(-1)
        move = int2action(move)
        #state.change_init(move,rank)
        if return_prob:
            return move, move_probs
        else:
            return move

    def __str__(self):
        return "MCTS {}".format(self.player)
