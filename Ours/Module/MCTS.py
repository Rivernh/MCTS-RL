"""
Adapted from https://github.com/tensorflow/minigo/blob/master/mcts.py

Implementation of the Monte-Carlo tree search algorithm as detailed in the
AlphaGo Zero paper (https://www.nature.com/articles/nature24270).
"""
import math
import random as rd
import collections
import numpy as np

# Exploration constant计算UCB公式 控制探索的占比权重
c_PUCT = 1.38
# Dirichlet noise alpha parameter.可以控制进所有动作的尝试
D_NOISE_ALPHA = 0.03
# Number of steps into the episode after which we always select the
# action with highest action probability rather than selecting randomly 决策的深度，即N
TEMP_THRESHOLD = 5


class DummyNode:
    """
    Special node that is used as the node above the initial root node to
    prevent having to deal with special cases when traversing the tree.
    用于初始根节点之上的特殊节点
    避免在遍历树时不得不处理特殊情况
    """

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)

    def revert_virtual_loss(self, up_to=None): pass

    def add_virtual_loss(self, up_to=None): pass

    def revert_visits(self, up_to=None): pass

    def backup_value(self, value, up_to=None): pass


class MCTSNode:
    """
    Represents a node in the Monte-Carlo search tree. Each node holds a single
    environment state.
    表示蒙特卡罗搜索树中的节点。每个节点都有一个环境状态
    """

    def __init__(self, state, n_actions, TreeEnv, action=None, parent=None):
        """
        :param state: State that the node should hold.
        :param n_actions: Number of actions that can be performed in each
        state. Equal to the number of outgoing edges of the node.
        :param TreeEnv: Static class that defines the environment dynamics,
        e.g. which state follows from another state when performing an action.
        :param action: Index of the action that led from the parent node to
        this node.
        :param parent: Parent node.

        :param state:节点应该持有的状态。
        :param n_actions:每个操作中可执行的操作数状态。等于节点的出边数。
        :param TreeEnv: 定义环境动态的静态类
        例如，当执行一个动作时，哪个状态紧随另一个状态。
        :param action:父节点指向的动作索引这个节点。
        :param parent:父节点。
        """
        self.TreeEnv = TreeEnv
        if parent is None:
            self.depth = 0
            parent = DummyNode()
        else:
            self.depth = parent.depth+1
        self.parent = parent
        self.action = action
        self.state = state
        self.n_actions = n_actions
        self.is_expanded = False
        self.n_vlosses = 0  # Number of virtual losses on this node 该节点的虚拟损失数
        self.child_N = np.zeros([n_actions], dtype=np.float32)
        self.child_W = np.zeros([n_actions], dtype=np.float32)
        # Save copy of original prior before it gets mutated by dirichlet noise 在狄利克雷噪声引起突变之前保存原始先验的副本
        self.original_prior = np.zeros([n_actions], dtype=np.float32)
        self.child_prior = np.zeros([n_actions], dtype=np.float32)
        self.children = {}

    @property
    def N(self):
        """
        Returns the current visit count of the node.
        返回该节点的当前访问计数。
        """
        return self.parent.child_N[self.action]

    @N.setter 
    def N(self, value):
        self.parent.child_N[self.action] = value

    @property
    def W(self):
        """
        Returns the current total value of the node.
        返回节点的当前总价值。
        """
        return self.parent.child_W[self.action]

    @W.setter
    def W(self, value):
        self.parent.child_W[self.action] = value

    @property
    def Q(self):
        """
        Returns the current action value of the node.
        返回节点的当前动作价值。
        """
        return self.W / (1 + self.N)

    @property
    def child_Q(self):
        return self.child_W / (1 + self.child_N)

    @property
    def child_U(self):
        return (c_PUCT * math.sqrt(1 + self.N) *
                self.child_prior / (1 + self.child_N))

    @property
    def child_action_score(self):
        """
        Action_Score(s, a) = Q(s, a) + U(s, a) as in paper. A high value
        means the node should be traversed.
        Action_Score(s, a) = Q(s, a) + U(s, a)。高值表示应该遍历该节点。
        """
        return self.child_Q + self.child_U

    def select_leaf(self):
        """
        Traverses the MCT rooted in the current node until it finds a leaf
        (i.e. a node that only exists in its parent node in terms of its
        child_N and child_W values but not as a dedicated node in the parent's
        children-mapping). Nodes are selected according to child_action_score.
        It expands the leaf by adding a dedicated MCTSNode. Note that the
        estimated value and prior probabilities still have to be set with
        `incorporate_estimates` afterwards.
        :return: Expanded leaf MCTSNode.
        遍历植根于当前节点的MCT,直到找到一个叶子节点(例如一个节点只存在于它的父节点child_N和child_W值但不是父节点中的专用节点男孩标记)。
        根据child_action_score选择节点。它通过添加专用的MCTSNode来扩展叶子。注意估计值和先验概率仍然要用“incorporate_estimates”之后。
        :返回:展开叶MCTSNode。
        """
        current = self
        while True:
            current.N += 1
            # Encountered leaf node (i.e. node that is not yet expanded).选择一个未被访问的叶子结点
            if not current.is_expanded:
                break
            # Choose action with highest score.选择最高得分的节点
            best_move = np.argmax(current.child_action_score)
            current = current.maybe_add_child(best_move)
        return current

    def maybe_add_child(self, action):
        """
        Adds a child node for the given action if it does not yet exists, and
        returns it.
        :param action: Action to take in current state which leads to desired child node.
        :return: Child MCTSNode.
        如果给定操作还不存在，则为该操作添加子节点，并返回它。
        :param action:在当前状态下执行的操作，将该操作将指向所需的子节点。
        :return:子MCTSNode。
        """
        if action not in self.children:
            # Obtain state following given action.
            new_state = self.TreeEnv.next_state(self.state, action)
            self.children[action] = MCTSNode(new_state, self.n_actions,
                                             self.TreeEnv,
                                             action=action, parent=self)
        return self.children[action]

    def add_virtual_loss(self, up_to):
        """
        Propagate a virtual loss up to a given node.
        :param up_to: The node to propagate until.
        将虚拟损失传播到给定节点。
        :param up to:传播到的节点。
        """
        self.n_vlosses += 1
        self.W -= 1
        if self.parent is None or self is up_to:
            return
        self.parent.add_virtual_loss(up_to)

    def revert_virtual_loss(self, up_to):
        """
        Undo adding virtual loss.撤销
        :param up_to: The node to propagate until.
        """
        self.n_vlosses -= 1
        self.W += 1
        if self.parent is None or self is up_to:
            return
        self.parent.revert_virtual_loss(up_to)

    def revert_visits(self, up_to):
        """
        Revert visit increments.
        Sometimes, repeated calls to select_leaf return the same node.
        This is rare and we're okay with the wasted computation to evaluate
        the position multiple times by the dual_net. But select_leaf has the
        side effect of incrementing visit counts. Since we want the value to
        only count once for the repeatedly selected node, we also have to
        revert the incremented visit counts.
        :param up_to: The node to propagate until.
        """
        self.N -= 1
        if self.parent is None or self is up_to:
            return
        self.parent.revert_visits(up_to)

    def incorporate_estimates(self, action_probs, value, up_to):
        """
        如果节点刚刚通过' select_leaf '展开，则调用该函数，以合并神经网络估计的先前动作概率和状态值。
        :param action_probs:神经网络预测的当前节点状态的动作概率。
        :param value:神经网络预测的当前节点状态值。
        :param up_to:要传播到的节点。
        """
        # A done node (i.e. episode end) should not go through this code path.
        # Rather it should directly call `backup_value` on the final node.
        # TODO: Add assert here
        # Another thread already expanded this node in the meantime.
        # Ignore wasted computation but correct visit counts.
        
        if self.is_expanded:
            self.revert_visits(up_to=up_to)
            return
        self.is_expanded = True
        self.original_prior = self.child_prior = action_probs
        # This is a deviation from the paper that led to better results in
        # practice (following the MiniGo implementation).
        self.child_W = np.ones([self.n_actions], dtype=np.float32) * value
        self.backup_value(value, up_to=up_to)

    def backup_value(self, value, up_to):
        """
        Propagates a value estimation up to the root node.
        :param value: Value estimate to be propagated.
        :param up_to: The node to propagate until.
        """
        self.W += value
        if self.parent is None or self is up_to:
            return
        self.parent.backup_value(value, up_to)

    def is_done(self):
        return self.TreeEnv.is_done_state(self.state, self.depth)

    def inject_noise(self):
        dirch = np.random.dirichlet([D_NOISE_ALPHA] * self.n_actions)
        self.child_prior = self.child_prior * 0.75 + dirch * 0.25

    def visits_as_probs(self, squash=False):
        """
        Returns the child visit counts as a probability distribution.
        :param squash: If True, exponentiate the probabilities by a temperature
        slightly large than 1 to encourage diversity in early steps.
        :return: Numpy array of shape (n_actions).
        """
        probs = self.child_N
        if squash:
            probs = probs ** .95
        return probs / np.sum(probs)

    def print_tree(self, level=0):
        node_string = "\033[94m|" + "----"*level
        node_string += "Node: action={}\033[0m".format(self.action)
        node_string += "\n• state:\n{}".format(self.state)
        node_string += "\n• N={}".format(self.N)
        node_string += "\n• score:\n{}".format(self.child_action_score)
        node_string += "\n• Q:\n{}".format(self.child_Q)
        node_string += "\n• P:\n{}".format(self.child_prior)
        print(node_string)
        for _, child in sorted(self.children.items()):
            child.print_tree(level+1)


class MCTS:
    """
    Represents a Monte-Carlo search tree and provides methods for performing
    the tree search.
    """

    def __init__(self, agent_netw, TreeEnv, seconds_per_move=None,
                 simulations_per_move=800, num_parallel=8):
        """
        :param agent_netw: Network for predicting action probabilities and
        state value estimate.
        :param TreeEnv: Static class that defines the environment dynamics,
        e.g. which state follows from another state when performing an action.
        :param seconds_per_move: Currently unused.
        :param simulations_per_move: Number of traversals through the tree
        before performing a step.
        :param num_parallel: Number of leaf nodes to collect before evaluating
        them in conjunction.

        :param agent_netw:预测动作概率和状态值估计的网络。
        定义环境动态的静态类，例如:当执行一个操作时，哪个状态是另一个状态的后续状态。
        :param seconds_per_move:当前未使用。
        :param simulations_per_move:在执行一个步骤之前遍历树的次数。
        :param num_parallel:在联合计算叶节点之前要收集的叶节点数量。
        """
        self.agent_netw = agent_netw
        self.TreeEnv = TreeEnv
        self.seconds_per_move = seconds_per_move
        self.simulations_per_move = simulations_per_move
        self.num_parallel = num_parallel
        self.temp_threshold = None        # Overwritten in initialize_search

        self.qs = []
        self.rewards = []
        self.searches_pi = []
        self.obs = []

        self.root = None

    def initialize_search(self, state=None):
        init_state = self.TreeEnv.initial_state()
        n_actions = self.TreeEnv.n_actions
        self.root = MCTSNode(init_state, n_actions, self.TreeEnv)
        # Number of steps into the episode after which we always select the
        # action with highest action probability rather than selecting randomly
        #进入情节的步数，在此之后我们总是选择具有最高行动概率的行动，而不是随机选择
        self.temp_threshold = TEMP_THRESHOLD

        self.qs = []
        self.rewards = []
        self.searches_pi = []
        self.obs = []

    def tree_search(self, num_parallel=None):
        """
        Performs multiple simulations in the tree (following trajectories)
        until a given amount of leaves to expand have been encountered.
        Then it expands and evalutes these leaf nodes.
        :param num_parallel: Number of leaf states which the agent network can
        evaluate at once. Limits the number of simulations.
        :return: The leaf nodes which were expanded.
        在树中执行多次模拟(遵循轨迹)，直到遇到给定数量的要展开的叶。
        然后它展开并计算这些叶节点。
        :param num_parallel:代理网络一次可以计算的叶状态数。限制模拟次数。
        :return:展开的叶节点。
        """
        if num_parallel is None:
            num_parallel = self.num_parallel
        leaves = []
        # Failsafe for when we encounter almost only done-states which would故障保险，当我们遇到几乎只有完成状态，将防止循环永远结束。
        # prevent the loop from ever ending.
        failsafe = 0
        while len(leaves) < num_parallel and failsafe < num_parallel * 2:
            failsafe += 1
            # self.root.print_tree()
            # print("_"*50)
            leaf = self.root.select_leaf()
            # If we encounter done-state, we do not need the agent network to
            # bootstrap. We can backup the value right away.
            if leaf.is_done():
                value = self.TreeEnv.get_return(leaf.state, leaf.depth)
                leaf.backup_value(value, up_to=self.root)
                continue
            # Otherwise, discourage other threads to take the same trajectory
            # via virtual loss and enqueue the leaf for evaluation by agent
            # network.
            leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)
        # Evaluate the leaf-states all at once and backup the value estimates.
        if leaves:
            action_probs, values = self.agent_netw.step(
                self.TreeEnv.get_obs_for_states([leaf.state for leaf in leaves]))
            for leaf, action_prob, value in zip(leaves, action_probs, values):
                leaf.revert_virtual_loss(up_to=self.root)
                leaf.incorporate_estimates(action_prob, value, up_to=self.root)
        return leaves

    def pick_action(self):
        """
        Selects an action for the root state based on the visit counts.根据访问计数为根状态选择一个操作。
        """
        if self.root.depth > self.temp_threshold:
            action = np.argmax(self.root.child_N)
        else:
            cdf = self.root.child_N.cumsum()
            cdf /= cdf[-1]
            selection = rd.random()
            action = cdf.searchsorted(selection)
            assert self.root.child_N[action] != 0
        return action

    def take_action(self, action):
        """
        Takes the specified action for the root state. The subsequent child
        state becomes the new root state of the tree.
        :param action: Action to take for the root state.
        为根状态执行指定的操作。随后的子状态成为树的新根状态。
        :param action:根状态所采取的动作。
        """
        # Store data to be used as experience tuples.
        ob = self.TreeEnv.get_obs_for_states([self.root.state])
        self.obs.append(ob)
        self.searches_pi.append(
            self.root.visits_as_probs()) # TODO: Use self.root.position.n < self.temp_threshold as argument
        self.qs.append(self.root.Q)
        reward = (self.TreeEnv.get_return(self.root.children[action].state,
                                          self.root.children[action].depth)
                  - sum(self.rewards))
        self.rewards.append(reward)

        # Resulting state becomes new root of the tree.
        self.root = self.root.maybe_add_child(action)
        del self.root.parent.children


def execute_episode(agent_netw, num_simulations, TreeEnv):
    """
    Executes a single episode of the task using Monte-Carlo tree search with
    the given agent network. It returns the experience tuples collected during
    the search.
    :param agent_netw: Network for predicting action probabilities and state
    value estimate.
    :param num_simulations: Number of simulations (traverses from root to leaf)
    per action.
    :param TreeEnv: Static environment that describes the environment dynamics.
    :return: The observations for each step of the episode, the policy outputs
    as output by the MCTS (not the pure neural network outputs), the individual
    rewards in each step, total return for this episode and the final state of
    this episode.

    使用给定代理网络的蒙特卡罗树搜索执行任务的单个插曲。它返回搜索过程中收集到的经验元组。
    :param agent_netw:预测动作概率和状态值估计的网络。
    :param num_simulations:每个动作的模拟次数(从根到叶的遍历)。
    :param TreeEnv:描述环境动态的静态环境。
    :return:该集每一步的观察结果.MCTS输出的策略输出(不是纯神经网络输出)，每一步的个人奖励，该集的总回报和该集的最终状态。
    """
    mcts = MCTS(agent_netw, TreeEnv)

    mcts.initialize_search()

    # Must run this once at the start, so that noise injection actually affects
    # the first action of the episode.
    first_node = mcts.root.select_leaf()
    probs, vals = agent_netw.step(
        TreeEnv.get_obs_for_states([first_node.state]))
    first_node.incorporate_estimates(probs[0], vals[0], first_node)

    while True:
        mcts.root.inject_noise()
        current_simulations = mcts.root.N

        # We want `num_simulations` simulations per action not counting我们想要' num_simulations '每个操作的模拟，而不是从以前的操作计算模拟。
        # simulations from previous actions.
        while mcts.root.N < current_simulations + num_simulations:
            mcts.tree_search()

        # mcts.root.print_tree()
        # print("_"*100)

        action = mcts.pick_action()
        mcts.take_action(action)

        if mcts.root.is_done():
            break

    # Computes the returns at each step from the list of rewards obtained at
    # each step. The return is the sum of rewards obtained *after* the step.
    ret = [TreeEnv.get_return(mcts.root.state, mcts.root.depth) for _
           in range(len(mcts.rewards))]

    total_rew = np.sum(mcts.rewards)

    obs = np.concatenate(mcts.obs)
    return (obs, mcts.searches_pi, ret, total_rew, mcts.root.state)

def episode_pureMCTS(num_simulations, TreeEnv):
    """
    使用纯MCTS给定代理网络的蒙特卡罗树搜索执行任务的单个回合。它返回搜索过程中收集到的经验元组。
    :param num_simulations:每个动作的模拟次数(从根到叶的遍历)。
    :param TreeEnv:描述环境动态的静态环境。
    :return:该集每一步的观察结果.MCTS输出的策略输出(不是纯神经网络输出)，每一步的个人奖励，该集的总回报和该集的最终状态。
    """
    mcts = MCTS(None, TreeEnv)

    mcts.initialize_search()

    # Must run this once at the start, so that noise injection actually affects
    # the first action of the episode.
    first_node = mcts.root.select_leaf()
    probs, vals = agent_netw.step(
        TreeEnv.get_obs_for_states([first_node.state]))
    first_node.incorporate_estimates(probs[0], vals[0], first_node)

    while True:
        mcts.root.inject_noise()
        current_simulations = mcts.root.N

        # We want `num_simulations` simulations per action not counting我们想要' num_simulations '每个操作的模拟，而不是从以前的操作计算模拟。
        # simulations from previous actions.
        while mcts.root.N < current_simulations + num_simulations:
            mcts.tree_search()

        action = mcts.pick_action()
        mcts.take_action(action)

        if mcts.root.is_done():
            break

    # Computes the returns at each step from the list of rewards obtained at
    # each step. The return is the sum of rewards obtained *after* the step.
    ret = [TreeEnv.get_return(mcts.root.state, mcts.root.depth) for _
           in range(len(mcts.rewards))]

    total_rew = np.sum(mcts.rewards)

    obs = np.concatenate(mcts.obs)
    return (obs, mcts.searches_pi, ret, total_rew, mcts.root.state)