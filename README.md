# Game theoretic planning based on MCTS for autonomous driving

This is a code repository based on MTCS to achieve autonomous driving planning

policynet
输入：当前状态与观测
输入：当前状态节点的子节点的概率与当前节点的预估奖励





网络的输入：多智能体状态信息

单个智能体观测信息定义如下:（速度（float），位置（3-d float），上一步动作，指令路径点（2 point））



网络结构：Transformer架构

输入为不定长的智能体观测向量，依次输出各智能体状态的预估价值和下一状态的转移概率



MCTS根据斯坦克伯格博弈或者优势分解定理（参考MAT论文），可以依次进行决策，随机排序进行决策。



