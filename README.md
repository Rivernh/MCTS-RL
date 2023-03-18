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



每个智能体的观测包括：



编码器的输入为obs观测向量，每个智能体的观测向量等长

观测向量包括：x,y,yaw,speed,last_speed,last_steer,goal_x,goal_y,goal_yaw

编码器的输出经过MLP为当前状态或者观测的价值估计；

解码器的输入为之前所有智能体的转移概率分布和编码器输出的观测的重构值；编码器的输出经过MLP为

智能体的转移概率；

维度规定：

embedding dim 128

obs dim n_obs 16

action dim 25

value dim 1



在线学习

init policy--------->data-------->policy
