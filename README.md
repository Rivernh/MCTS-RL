# Game theoretic planning based on MCTS for autonomous driving

This is a code repository based on MTCS to achieve autonomous driving planning

policynet
输入：当前状态与观测，驾驶风格参数
输入：当前状态节点的子节点的概率与当前节点的预估奖励

训练过程：
1.纯MCTS历程采集数据
2.根据采集数据训练policy网络
    最新的policy和通过率最高的policy
3.利用训练好的policy网络进行测试



网络的输入：

栅格地图：

state：ve,re,vo,ro,d,theta 6



