import numpy as np

"""
before prediction,we assume that the number of people
whose dirving style in [0,1] is averaged,so the starting
driving style is 0.5 ,for the prior prob

推理输入为状态向量：己方速度，己方驾驶风格参数，最近智能体速度，最近智能体驾驶参数，距离，方位角

"""
class Bayes():
    def __init__(self,n_candidate = 9):
        self.n_candidate = n_candidate
        self.candidate = np.arange(0.1, 1, 0.9 / self.n_candidate)
        self.weight = np.ones(self.n_candidate) / self.n_candidate
        self.p_update = np.ones(self.n_candidate)
        self.reward = np.ones(self.n_candidate)
        self.gama = 0.5
        
    #归一化
    def Normalize(self):
        weight_sum = sum(self.weight)
        self.weight = self.weight / weight_sum
        return self.weight
    
    #通过网络获得奖励
    def update_reward(self, reward_order):
        reward = 1
        self.reward[reward_order] = reward
        return self.reward

    #更新类条件概率
    def update_prob(self):
        self.p_update = np.exp(self.reward) / np.sum(np.exp(self.reward))
        return self.p_update

    #计算驾驶风格
    def compute(self):
        self.gama = self.weight @ self.candidate
        return self.gama
    
    #推理 需外部调用
    def inference(self):
        for i in range(0, self.n_candidate):
            self.update_reward(i)
        self.update_prob()
        self.weight = self.weight * self.p_update
        self.Normalize()
        self.compute()
        return self.gama

if __name__ == '__main__':
    a = Bayes()
    a.inference()
    
    print(a.weight)
    print(a.gama)