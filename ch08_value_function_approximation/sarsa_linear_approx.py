"""
第八章：基于线性函数近似的 Semi-Gradient SARSA
理论依据：Sutton & Barto 第 8.4 节
核心跃迁：从查表 Q(s,a) 转变为梯度下降求 w
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import random
from grid_world import GridWorld
from arguments import args

# ==========================================
# 1. 特征工程层 (Feature Engineering)
# ==========================================
class StateActionFeatureExtractor:
    """
    将 转化为线性特征向量
    设计哲学：使用交叉特征使算法具备泛化能力，而不是 One-Hot 的死记硬背
    """
    def __init__(self, env_size: tuple):
        self.env_size = env_size
        # 特征维度: [x_norm, y_norm, dx, dy, x_norm*dx, y_norm*dy, bias]
        self.feature_dim = 7

    def transform(self, state: tuple, action: tuple) -> np.ndarray:
        x, y = state
        dx, dy = action
        
        # 归一化坐标到 [0, 1]，防止数值过大导致梯度爆炸
        norm_x = x / (self.env_size[0] - 1) if self.env_size[0] > 1 else 0
        norm_y = y / (self.env_size[1] - 1) if self.env_size[1] > 1 else 0
        
        # 构建特征向量
        features = np.array([
            norm_x,
            norm_y,
            dx,
            dy,
            norm_x * dx,  # 交叉特征：x 越大，向右(dx=1)的影响越大
            norm_y * dy,  # 交叉特征：y 越大，向下(dy=1)的影响越大
            1.0           # 偏置项，类似线性回归中的截距
        ])
        return features

# ==========================================
# 2. 函数近似器层
# ==========================================
class LinearQFunction:
    """线性 Q 值函数：q(s,a) = x^T * w"""
    def __init__(self, feature_dim: int):
        # 用极小的随机数初始化权重，打破对称性
        self.w = np.random.randn(feature_dim) * 0.01

    def predict(self, features: np.ndarray) -> float:
        """计算 Q 值的估计值"""
        return np.dot(features, self.w)

    def update(self, features: np.ndarray, target: float, alpha: float):
        """半梯度更新权重 w"""
        current_q = self.predict(features)
        td_error = target - current_q
        # 梯度就是特征本身 x(s,a)
        self.w += alpha * td_error * features

# ==========================================
# 3. SARSA 智能体层
# ==========================================
class SARSAAgent:
    def __init__(self, env: GridWorld, gamma: float = 0.9, 
                 alpha: float = 0.01, epsilon: float = 0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha  # 注意：函数近似中 alpha 必须很小 (0.001 ~ 0.01)，否则会发散
        self.epsilon = epsilon

        self.action_space = [tuple(a) for a in self.env.action_space]
        
        # 实例化特征提取器和 Q 函数
        self.feature_extractor = StateActionFeatureExtractor(self.env.env_size)
        self.q_function = LinearQFunction(self.feature_extractor.feature_dim)
        
        self.max_steps = 500

    def choose_action(self, state: tuple) -> tuple:
        """工程版 Epsilon-Greedy，但价值来源变成了 Q 函数"""
        if state == self.env.target_state:
            return None
            
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
            
        # 【核心变化】不再是查字典，而是让每个动作过一遍 Q 函数，选最大的
        q_values = [self.q_function.predict(self.feature_extractor.transform(state, a)) 
                    for a in self.action_space]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(self.action_space, q_values) if np.isclose(q, max_q)]
        return random.choice(best_actions)

    def learn(self, n_episodes: int):
        """Semi-Gradient SARSA 训练循环"""
        print(f"开始 Semi-Gradient SARSA 训练 (特征维度: {self.feature_extractor.feature_dim})...")
        
        for ep in range(1, n_episodes + 1):
            state, _ = self.env.reset()
            action = self.choose_action(state)
            
            for _ in range(self.max_steps):
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.choose_action(next_state)
                
                # 1. 提取特征
                feat_s_a = self.feature_extractor.transform(state, action)
                
                # 2. 构建 TD Target (Sutton 公式 8.7 的中括号部分)
                if done:
                    td_target = reward
                else:
                    # 注意：这里计算 S'A' 用的还是更新前的旧权重 w
                    q_next = self.q_function.predict(
                        self.feature_extractor.transform(next_state, next_action)
                    )
                    td_target = reward + self.gamma * q_next
                
                # 3. 更新权重 w
                self.q_function.update(feat_s_a, td_target, self.alpha)
                
                if done:
                    break
                    
                state = next_state
                action = next_action

            # Epsilon 衰减
            self.epsilon = max(0.01, self.epsilon - (0.5 / n_episodes))
            
            if ep % 1000 == 0:
                print(f"Episode {ep}/{n_episodes} 完成, $\epsilon$={self.epsilon:.4f}")
                
        print("训练结束！")

    # 为了兼容 GridWorld 的可视化，将 Q 函数推导出的 V 表转成列表
    def get_state_value_list(self) -> list:
        env_size = self.env.env_size
        value_list = []
        for y in range(env_size[1]):
            for x in range(env_size[0]):
                state = (x, y)
                if state == self.env.target_state:
                    value_list.append(0.0)
                else:
                    # V(s) = max_a q(s,a)
                    q_values = [self.q_function.predict(self.feature_extractor.transform(state, a)) 
                                for a in self.action_space]
                    value_list.append(max(q_values))
        return value_list

    def get_policy_matrix(self) -> np.ndarray:
        env_size = self.env.env_size
        num_states = env_size[0] * env_size[1]
        action_num = len(self.action_space)
        policy_matrix = np.zeros((num_states, action_num))

        for y in range(env_size[1]):
            for x in range(env_size[0]):
                state_idx = y * env_size[0] + x
                state = (x, y)
                if state == self.env.target_state:
                    continue
                q_values = [self.q_function.predict(self.feature_extractor.transform(state, a)) 
                            for a in self.action_space]
                max_q = max(q_values)
                for i, a in enumerate(self.action_space):
                    if np.isclose(self.q_function.predict(self.feature_extractor.transform(state, a)), max_q):
                        policy_matrix[state_idx, i] = 1.0
        return policy_matrix
    
if __name__ == "__main__":
    args.debug = False 
    env = GridWorld()

    # 初始化 SARSA 智能体
    # 【注意】alpha 必须非常小！如果大于 0.05，权重 w 会因为步长过大而发散（数值变成 NaN）
    agent = SARSAAgent(env=env, gamma=0.9, alpha=0.005, epsilon=0.5)
    
    # 训练回合数可以设大一点，因为它是单步更新，收敛慢
    agent.learn(n_episodes=10000)

    # 可视化
    print("正在渲染 Semi-Gradient SARSA 结果...")
    env.render(animation_interval=2)
    env.add_policy(agent.get_policy_matrix())
    # 注意精度设为 2，因为线性近似的值通常带小数
    env.add_state_values(agent.get_state_value_list(), precision=2) 
    input("按 Enter 退出...")