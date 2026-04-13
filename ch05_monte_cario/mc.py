"""
蒙特卡洛控制算法实现
理论依据：Sutton & Barto 《强化学习》第 5.4 章 同策略首次访问 $\epsilon$-greedy MC 控制
适配环境：GridWorld (西湖大学智能无人系统实验室)
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import random
from collections import defaultdict
from grid_world import GridWorld
from arguments import args

class MCAgent:
    def __init__(self, env: GridWorld, gamma: float = 0.9, 
                 alpha: float = 0.1, epsilon: float = 0.2):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha  # 采用常数步长，在实践中比样本均值更稳定
        self.epsilon = epsilon

        # 强制将环境的动作空间统一为 tuple，作为字典的可靠键
        self.action_space = [tuple(a) for a in self.env.action_space]
        
        # Q(s, a) 表：嵌套 defaultdict 避免 KeyError
        self.Q = defaultdict(lambda: defaultdict(float))
        
        # 防止死循环的硬性限制（MC 算法的生命线）
        self.max_steps_per_episode = 500 

    def choose_action(self, state: tuple) -> tuple:
        """
        行为策略：$\epsilon$-greedy
        【理论修正】直接基于 Q 表计算，严格处理平局
        """
        if state == self.env.target_state:
            return None

        # 以 epsilon 的概率随机探索
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        
        # 以 1-epsilon 的概率贪婪利用
        q_values = [self.Q[state][a] for a in self.action_space]
        max_q = max(q_values)
        
        # 【理论关键】找到所有等于最大值的动作，随机打破平局
        best_actions = [a for a, q in zip(self.action_space, q_values) if np.isclose(q, max_q)]
        return random.choice(best_actions)

    def generate_episode(self) -> list:
        """
        生成一条完整的轨迹
        返回格式：[(S_0, A_0, R_1), (S_1, A_1, R_2), ..., (S_T-1, A_T-1, R_T)]
        """
        episode = []
        state, _ = self.env.reset()
        
        for _ in range(self.max_steps_per_episode):
            action = self.choose_action(state)
            if action is None: break # 终止状态
                
            next_state, reward, done, _ = self.env.step(action) # 适配底层环境输入
            episode.append((state, tuple(action), reward)) # 存储时强制转为 tuple
            
            if done:
                break
            state = next_state
            
        return episode

    def mc_control(self, n_episodes: int):
        """
        同策略首次访问 MC 控制
        """
        print(f"开始训练 MC 控制: {n_episodes} episodes...")
        for ep in range(1, n_episodes + 1):
            episode = self.generate_episode()
            G = 0
            visited = set() # 记录本回合出现过的 (S, A) 对，保证“首次访问”

            # 逆序遍历计算回报 G_t
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = reward + self.gamma * G
                sa_pair = (state, action)

                # 首次访问时更新 Q(s,a)
                if sa_pair not in visited:
                    visited.add(sa_pair)
                    # 增量式常数步长更新：Q(S,A) <- Q(S,A) + alpha * (G - Q(S,A))
                    self.Q[state][action] += self.alpha * (G - self.Q[state][action])
            
            # 可选：随着训练进行，衰减 epsilon 满足 GLIE 条件 (Greedy in the Limit with Infinite Exploration)
            # 这里采用简单的线性衰减，下限保底 0.01
            self.epsilon = max(0.01, self.epsilon - (0.2 / n_episodes) * 2)

            if ep % 1000 == 0:
                print(f"Episode {ep}/{n_episodes} 完成, 当前 $\epsilon$={self.epsilon:.3f}")

        print("训练结束！")

    def get_policy_matrix(self) -> np.ndarray:
        """
        将 Q 表转换为 GridWorld 所需的 policy_matrix (用于画箭头)
        """
        env_size = self.env.env_size
        num_states = env_size[0] * env_size[1]
        action_num = len(self.action_space)
        policy_matrix = np.zeros((num_states, action_num))

        # 严格对齐 GridWorld.add_policy 中的索引映射规则
        for y in range(env_size[1]):
            for x in range(env_size[0]):
                state_idx = y * env_size[0] + x
                state = (x, y)
                
                if state == self.env.target_state:
                    continue
                
                # 获取当前状态所有动作的 Q 值
                q_values = [self.Q[state][a] for a in self.action_space]
                max_q = max(q_values) if q_values else 0
                
                # 计算贪婪策略的概率分布（这里为了画图清晰，画纯贪婪策略）
                for i, a in enumerate(self.action_space):
                    if np.isclose(self.Q[state][a], max_q):
                        policy_matrix[state_idx, i] = 1.0
                    else:
                        policy_matrix[state_idx, i] = 0.0
                        
        return policy_matrix

    def get_state_value_list(self) -> list:
        """
        将 Q 表推导出 V 表，转换为 GridWorld 所需的列表格式 (用于画数值)
        V(s) = max_a Q(s, a)
        """
        env_size = self.env.env_size
        value_list = []
        for y in range(env_size[1]):
            for x in range(env_size[0]):
                state = (x, y)
                if state == self.env.target_state:
                    value_list.append(0.0) # 终点价值通常设为 0 或保持原样
                else:
                    q_values = [self.Q[state][a] for a in self.action_space]
                    value_list.append(max(q_values) if q_values else 0.0)
        return value_list


if __name__ == "__main__":
    # 1. 初始化环境
    # 确保关闭 debug 模式，否则 render 时会卡住等待输入
    args.debug = False 
    env = GridWorld()

    # 2. 初始化智能体
    agent = MCAgent(
        env=env,
        gamma=0.9,      # 折扣因子
        alpha=0.05,     # 学习率（常数步长，不宜过大）
        epsilon=0.6     # 初始探索率
    )

    # 3. 执行同策略首次访问 MC 控制
    agent.mc_control(n_episodes=50000)

    # 4. 可视化结果
    print("正在渲染最终策略与状态价值...")
    env.render(animation_interval=2) # 先画出底图
    
    # 叠加状态价值 (数值)
    value_list = agent.get_state_value_list()
    env.add_state_values(value_list, precision=1)
    
    # 叠加策略箭头 (只画贪婪动作)
    policy_matrix = agent.get_policy_matrix()
    env.add_policy(policy_matrix)

    # 阻塞窗口，方便观察
    input("按 Enter 键退出...")