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
from abc import ABC, abstractmethod
from collections import defaultdict
from grid_world import GridWorld
from arguments import args

# ==========================================
# 1. 策略模式抽象层
# ==========================================
class ActionStrategy(ABC):
    """动作选择策略的抽象基类"""
    @abstractmethod
    def select_action(self, state: tuple, action_space: list, q_table: dict, 
                      target_state: tuple = None, **kwargs) -> tuple:
        pass

class GreedyStrategy(ActionStrategy):
    """纯贪心策略：直接选最大法"""
    def select_action(self, state: tuple, action_space: list, q_table: dict, 
                      target_state: tuple = None, **kwargs) -> tuple:
        if state == target_state:
            return None
        q_values = [q_table[state][a] for a in action_space]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(action_space, q_values) if np.isclose(q, max_q)]
        return random.choice(best_actions)

class EpsilonGreedyStrategyMath(ActionStrategy):
    """
    数学严谨版 (终极纯软分布形态)
    【特点】一次性构建符合书本公式的全局概率质量函数，统一归一化采样。
    """
    def __init__(self, epsilon: float = 0.1):
        self.default_epsilon = epsilon

    def select_action(self, state: tuple, action_space: list, q_table: dict, 
                      target_state: tuple = None, epsilon: float = None, **kwargs) -> tuple:
        current_epsilon = epsilon if epsilon is not None else self.default_epsilon
        if state == target_state:
            return None
            
        action_num = len(action_space)
        q_values = [q_table[state][a] for a in action_space]
        max_q = max(q_values)
        
        # 1. 初始化：所有动作无条件平分 epsilon 的底座概率
        probs = np.full(action_num, current_epsilon / action_num)
        
        # 2. 找出所有达到最大值的动作索引 (处理平局)
        best_indices = np.where(np.isclose(q_values, max_q))[0]
        
        # 3. 将 (1 - epsilon) 的概率严格按照书本公式叠加给最优动作
        probs[best_indices] += (1.0 - current_epsilon) / len(best_indices)
        
        # 4. 【理论护城河 & 消灭 Bug】绝对归一化
        # 无论 eps 是多少，无论怎么平分，这一行保证总和铁定等于 1.0
        probs = probs / np.sum(probs)
        
        # 5. 采样索引并映射回动作 (绝对不会报错的做法)
        chosen_idx = np.random.choice(action_num, p=probs)
        return action_space[chosen_idx]

class EpsilonGreedyStrategyEng(ActionStrategy):
    """
    工程实用版 $\epsilon$-greedy 策略
    【特点】利用 if/else 提前截断无效计算路径。在保证数学期望完全等同于书本公式的前提下，追求极致性能。
    """
    def __init__(self, epsilon: float = 0.1):
        self.default_epsilon = epsilon

    def select_action(self, state: tuple, action_space: list, q_table: dict, 
                      target_state: tuple = None, epsilon: float = None, **kwargs) -> tuple:
        current_epsilon = epsilon if epsilon is not None else self.default_epsilon
        if state == target_state:
            return None
            
        # 【工程优化】以极小的计算代价直接处理 epsilon 的探索部分
        if random.random() < current_epsilon:
            return random.choice(action_space)
            
        # 剩余 (1 - epsilon) 的概率在此处执行
        q_values = [q_table[state][a] for a in action_space]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(action_space, q_values) if np.isclose(q, max_q)]
        
        # 在最优动作中均匀采样 (因为在条件概率下，它们的概率已经被放大)
        return random.choice(best_actions)


class MCAgent:
    def __init__(self, env: GridWorld, gamma: float = 0.9, 
                 alpha: float = 0.1, strategy: ActionStrategy = None):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha 
        # 默认使用工程版，兼顾速度与正确性
        self.strategy = strategy if strategy is not None else EpsilonGreedyStrategyEng()
        
        # 【类型安全】强制将环境的动作空间统一为 tuple
        self.action_space = [tuple(a) for a in self.env.action_space]
        self.Q = defaultdict(lambda: defaultdict(float))
        self.max_steps_per_episode = 500 
        
    def choose_action(self, state: tuple, **kwargs) -> tuple:
        """上下文传递"""
        return self.strategy.select_action(
            state=state,
            action_space=self.action_space,
            q_table=self.Q,
            target_state=self.env.target_state,
            **kwargs 
        )
        

    def generate_episode(self, **kwargs) -> list:
        """生成轨迹"""
        episode = []
        state, _ = self.env.reset()
        
        for _ in range(self.max_steps_per_episode):
            action = self.choose_action(state, **kwargs)
            if action is None: break
                
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, tuple(action), reward)) # 存储强制 tuple
            
            if done:
                break
            state = next_state
            
        return episode

    def mc_control(self, n_episodes: int, epsilon_start: float = 0.3, epsilon_end: float = 0.01):
        """同策略首次访问 MC 控制"""
        print(f"开始训练 (策略: {self.strategy.__class__.__name__})...")
        epsilon = epsilon_start
        
        for ep in range(1, n_episodes + 1):
            episode = self.generate_episode(epsilon=epsilon)
            
            G = 0
            visited = set()

            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = reward + self.gamma * G # 计算真实回报
                sa_pair = (state, action)

                if sa_pair not in visited:
                    visited.add(sa_pair)
                    self.Q[state][action] += self.alpha * (G - self.Q[state][action])
            
            # 数学级平滑线性衰减
            progress = (ep - 1) / max(1, n_episodes - 1)
            epsilon = epsilon_start - (epsilon_start - epsilon_end) * progress

            if ep % 1000 == 0:
                print(f"Episode {ep}/{n_episodes} 完成, $\epsilon$={epsilon:.4f}")
        print("训练结束！")

    def get_policy_matrix(self) -> np.ndarray:
        """将 Q 表转换为纯贪心策略矩阵"""
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
                q_values = [self.Q[state][a] for a in self.action_space]
                max_q = max(q_values) if q_values else 0
                for i, a in enumerate(self.action_space):
                    if np.isclose(self.Q[state][a], max_q):
                        policy_matrix[state_idx, i] = 1.0
        return policy_matrix

    def get_state_value_list(self) -> list:
        """将 Q 表推导出 V 表"""
        env_size = self.env.env_size
        value_list = []
        for y in range(env_size[1]):
            for x in range(env_size[0]):
                state = (x, y)
                if state == self.env.target_state:
                    value_list.append(0.0)
                else:
                    q_values = [self.Q[state][a] for a in self.action_space]
                    value_list.append(max(q_values) if q_values else 0.0)
        return value_list
# ==========================================
# 3. 调用层：三种策略的终极对决
# ==========================================
if __name__ == "__main__":
    args.debug = False 

    # ---------------- 场景 1：数学严谨版训练 ----------------
    print("=== 场景 1: Sutton 书本数学版 $\epsilon$-greedy 训练 ===")
    env_math = GridWorld() 
    agent_math = MCAgent(env=env_math, gamma=0.9, alpha=0.05, strategy=EpsilonGreedyStrategyMath())
    agent_math.mc_control(n_episodes=5000, epsilon_start=0.4)
    
    env_math.render(animation_interval=2)
    env_math.add_policy(agent_math.get_policy_matrix())
    env_math.add_state_values(agent_math.get_state_value_list(), precision=1)
    input("按 Enter 查看下一个场景...")

    # ---------------- 场景 2：工程实用版训练 ----------------
    print("\n=== 场景 2: 工程实用版 $\epsilon$-greedy 训练 ===")
    env_eng = GridWorld() # 新开环境防止画图叠加
    agent_eng = MCAgent(env=env_eng, gamma=0.9, alpha=0.05, strategy=EpsilonGreedyStrategyEng())
    agent_eng.mc_control(n_episodes=5000, epsilon_start=0.4)
    
    env_eng.render(animation_interval=2)
    env_eng.add_policy(agent_eng.get_policy_matrix())
    env_eng.add_state_values(agent_eng.get_state_value_list(), precision=1)
    input("按 Enter 查看下一个场景...")

    # ---------------- 场景 3：纯贪心策略测试 (高光时刻) ----------------
    # 1. 全新的环境
    env_greedy = GridWorld()
    
    # 2. 全新的智能体，使用纯 Greedy 策略
    agent_greedy = MCAgent(env=env_greedy, gamma=0.9, alpha=0.05, strategy=GreedyStrategy())
    
    # 3. 关键：调用 mc_control，但是把 epsilon_start 和 epsilon_end 都死死卡在 0！
    # 注意：因为策略是 GreedyStrategy，它根本不看 epsilon 参数，所以传 0 只是语义上的严谨
    agent_greedy.mc_control(n_episodes=5000, epsilon_start=0.0, epsilon_end=0.0)
    
    # 4. 渲染结果
    env_greedy.render(animation_interval=2)
    env_greedy.add_policy(agent_greedy.get_policy_matrix())
    env_greedy.add_state_values(agent_greedy.get_state_value_list(), precision=1)
    
    print("\n【实验结论预测】：")
    print("由于初始 Q 表全为 0，平局随机打破时，纯贪心极大概率在第一步就")
    print("锁死了一个非最优方向（比如一直往左撞墙），导致它永远无法发现终点，")
    print("Q 表永远不更新，最终画出的箭头是一片混乱，数值全是 0。")
    
    input("按 Enter 退出...")
    
    input("按 Enter 退出...")