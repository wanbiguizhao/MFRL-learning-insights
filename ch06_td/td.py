__credits__ = ["Intelligent Unmanned Systems Laboratory at Westlake University."]
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from grid_world import GridWorld
from arguments import args
import matplotlib.pyplot as plt

class TDLearning:
    def __init__(self, env: GridWorld, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        TD学习基类
        :param env: 网格世界环境实例
        :param alpha: 学习率
        :param gamma: 折扣因子
        :param epsilon: ε-贪心策略参数
        """
        self.env = env
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # ε-贪心

        # 状态值函数 V(s) 初始化（TD(0)用）
        self.V = np.zeros(self.env.num_states)
        # Q值函数 Q(s,a) 初始化（SARSA/Q-Learning用）
        self.Q = np.zeros((self.env.num_states, len(self.env.action_space)))

        # 状态索引映射（将(x,y)转换为一维索引）
        self.state_to_idx = lambda s: s[0] + s[1] * self.env.env_size[0]

    def epsilon_greedy(self, state_idx):
        """ε-贪心策略选择动作"""
        if np.random.uniform(0, 1) < self.epsilon:
            # 随机选择动作（探索）
            return np.random.choice(len(self.env.action_space))
        else:
            # 贪心选择最优动作（利用）
            return np.argmax(self.Q[state_idx])

    def td0_update(self, state, reward, next_state):
        """TD(0) 状态值函数更新"""
        s_idx = self.state_to_idx(state)
        s_next_idx = self.state_to_idx(next_state)
        # TD(0) 更新公式: V(s) ← V(s) + α[r + γV(s') - V(s)]
        self.V[s_idx] += self.alpha * (reward + self.gamma * self.V[s_next_idx] - self.V[s_idx])

    def sarsa_update(self, state, action, reward, next_state, next_action, done):
        """SARSA（在线TD）Q值更新"""
        s_idx = self.state_to_idx(state)
        s_next_idx = self.state_to_idx(next_state)
        
        if done:
            target = reward  # 终止状态无后续奖励
        else:
            # SARSA 更新公式: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
            target = reward + self.gamma * self.Q[s_next_idx, next_action]
        
        self.Q[s_idx, action] += self.alpha * (target - self.Q[s_idx, action])

    def q_learning_update(self, state, action, reward, next_state, done):
        """Q-Learning（离线TD）Q值更新"""
        s_idx = self.state_to_idx(state)
        s_next_idx = self.state_to_idx(next_state)
        
        if done:
            target = reward
        else:
            # Q-Learning 更新公式: Q(s,a) ← Q(s,a) + α[r + γmax_a'Q(s',a') - Q(s,a)]
            target = reward + self.gamma * np.max(self.Q[s_next_idx])
        
        self.Q[s_idx, action] += self.alpha * (target - self.Q[s_idx, action])

    def train_td0(self, episodes=1000):
        """训练TD(0)算法"""
        rewards_history = []
        for ep in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                # TD(0) 需随机选择动作（无策略优化，仅值函数估计）
                action_idx = np.random.choice(len(self.env.action_space))
                action = self.env.action_space[action_idx]
                
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                # TD(0) 更新
                self.td0_update(state, reward, next_state)
                
                state = next_state
                # 渲染（可选，控制频率避免卡顿）
                if args.debug and ep % 100 == 0:
                    self.env.render()

            rewards_history.append(total_reward)
            if (ep + 1) % 100 == 0:
                print(f"TD(0) 第{ep+1}轮, 总奖励: {total_reward:.2f}")
        
        return rewards_history

    def train_sarsa(self, episodes=1000):
        """训练SARSA算法"""
        rewards_history = []
        for ep in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            # 初始动作
            s_idx = self.state_to_idx(state)
            action_idx = self.epsilon_greedy(s_idx)
            action = self.env.action_space[action_idx]

            while not done:
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                # 下一个动作
                s_next_idx = self.state_to_idx(next_state)
                next_action_idx = self.epsilon_greedy(s_next_idx)
                next_action = self.env.action_space[next_action_idx]

                # SARSA 更新
                self.sarsa_update(state, action_idx, reward, next_state, next_action_idx, done)

                # 迭代状态和动作
                state = next_state
                action_idx = next_action_idx
                action = next_action

                # 渲染（可选）
                if args.debug and ep % 100 == 0:
                    self.env.render()

            rewards_history.append(total_reward)
            if (ep + 1) % 100 == 0:
                print(f"SARSA 第{ep+1}轮, 总奖励: {total_reward:.2f}")
        
        return rewards_history

    def train_q_learning(self, episodes=1000):
        """训练Q-Learning算法"""
        rewards_history = []
        for ep in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                # 选择动作
                s_idx = self.state_to_idx(state)
                action_idx = self.epsilon_greedy(s_idx)
                action = self.env.action_space[action_idx]

                # 执行动作
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                # Q-Learning 更新
                self.q_learning_update(state, action_idx, reward, next_state, done)

                # 迭代状态
                state = next_state

                # 渲染（可选）
                if args.debug and ep % 100 == 0:
                    self.env.render()

            rewards_history.append(total_reward)
            if (ep + 1) % 100 == 0:
                print(f"Q-Learning 第{ep+1}轮, 总奖励: {total_reward:.2f}")
        
        return rewards_history

    def get_policy_matrix(self):
        """从Q值生成策略矩阵（用于可视化）"""
        policy_matrix = np.zeros((self.env.num_states, len(self.env.action_space)))
        for s_idx in range(self.env.num_states):
            best_action = np.argmax(self.Q[s_idx])
            policy_matrix[s_idx, best_action] = 1.0  # 最优动作概率为1
        return policy_matrix

    def plot_rewards(self, rewards, title):
        """绘制奖励曲线"""
        plt.figure(figsize=(10, 6))
        # 滑动平均平滑曲线
        window_size = 10
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_rewards)
        plt.xlabel("Episodes (smoothed)")
        plt.ylabel("Total Reward per Episode")
        plt.title(title)
        plt.grid(True)
        plt.show()

    def visualize_results(self):
        """可视化值函数和策略"""
        # 绘制状态值函数
        self.env.render()
        self.env.add_state_values(self.V)
        # 绘制最优策略
        policy_matrix = self.get_policy_matrix()
        self.env.add_policy(policy_matrix)
        self.env.render()
        #plt.show(block=True)


# 主函数：测试TD算法
if __name__ == "__main__":
    # 初始化环境（兼容config.yaml/命令行参数）
    env = GridWorld()

    # 初始化TD学习器
    td_agent = TDLearning(
        env=env,
        alpha=0.1,    # 可通过config.yaml扩展配置
        gamma=0.9,    # 可通过config.yaml扩展配置
        epsilon=0.1   # 可通过config.yaml扩展配置
    )

    # 选择训练的TD算法（三选一）
    # 1. 训练TD(0)
    # rewards_td0 = td_agent.train_td0(episodes=2000)
    # td_agent.plot_rewards(rewards_td0, "TD(0) Reward History")

    # 2. 训练SARSA
    # rewards_sarsa = td_agent.train_sarsa(episodes=2000)
    # td_agent.plot_rewards(rewards_sarsa, "SARSA Reward History")

    # 3. 训练Q-Learning
    rewards_q = td_agent.train_q_learning(episodes=2000)
    #td_agent.plot_rewards(rewards_q, "Q-Learning Reward History")

    # 可视化最终结果（值函数+策略）
    td_agent.visualize_results()