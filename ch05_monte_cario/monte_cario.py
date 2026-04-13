from collections import defaultdict
import sys
import os 
# 路径配置（精简写法）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from grid_world import GridWorld  # 导入你的GridWorld类
from arguments import args        # 复用项目的参数配置

class MCAgent:
    """
    适配GridWorld的蒙特卡洛(MC)智能体
    兼容GridWorld的状态(tuple)、动作(list)、奖励规则和终止条件
    支持：首次访问/每访问MC预测（V(s)/Q(s,a)）、ε-贪心MC控制（On-Policy）
    """
    def __init__(self, env: GridWorld, gamma: float = 0.9, 
                 alpha: float = 0.6, epsilon: float = 0.5):
        """
        初始化MC智能体
        :param env: GridWorld实例（已初始化，含action_space/状态规则）
        :param gamma: 折扣因子（默认从args读取）
        :param alpha: 增量更新学习率（None时用样本均值更新）
        :param epsilon: ε-贪心探索率
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        # 状态价值函数 V(s)：键=(x,y)，值=状态价值
        self.V = defaultdict(float)
        # 动作价值函数 Q(s,a)：键1=(x,y)，键2=tuple(action)，值=动作价值
        self.Q = defaultdict(lambda: defaultdict(float))
        # 策略：键=(x,y)，值={tuple(action): 概率}（动作转tuple保证可哈希）
        self.policy = self._init_random_policy()

    def _action2hashable(self, action) -> tuple:
        """将GridWorld的列表型动作转为可哈希的tuple（如[0,1]→(0,1)）"""
        return tuple(action)

    def _init_random_policy(self) -> defaultdict:
        """初始化随机策略：每个非终止状态下所有动作等概率"""
        policy = defaultdict(dict)
        action_space = self.env.action_space
        action_num = len(action_space)
        
        # 遍历GridWorld所有可能状态（基于env_size生成）
        for x in range(self.env.env_size[0]):
            for y in range(self.env.env_size[1]):
                state = (x, y)
                # 终止状态（目标状态）无动作
                if state == self.env.target_state:
                    policy[state] = {}
                    continue
                # 所有动作等概率分配
                for action in action_space:
                    policy[state][self._action2hashable(action)] = 1.0 / action_num
        return policy

    def choose_action(self, state: tuple) -> list | None:
        """
        根据ε-贪心策略选择动作（适配GridWorld的列表型动作）
        :param state: (x,y)型状态
        :return: 列表型动作（如[0,1]），终止状态返回None
        """
        # 终止状态无动作
        if state == self.env.target_state:
            return None
        
        # 提取当前状态的动作概率
        action_probs = self.policy[state]
        actions = [list(act) for act in action_probs.keys()]  # 转回列表型动作
        probs = list(action_probs.values())
        
        # ε-贪心：以1-ε选最优动作，ε随机选
        if np.random.random() < (1 - self.epsilon):
            # 选Q值最大的动作
            q_vals = {self._action2hashable(act): self.Q[state][self._action2hashable(act)] for act in actions}
            best_action = max(q_vals, key=q_vals.get)
            return best_action
        else:
            # 随机选动作
            action_indices = np.arange(len(actions))  # 一维索引：[0,1,2,3]
            selected_idx = np.random.choice(action_indices)  # 选索引（符合一维要求）
            selected_action = actions[selected_idx]  # 映射回原列表型动作
            return selected_action

    def generate_episode(self) -> list:
        """
        生成完整轨迹：[(s0, a0, r1), (s1, a1, r2), ..., (sn-1, an-1, rn)]
        适配GridWorld的reset/step接口
        :return: 轨迹列表，每个元素为((x,y), [action], reward)
        """
        episode = []
        # 重置环境，获取初始状态
        state, _ = self.env.reset()
        done = False

        while not done:
            # 选择动作
            action = self.choose_action(state)
            # 执行动作（GridWorld的step返回(next_state, reward, done, {})）
            next_state, reward, done, _ = self.env.step(action)
            # 存储轨迹：(当前状态, 动作, 即时奖励)
            episode.append((state, action, reward))
            # 更新状态
            state = next_state

        return episode

    def first_visit_mc_prediction(self, n_episodes: int = 1000, evaluate_q: bool = False) -> defaultdict:
        """
        首次访问型MC预测：评估当前策略的V(s)或Q(s,a)
        :param n_episodes: 迭代轨迹数
        :param evaluate_q: True=评估Q(s,a)，False=评估V(s)
        :return: 更新后的V或Q
        """
        returns = defaultdict(list)  # 存储状态/状态-动作对的回报

        for _ in range(n_episodes):
            episode = self.generate_episode()
            G = 0  # 回报累计值
            visited = set()  # 记录首次访问的状态/状态-动作对

            # 逆序遍历轨迹计算回报
            for step in reversed(episode):
                state, action, reward = step
                G = self.gamma * G + reward

                # 构建首次访问的key（状态 或 状态-动作对）
                if evaluate_q:
                    key = (state, self._action2hashable(action))
                else:
                    key = state

                # 首次访问时更新价值
                if key not in visited:
                    visited.add(key)
                    returns[key].append(G)
                    # 增量更新或均值更新
                    if self.alpha is not None:
                        if evaluate_q:
                            self.Q[state][self._action2hashable(action)] += self.alpha * (G - self.Q[state][self._action2hashable(action)])
                        else:
                            self.V[state] += self.alpha * (G - self.V[state])
                    else:
                        if evaluate_q:
                            self.Q[state][self._action2hashable(action)] = np.mean(returns[key])
                        else:
                            self.V[state] = np.mean(returns[key])

        return self.Q if evaluate_q else self.V

    def every_visit_mc_prediction(self, n_episodes: int = 1000, evaluate_q: bool = False) -> defaultdict:
        """
        每访问型MC预测：每次访问状态/状态-动作对都更新价值
        :param n_episodes: 迭代轨迹数
        :param evaluate_q: True=评估Q(s,a)，False=评估V(s)
        :return: 更新后的V或Q
        """
        returns = defaultdict(list)

        for _ in range(n_episodes):
            episode = self.generate_episode()
            G = 0

            # 逆序遍历（每访问都更新，无需visited集合）
            for step in reversed(episode):
                state, action, reward = step
                G = self.gamma * G + reward

                # 构建key
                if evaluate_q:
                    key = (state, self._action2hashable(action))
                else:
                    key = state

                returns[key].append(G)
                # 更新价值
                if self.alpha is not None:
                    if evaluate_q:
                        self.Q[state][self._action2hashable(action)] += self.alpha * (G - self.Q[state][self._action2hashable(action)])
                    else:
                        self.V[state] += self.alpha * (G - self.V[state])
                else:
                    if evaluate_q:
                        self.Q[state][self._action2hashable(action)] = np.mean(returns[key])
                    else:
                        self.V[state] = np.mean(returns[key])

        return self.Q if evaluate_q else self.V

    def mc_control_epsilon_greedy(self, n_episodes: int = 10000) -> defaultdict:
        """
        ε-贪心MC控制（On-Policy）：优化策略以最大化Q(s,a)
        :param n_episodes: 迭代轨迹数
        :return: 优化后的策略
        """
        returns = defaultdict(list)

        for _ in range(n_episodes):
            episode = self.generate_episode()
            G = 0
            visited = set()

            # 逆序计算回报并更新Q
            for step in reversed(episode):
                state, action, reward = step
                G = self.gamma * G + reward
                action_hash = self._action2hashable(action)
                key = (state, action_hash)

                # 首次访问时更新Q并改进策略
                if key not in visited:
                    visited.add(key)
                    returns[key].append(G)
                    # 更新Q(s,a)
                    if self.alpha is not None:
                        self.Q[state][action_hash] += self.alpha * (G - self.Q[state][action_hash])
                    else:
                        self.Q[state][action_hash] = np.mean(returns[key])
                    # 基于新的Q值改进策略（ε-贪心）
                    self._update_policy_epsilon_greedy(state)

        return self.policy

    def _update_policy_epsilon_greedy(self, state: tuple):
        """
        对指定状态更新ε-贪心策略：
        - 最优动作概率 = 1-ε + ε/|A|
        - 其他动作概率 = ε/|A|
        """
        action_space = self.env.action_space
        action_num = len(action_space)
        action_hashes = [self._action2hashable(act) for act in action_space]

        # 终止状态无需更新
        if state == self.env.target_state:
            return

        # 找到当前状态的最优动作（Q值最大）
        q_vals = {act_h: self.Q[state][act_h] for act_h in action_hashes}
        best_action_h = max(q_vals, key=q_vals.get)

        # 初始化所有动作概率为 ε/|A|
        for act_h in action_hashes:
            self.policy[state][act_h] = self.epsilon / action_num
        # 最优动作额外分配 1-ε 的概率
        self.policy[state][best_action_h] += 1 - self.epsilon

    def get_policy_matrix(self) -> np.ndarray:
        """
        将策略转换为GridWorld的policy_matrix格式（适配add_policy方法）
        :return: shape=(num_states, action_num)的概率矩阵
        """
        env_size = self.env.env_size
        num_states = env_size[0] * env_size[1]
        action_num = len(self.env.action_space)
        policy_matrix = np.zeros((num_states, action_num))

        # 映射：状态(x,y) → 一维索引，动作 → 索引
        for x in range(env_size[0]):
            for y in range(env_size[1]):
                state = (x, y)
                state_idx = y * env_size[0] + x  # 匹配GridWorld的add_policy索引规则
                # 遍历所有动作
                for act_idx, action in enumerate(self.env.action_space):
                    act_h = self._action2hashable(action)
                    policy_matrix[state_idx, act_idx] = self.policy[state].get(act_h, 0.0)

        return policy_matrix

    def get_state_value_list(self) -> list:
        """
        将V(s)转换为GridWorld的state_values格式（适配add_state_values方法）
        :return: 按状态一维索引排序的价值列表
        """
        env_size = self.env.env_size
        value_list = []
        for y in range(env_size[1]):
            for x in range(env_size[0]):
                state = (x, y)
                value_list.append(self.V[state])
        return value_list
    
# 1. 初始化GridWorld（基于args配置）
env = GridWorld()

# 2. 初始化MC Agent
agent = MCAgent(
    env=env,
    gamma=0.9,    # 折扣因子
    alpha=0.01,   # 学习率
    epsilon=0.1   # ε-贪心探索率
)

# 3. 示例1：首次访问MC预测（评估随机策略的状态价值V(s)）
V = agent.first_visit_mc_prediction(n_episodes=5000, evaluate_q=False)
print("状态(0,0)的价值：", V[(0,0)])

# 4. 示例2：ε-贪心MC控制（优化策略）
optimal_policy = agent.mc_control_epsilon_greedy(n_episodes=10000)

# 5. 可视化策略和状态价值（适配GridWorld的render方法）
env.render()
# 转换策略为GridWorld兼容的矩阵格式
policy_matrix = agent.get_policy_matrix()
env.add_policy(policy_matrix)
# 转换状态价值为列表格式
value_list = agent.get_state_value_list()
env.add_state_values(value_list)
#
# 6. 测试优化后的策略（生成轨迹）
episode = agent.generate_episode()
print("优化后轨迹长度：", len(episode))
print("最终状态：", episode[-1][0])