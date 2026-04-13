import sys
sys.path.append("..")
import numpy as np
from grid_world import GridWorld

# ===================== 工具函数 =====================
def get_all_states(env):
    states = []
    for x in range(env.env_size[0]):
        for y in range(env.env_size[1]):
            states.append((x, y))
    return states

def state_to_idx(env, state):
    x, y = state
    return y * env.env_size[0] + x

# ===================== 策略评估 =====================
def policy_evaluation(env:GridWorld, policy, gamma=0.9, theta=1e-6):
    V = {s: 0.0 for s in get_all_states(env)}
    
    while True:
        delta = 0.0
        for state in get_all_states(env):
            if state == env.target_state:
                continue

            old_v = V[state]
            new_v = 0.0

            for action_arr, prob in policy[state].items():
                action = list(action_arr)
                next_state, reward = env._get_next_state_and_reward(state, action)
                done = env._is_done(next_state)
                if done:
                    new_v += prob * reward
                else:
                    new_v += prob * (reward + gamma * V[next_state])

            V[state] = new_v
            delta = max(delta, abs(old_v - new_v))

        if delta < theta:
            break
    return V

# ===================== 策略改进 =====================
# ===================== 修复：增加 policy 参数 =====================
def policy_improvement(env:GridWorld, policy, V, gamma=0.9):
    new_policy = {}
    policy_stable = True

    for state in get_all_states(env):
        if state == env.target_state:
            new_policy[state] = {tuple(a): 0.0 for a in env.action_space}
            continue

        # 现在 policy 已定义，不会报错
        old_best = [a for a, p in policy[state].items() if p == max(policy[state].values())]

        q = {}
        for action in env.action_space:
            next_state, reward = env._get_next_state_and_reward(state, action)
            done = env._is_done(next_state)
            q[tuple(action)] = reward if done else reward + gamma * V[next_state]

        max_q = max(q.values())
        best_acts = [a for a, v in q.items() if v == max_q]
        p = 1.0 / len(best_acts)

        new_policy[state] = {tuple(a): p if tuple(a) in best_acts else 0.0 for a in env.action_space}

        new_best = [a for a, v in new_policy[state].items() if v > 0]
        if set(old_best) != set(new_best):
            policy_stable = False

    return new_policy, policy_stable

# ===================== 策略迭代 =====================
def policy_iteration(env, gamma=0.9, theta=1e-6):
    n_actions = len(env.action_space)
    init_policy = {}

    for state in get_all_states(env):
        if state == env.target_state:
            init_policy[state] = {tuple(a): 0.0 for a in env.action_space}
            init_policy[state][(0,0)]=1.0
        else:
            init_policy[state] = {tuple(a): 1.0 / n_actions for a in env.action_space}

    policy = init_policy
    iter_num = 0

    while True:
        iter_num += 1
        V = policy_evaluation(env, policy, gamma, theta)
        
        # ===================== 修复：传入 policy =====================
        policy, stable = policy_improvement(env, policy, V, gamma)
        
        if stable:
            print(f"✅ 策略迭代收敛，迭代次数：{iter_num}")
            break
    return policy, V

# ===================== 值迭代 =====================
def value_iteration(env, gamma=0.9, theta=1e-6):
    V = {s: 0.0 for s in get_all_states(env)}
    iter_num = 0

    while True:
        delta = 0.0
        for state in get_all_states(env):
            if state == env.target_state:
                continue

            old_v = V[state]
            q_values = []
            for a in env.action_space:
                ns, r = env._get_next_state_and_reward(state, a)
                done = env._is_done(ns)
                q_values.append(r if done else r + gamma * V[ns])

            V[state] = max(q_values)
            delta = max(delta, abs(old_v - V[state]))

        iter_num += 1
        if delta < theta:
            print(f"✅ 值迭代收敛，迭代次数：{iter_num}")
            break

    optimal_policy = {}
    for state in get_all_states(env):
        if state == env.target_state:
            optimal_policy[state] = {tuple(a): 0.0 for a in env.action_space}
            continue

        q = {}
        for a in env.action_space:
            ns, r = env._get_next_state_and_reward(state, a)
            done = env._is_done(ns)
            q[tuple(a)] = r if done else r + gamma * V[ns]

        max_q = max(q.values())
        best_acts = [a for a, v in q.items() if v == max_q]
        p = 1.0 / len(best_acts)
        optimal_policy[state] = {tuple(a): p if tuple(a) in best_acts else 0.0 for a in env.action_space}

    return optimal_policy, V

# ===================== 截断策略迭代 =====================
def truncated_policy_iteration(env, gamma=0.9, theta=1e-6, max_value_iter=10):
    """
    截断策略迭代（Truncated Policy Iteration）
    严格对齐原始值迭代/策略迭代代码风格
    修复代码重复问题 | V 为字典，无索引错误
    """
    # 工具函数：计算Q(s,a)，消除代码重复（极简内嵌，不破坏风格）
    def compute_q(state, action):
        ns, r = env._get_next_state_and_reward(state, action)
        done = env._is_done(ns)
        return r if done else r + gamma * V[ns]

    # 初始化价值函数和策略
    V = {s: 0.0 for s in get_all_states(env)}
    policy = {s: {tuple(a): 1/len(env.action_space) for a in env.action_space} for s in get_all_states(env)}
    
    while True:
        # --------------------------
        # 截断策略评估
        # --------------------------
        for _ in range(max_value_iter):
            delta = 0.0
            for state in get_all_states(env):
                if state == env.target_state:
                    continue

                old_v = V[state]
                v_new = 0.0
                for a in env.action_space:
                    a_key = tuple(a)
                    q = compute_q(state, a)  # 调用函数，无重复
                    v_new += policy[state][a_key] * q

                V[state] = v_new
                delta = max(delta, abs(old_v - V[state]))

            if delta < theta:
                break

        # --------------------------
        # 策略改进
        # --------------------------
        policy_stable = True
        for state in get_all_states(env):
            if state == env.target_state:
                continue

            old_p = policy[state].copy()
            q = {}
            for a in env.action_space:
                a_key = tuple(a)
                q[a_key] = compute_q(state, a)  # 调用函数，无重复

            # 贪心更新策略
            max_q = max(q.values())
            best_acts = [a for a, v in q.items() if v == max_q]
            p = 1.0 / len(best_acts)
            for a_key in policy[state]:
                policy[state][a_key] = p if a_key in best_acts else 0.0

            if policy[state] != old_p:
                policy_stable = False

        if policy_stable:
            print(f"✅ 截断策略迭代收敛")
            break

    return policy, V
# ===================== 转可视化矩阵 =====================
def policy_to_matrix(env, policy):
    n_states = env.num_states
    n_actions = len(env.action_space)
    mat = np.zeros((n_states, n_actions))

    for state in get_all_states(env):
        idx = state_to_idx(env, state)
        for i, action in enumerate(env.action_space):
            mat[idx][i] = policy[state][tuple(action)]
    return mat

def value_to_list(env, V):
    return [V[s] for s in get_all_states(env)]