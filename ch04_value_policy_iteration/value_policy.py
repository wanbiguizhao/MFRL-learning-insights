import numpy as np

# ====================== 1. 修复后的网格环境 ======================
class GridWorld:
    def __init__(self):
        self.n_states = 9
        self.n_actions = 4  # 上0 下1 左2 右3
        self.terminal = 8   # 终点

    def get_model(self, s, a):
        if s == self.terminal:
            return s, 0, True

        # 👇 修复后的正确移动逻辑
        if a == 0:   # 上：不在第一行才能上
            s_next = s - 3 if s >= 3 else s
        elif a == 1: # 下：不在最后一行才能下（修复点！）
            s_next = s + 3 if s < 6 else s
        elif a == 2: # 左：不在最左列才能左
            s_next = s - 1 if s % 3 != 0 else s
        elif a == 3: # 右：不在最右列才能右
            s_next = s + 1 if s % 3 != 2 else s

        # 奖励
        r = 100 if s_next == self.terminal else -1
        done = (s_next == self.terminal)
        return s_next, r, done

# ====================== 2. 策略迭代 ======================
def policy_iteration(env, gamma=0.9, theta=1e-6):
    pi = np.ones([env.n_states, env.n_actions]) / env.n_actions
    V = np.zeros(env.n_states)

    while True:
        # 策略评估
        while True:
            delta = 0
            for s in range(env.n_states):
                v = 0
                for a in range(env.n_actions):
                    s_next, r, done = env.get_model(s, a)
                    v += pi[s][a] * (r + gamma * V[s_next])
                delta = max(delta, abs(v - V[s]))
                V[s] = v
            if delta < theta:
                break

        # 策略提升
        policy_stable = True
        for s in range(env.n_states):
            old_action = np.argmax(pi[s])
            q_list = []
            for a in range(env.n_actions):
                s_next, r, done = env.get_model(s, a)
                q = r + gamma * V[s_next]
                q_list.append(q)
            best_action = np.argmax(q_list)
            pi[s] = np.eye(env.n_actions)[best_action]
            if old_action != best_action:
                policy_stable = False

        if policy_stable:
            break
    return V, pi

# ====================== 3. 值迭代 ======================
def value_iteration(env, gamma=0.9, theta=1e-6):
    V = np.zeros(env.n_states)
    while True:
        delta = 0
        for s in range(env.n_states):
            v = V[s]
            q_list = []
            for a in range(env.n_actions):
                s_next, r, done = env.get_model(s, a)
                q = r + gamma * V[s_next]
                q_list.append(q)
            V[s] = max(q_list)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    pi = np.zeros([env.n_states, env.n_actions])
    for s in range(env.n_states):
        q_list = []
        for a in range(env.n_actions):
            s_next, r, done = env.get_model(s, a)
            q = r + gamma * V[s_next]
            q_list.append(q)
        best_action = np.argmax(q_list)
        pi[s][best_action] = 1
    return V, pi

# ====================== 运行 ======================
if __name__ == "__main__":
    env = GridWorld()
    print("==== 策略迭代 ====")
    V_pi, pi_pi = policy_iteration(env)
    print("状态价值：")
    print(np.round(V_pi.reshape(3,3), 1))
    print("最优策略（上0/下1/左2/右3）：")
    print(np.argmax(pi_pi, axis=1).reshape(3,3))

    print("\n==== 值迭代 ====")
    V_vi, pi_vi = value_iteration(env)
    print("状态价值：")
    print(np.round(V_vi.reshape(3,3), 1))
    print("最优策略（上0/下1/左2/右3）：")
    print(np.argmax(pi_vi, axis=1).reshape(3,3))