import matplotlib.pyplot as plt
import sys
import os

# 路径配置（精简写法）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from grid_world import GridWorld
from value_policy import (
    policy_iteration, value_iteration, truncated_policy_iteration,
    policy_to_matrix, value_to_list
)

# --------------------------
# 【小工具函数】仅封装重复逻辑，无大函数
# --------------------------
def show_result(env, policy, value, title):
    """统一可视化：转换策略/价值 + 渲染 + 标题"""
    env.reset()  # 重置环境
    p_matrix = policy_to_matrix(env, policy)
    v_list = value_to_list(env, value)
    env.render()
    env.add_policy(p_matrix)
    env.add_state_values(v_list, precision=1)
    plt.title(title)

def print_best_action(env, policy, algo_name):
    """统一打印最优动作（精简重复打印）"""
    start_state = env.start_state
    best_act = max(policy[start_state].items(), key=lambda x: x[1])[0]
    print(f"{algo_name} 起始状态({start_state})最优动作：{list(best_act)}")

# --------------------------
# 主程序（极简流程）
# --------------------------
if __name__ == "__main__":
    env = GridWorld()  # 只初始化一次环境

    # 1. 策略迭代
    print("="*50 + " 策略迭代 " + "="*50)
    pi_policy, pi_V = policy_iteration(env, gamma=0.9)
    show_result(env, pi_policy, pi_V, "Policy Iteration")

    # 2. 值迭代
    print("\n" + "="*50 + " 值迭代 " + "="*50)
    vi_policy, vi_V = value_iteration(env, gamma=0.9)
    show_result(env, vi_policy, vi_V, "Value Iteration")

    # 3. 截断策略迭代（修复原变量命名混乱问题）
    print("\n" + "="*50 + " 截断策略迭代 " + "="*50)
    tpi_policy, tpi_V = truncated_policy_iteration(env, gamma=0.9, max_value_iter=5)
    show_result(env, tpi_policy, tpi_V, "Truncated Policy Iteration")

    # --------------------------
    # 统一打印结果验证
    # --------------------------
    print("\n=== 目标状态价值 ===")
    print(f"策略迭代：{pi_V[env.target_state]}")
    print(f"值迭代：{vi_V[env.target_state]}")
    print(f"截断策略迭代：{tpi_V[env.target_state]}")

    print("\n=== 起始状态最优动作 ===")
    print_best_action(env, pi_policy, "策略迭代")
    print_best_action(env, vi_policy, "值迭代")
    print_best_action(env, tpi_policy, "截断策略迭代")

    input("\n按下 Enter 键退出程序...")
    plt.show()  # 统一展示图像