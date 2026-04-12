import matplotlib.pyplot as plt
import sys
import os
# 获取当前文件的上一级目录（根目录）
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 将根目录加入系统路径
sys.path.append(parent_dir)
from grid_world import GridWorld
from value_policy_iteration import (
    policy_iteration, value_iteration,
    policy_to_matrix, value_to_list
)

if __name__ == "__main__":
    # 初始化网格世界（使用你提供的GridWorld类）
    env = GridWorld()
    state = env.reset()  
    # ===================== 测试策略迭代 =====================
    print("="*50 + " 策略迭代 " + "="*50)
    pi_policy, pi_V = policy_iteration(env, gamma=0.9)
    
    # 转换策略/值函数为可视化格式
    pi_policy_matrix = policy_to_matrix(env, pi_policy)
    pi_values_list = value_to_list(env, pi_V)
    
    # 可视化策略迭代结果
    # plt.figure(figsize=(8, 6))
    env.render()
    env.add_policy(pi_policy_matrix)
    env.add_state_values(pi_values_list, precision=1)
    
    plt.title("Policy Iteration")
    # plt.savefig("policy_iteration_result.png")
    # plt.pause(15)
    # plt.close()
    
    # ===================== 测试值迭代 =====================
    print("\n" + "="*50 + " 值迭代 " + "="*50)
    vi_policy, vi_V = value_iteration(env, gamma=0.9)
    
    # 转换策略/值函数为可视化格式
    vi_policy_matrix = policy_to_matrix(env, vi_policy)
    vi_values_list = value_to_list(env, vi_V)
    
    # 可视化值迭代结果
    # plt.figure(figsize=(8, 6))
    state = env.reset() 
    env.render()
    env.add_policy(vi_policy_matrix)
    env.add_state_values(vi_values_list, precision=1)
    env.render()
    plt.title("Value Iteration")
    # plt.savefig("value_iteration_result.png")
    
    # plt.show(block=True)
    
    # 打印关键结果验证
    print("\n=== 目标状态值 ===")
    print(f"策略迭代目标状态({env.target_state})值：{pi_V[env.target_state]}")
    print(f"值迭代目标状态({env.target_state})值：{vi_V[env.target_state]}")
    
    print("\n=== 起始状态最优动作 ===")
    start_state = env.start_state
    pi_best_action = max(pi_policy[start_state].items(), key=lambda x: x[1])[0]
    vi_best_action = max(vi_policy[start_state].items(), key=lambda x: x[1])[0]

    # 把 tuple 转回 list 输出，保持格式一致
    pi_best_action = list(pi_best_action)
    vi_best_action = list(vi_best_action)

    print(f"策略迭代起始状态({start_state})最优动作：{pi_best_action}")
    print(f"值迭代起始状态({start_state})最优动作：{vi_best_action}")
    input("\n按下 Enter 键退出程序...")