__credits__ = ["Intelligent Unmanned Systems Laboratory at Westlake University."]
'''
Specify parameters of the env
'''
from typing import Union
import numpy as np
import argparse
# 新增：导入yaml读取模块（需安装pyyaml，命令：pip install pyyaml）
import yaml
from pathlib import Path

def load_config(config_path: str = "config.yaml"):
    """加载配置文件（yaml格式），返回配置字典"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件 {config_path} 不存在，请检查路径！")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

# 1. 先加载配置文件，获取默认参数
config = load_config()  # 读取config.yaml中的配置

# 2. 初始化argparse，默认值从配置文件读取（不再写死）
parser = argparse.ArgumentParser("Grid World Environment")

## ==================== User settings ====================
# specify the number of columns and rows of the grid world
parser.add_argument("--env-size", type=Union[list, tuple, np.ndarray], 
                    default=config["env_size"] )   

# specify the start state
parser.add_argument("--start-state", type=Union[list, tuple, np.ndarray], 
                    default=config["start_state"])

# specify the target state
parser.add_argument("--target-state", type=Union[list, tuple, np.ndarray], 
                    default=config["target_state"])

# sepcify the forbidden states
parser.add_argument("--forbidden-states", type=list, 
                    default=config["forbidden_states"] )

# sepcify the reward when reaching target
parser.add_argument("--reward-target", type=float, 
                    default = config["reward_target"])

# sepcify the reward when entering into forbidden area
parser.add_argument("--reward-forbidden", type=float, 
                    default = config["reward_forbidden"])

# sepcify the reward for each step
parser.add_argument("--reward-step", type=float, 
                    default = config["reward_step"])
## ==================== End of User settings ====================


## ==================== Advanced Settings ====================
parser.add_argument("--action-space", type=list, 
                    default=config["action_space"] )  # down, right, up, left, stay           
parser.add_argument("--debug", type=bool, 
                    default=config["debug"])
parser.add_argument("--animation-interval", type=float, 
                    default = config["animation_interval"])
# 新增：允许指定自定义配置文件路径（可选）
parser.add_argument("--config-path", type=str, default="config.yaml", 
                    help="自定义配置文件路径（默认：config.yaml）")
## ==================== End of Advanced settings ====================

# 3. 解析命令行参数（若命令行输入了参数，会覆盖配置文件中的默认值）
args = parser.parse_args()     

# 补充：若命令行指定了自定义配置文件，重新加载配置并更新参数
if args.config_path != "config.yaml":
    config = load_config(args.config_path)
    # 将自定义配置文件的参数更新到args中（命令行参数仍优先）
    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) == config[key]:
            setattr(args, key, value)

# 原有参数校验逻辑（完全不变，直接复用）
def validate_environment_parameters(env_size, start_state, target_state, forbidden_states):
    # 补充：将输入的list转换为tuple，避免后续校验报错（可选，增强兼容性）
    env_size = tuple(env_size) if isinstance(env_size, list) else env_size
    start_state = tuple(start_state) if isinstance(start_state, list) else start_state
    target_state = tuple(target_state) if isinstance(target_state, list) else target_state
    forbidden_states = [tuple(state) for state in forbidden_states] if isinstance(forbidden_states, list) else forbidden_states
    
    if not (isinstance(env_size, tuple) or isinstance(env_size, list) or isinstance(env_size, np.ndarray)) or len(env_size) != 2:
        raise ValueError("Invalid environment size. Expected a tuple (rows, cols) with positive dimensions.")
    
    for i in range(2):
        assert start_state[i] < env_size[i], f"起始状态第{i}维超出网格范围（网格大小：{env_size}）"
        assert target_state[i] < env_size[i], f"目标状态第{i}维超出网格范围（网格大小：{env_size}）"
        for j in range(len(forbidden_states)):
            assert forbidden_states[j][i] < env_size[i], f"禁止状态第{j}个第{i}维超出网格范围（网格大小：{env_size}）"

try:
    validate_environment_parameters(args.env_size, args.start_state, args.target_state, args.forbidden_states)
    print("环境参数校验通过！")
except ValueError as e:
    print("Error:", e)
except AssertionError as e:
    print("参数校验失败:", e)