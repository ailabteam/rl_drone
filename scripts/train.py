# scripts/train.py

import os
import time
from datetime import datetime
import yaml
import argparse

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

# Import môi trường, dù không gọi trực tiếp nhưng cần để đăng ký với Gym
import gym_pybullet_drones

def load_config(config_path: str) -> dict:
    """Tải file cấu hình YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_experiment_dirs(base_log_path: str, tb_log_name: str) -> (str, str, str):
    """Tạo các thư mục cho một lần thực nghiệm."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{tb_log_name}_{timestamp}"

    log_dir = os.path.join(base_log_path, experiment_name)
    model_dir = os.path.join(log_dir, "models")
    
    os.makedirs(model_dir, exist_ok=True)
    
    return log_dir, model_dir

def train(config_path: str):
    """
    Hàm chính để huấn luyện mô hình.

    :param config_path: Đường dẫn tới file cấu hình YAML.
    """
    # 1. Tải cấu hình và thiết lập thư mục
    config = load_config(config_path)
    print("Configuration loaded:")
    print(config)

    log_dir, model_dir = create_experiment_dirs("results/logs", config['tb_log_name'])

    # 2. Tạo môi trường Vectorized
    # make_vec_env sẽ tự động bao bọc môi trường trong DummyVecEnv
    train_env = make_vec_env(
        config['env_id'],
        n_envs=config['n_envs'],
        seed=config['seed']
    )
    # (Optional but recommended) Chuẩn hóa observation và reward
    # Lưu ý: cần lưu lại stats của env nếu muốn đánh giá sau này
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    # 3. Thiết lập Callbacks
    # Callback để lưu model tốt nhất dựa trên kết quả đánh giá
    eval_callback = EvalCallback(
        train_env, # Sử dụng chính train_env để eval cho đơn giản, hoặc tạo một env riêng
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=max(config['eval_freq'] // config['n_envs'], 1),
        n_eval_episodes=config['n_eval_episodes'],
        deterministic=config['deterministic_eval'],
        render=False
    )
    
    # Callback để lưu checkpoint định kỳ
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // config['n_envs'], 1),
        save_path=model_dir,
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )


    # 4. Khởi tạo mô hình PPO
    # **kwargs cho phép truyền tất cả các siêu tham số từ file config vào model
    ppo_params = {k: v for k, v in config.items() if k in [
        'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma',
        'gae_lambda', 'clip_range', 'ent_coef', 'vf_coef', 'max_grad_norm'
    ]}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = PPO(
        config['policy'],
        train_env,
        tensorboard_log=os.path.join("results", "tensorboard_logs"),
        verbose=1,
        device=device,
        **ppo_params
    )

    # 5. Bắt đầu huấn luyện
    print(f"--- Starting training for {config['total_timesteps']} timesteps ---")
    start_time = time.time()
    
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=[eval_callback, checkpoint_callback], # Có thể dùng nhiều callback
        log_interval=config['log_interval'],
        tb_log_name=os.path.basename(log_dir) # Dùng tên thư mục thực nghiệm làm tên trên TensorBoard
    )
    
    end_time = time.time()
    print(f"--- Training finished in {(end_time - start_time) / 60:.2f} minutes ---")

    # 6. Lưu model cuối cùng và stats của VecNormalize
    model.save(os.path.join(model_dir, "final_model"))
    train_env.save(os.path.join(model_dir, "vec_normalize.pkl"))
    print(f"Final model and environment stats saved in: {model_dir}")


if __name__ == '__main__':
    # Thêm Argument Parser để có thể chạy từ dòng lệnh với các config khác nhau
    parser = argparse.ArgumentParser(description="Train an RL agent for drones.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ppo_hover_config.yaml",
        help="Path to the configuration file."
    )
    args = parser.parse_args()
    
    train(config_path=args.config)
