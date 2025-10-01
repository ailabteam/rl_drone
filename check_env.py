import torch
import gymnasium as gym
import gym_pybullet_drones

def run_check():
    # 1. Kiểm tra GPU
    print("--- 1. Kiểm tra PyTorch và GPU ---")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Thành công! Tìm thấy {gpu_count} GPU.")
        print(f"GPU 0: {gpu_name}")
    else:
        print("Cảnh báo: Không tìm thấy GPU. PyTorch sẽ chạy trên CPU.")
        return False # Dừng lại nếu không có GPU

    # 2. Kiểm tra môi trường mô phỏng
    print("\n--- 2. Kiểm tra môi trường gym-pybullet-drones ---")
    try:
        # Tạo một môi trường đơn giản (không cần GUI)
        env = gym.make("hover-aviary-v0")
        obs, info = env.reset()
        print("Thành công! Khởi tạo môi trường 'hover-aviary-v0'.")
        print(f"Kích thước Observation space: {env.observation_space.shape}")
        print(f"Kích thước Action space: {env.action_space.shape}")
        env.close()
    except Exception as e:
        print(f"Lỗi! Không thể khởi tạo môi trường mô phỏng: {e}")
        return False

    # 3. Kiểm tra Stable Baselines3
    print("\n--- 3. Kiểm tra Stable Baselines3 ---")
    try:
        from stable_baselines3 import PPO
        print("Thành công! Import PPO từ Stable Baselines3.")
    except Exception as e:
        print(f"Lỗi! Không thể import Stable Baselines3: {e}")
        return False

    print("\n>>> Tất cả kiểm tra đã hoàn tất. Môi trường của bạn đã sẵn sàng! <<<")
    return True

if __name__ == "__main__":
    run_check()
