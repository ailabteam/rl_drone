# RL-Drone: A Research Framework for Drone Control using Reinforcement Learning

[![Docker Hub](https://img.shields.io/docker/pulls/haodpsut/rl_drone.svg)](https://hub.docker.com/r/haodpsut/rl_drone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a scalable and reproducible framework for developing and testing Reinforcement Learning algorithms for drone control tasks. It is built upon the `gym-pybullet-drones` simulation environment and `stable-baselines3`.

The primary goal is to facilitate research by providing a clean, configurable, and containerized environment, enabling rapid and reliable experimentation.

## Getting Started with Docker (Recommended)

This is the easiest and most reliable way to use this framework. The pre-built Docker image contains all necessary dependencies and is available on Docker Hub.

**Note:** The default Docker image (`:latest` or `:1.0-cpu`) runs on **CPU**. This ensures it can be used on any machine, with or without an NVIDIA GPU.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your system.

### Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ailabteam/rl_drone.git
    cd rl_drone
    ```

2.  **Pull the Docker image from Docker Hub:**
    This downloads the ready-to-use CPU environment from the `haodpsut` repository.
    ```bash
    docker pull haodpsut/rl_drone:latest
    ```

3.  **Run the Docker container:**
    This command starts an interactive session inside the container and mounts your local project folder. This allows you to edit code locally and run it inside the container immediately.
    ```bash
    docker run --rm -it -v "$(pwd):/app" haodpsut/rl_drone:latest
    ```
    *(For Windows CMD/PowerShell, replace `$(pwd)` with `%cd%`)*

4.  **Start Training (Inside the Container):**
    Once inside the container's shell, you can run experiments. The training will use the CPU.
    ```bash
    # You are now inside the container
    python scripts/train.py --config configs/ppo_hover_config.yaml
    ```
    Results (models, logs) will be saved to your local `results/` folder.

<details>
<summary>For Developers: Building the image locally</summary>

If you modify the `Dockerfile` or want to build the image from scratch, run the following command from the root directory:
```bash
docker build -t haodpsut/rl_drone:latest .
```
</details>

## Local Installation (Alternative)

If you prefer not to use Docker, you can set up a local Conda environment.

<details>
<summary>Click to expand Conda installation instructions</summary>

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.10

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ailabteam/rl_drone.git
    cd rl_drone
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda create -n rl_drone python=3.10 -y
    conda activate rl_drone
    ```

3.  **Install all dependencies from the requirements file:**
    *Note: This `requirements.txt` file may contain GPU-specific versions of libraries like PyTorch. You may need to install the appropriate CPU/GPU version for your system first.*
    ```bash
    # Example for CPU PyTorch
    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
    ```

4.  **Start Training:**
    ```bash
    python scripts/train.py --config configs/ppo_hover_config.yaml
    ```
</details>

## How to Use

- **Training**: Run `scripts/train.py` with a specified config file.
- **Evaluation**: Run `scripts/evaluate.py --exp-dir path/to/experiment/folder` to visualize a trained agent.
- **Monitoring**: Use `tensorboard --logdir results/tensorboard_logs` to see training curves.

## Project Structure

```
rl_drone/
├── configs/                # YAML configuration files for experiments
├── notebooks/              # Jupyter notebooks for analysis and exploration
├── results/                # Output directory for models, logs, and plots
├── scripts/                # Main scripts (train, evaluate, etc.)
├── src/                    # Source code for custom environments, algorithms
├── .gitignore              # Files to be ignored by Git
├── Dockerfile              # Docker image definition for CPU
├── README.md               # This file
└── requirements.txt        # Frozen Python dependencies
```

## Contributing

Contributions are welcome! If you have ideas for new features, environments, or improvements, feel free to open an issue to discuss it or submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- The [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) team for the excellent simulation environment.
- The developers of [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) for their robust RL implementations.
