# RL-Drone: A Research Framework for Drone Control using Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a scalable and reproducible framework for developing and testing Reinforcement Learning algorithms for drone control tasks. It is built upon the robust `gym-pybullet-drones` simulation environment and leverages `stable-baselines3` for implementing state-of-the-art RL algorithms.

The primary goal of this framework is to facilitate research by providing a clean, configurable, and extensible codebase, enabling rapid prototyping and evaluation of new ideas for academic papers and personal projects.

## Features

- **Reproducible Environment**: Comes with a `requirements.txt` file and Docker support (coming soon) to ensure consistent experimental setups.
- **Configurable Experiments**: Utilizes YAML configuration files to easily manage environments, models, and hyperparameters without changing the source code.
- **Scalable Structure**: The project is organized to easily add new custom environments, algorithms, and experiments.
- **GPU Accelerated**: Optimized to leverage NVIDIA GPUs for faster training cycles.
- **TensorBoard Integration**: Track your training progress with real-time logs and visualizations.

## Getting Started

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) for environment management.
- An NVIDIA GPU with CUDA drivers is highly recommended for training. This project was developed and tested with CUDA 12.1.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ailabteam/rl_drone.git
    cd rl_drone
    ```

2.  **Create and activate the Conda environment:**
    This project uses Python 3.10.
    ```bash
    conda create -n rl_drone python=3.10 -y
    conda activate rl_drone
    ```

3.  **Install PyTorch with CUDA support:**
    Make sure to install the version compatible with your system's CUDA toolkit. The following command is for CUDA 12.1:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

4.  **Install all other dependencies:**
    The `requirements.txt` file contains the exact versions of all necessary packages to ensure reproducibility.
    ```bash
    pip install -r requirements.txt
    ```
    
    *Note: The `gym-pybullet-drones` library is installed in editable mode. If you haven't cloned it separately, you may need to do so:*
    ```bash
    # (Only if you encounter issues with the above command)
    # git clone https://github.com/utiasDSL/gym-pybullet-drones.git
    # pip install -e ./gym-pybullet-drones
    ```

## How to Use

### 1. Training an Agent

The main training script is `scripts/train.py`. It uses configuration files located in the `configs/` directory to define the experiment.

To start training, run the following command from the root directory:
```bash
python scripts/train.py --config configs/ppo_hover_config.yaml
```

- **Configuration**: You can create your own `.yaml` files in the `configs/` directory to define new experiments.
- **Results**: All training results, including models and logs, will be saved under the `results/` directory, organized by experiment name and timestamp.

### 2. Monitoring with TensorBoard

You can visualize the training progress in real-time using TensorBoard.

1.  Open a new terminal and activate the conda environment:
    ```bash
    conda activate rl_drone
    ```
2.  Start the TensorBoard server:
    ```bash
    tensorboard --logdir results/tensorboard_logs
    ```
3.  Open your web browser and navigate to `http://localhost:6006`.

### 3. Evaluating a Trained Agent

(Coming soon) A script `scripts/evaluate.py` will be provided to load a trained model and visualize its performance in the simulation environment with a GUI.

## Project Structure

```
rl_drone/
├── configs/                # YAML configuration files for experiments
├── notebooks/              # Jupyter notebooks for analysis and exploration
├── results/                # Output directory for models, logs, and plots
├── scripts/                # Main scripts (train, evaluate, etc.)
├── src/                    # Source code for custom environments, algorithms
│   ├── environments/
│   └── utils/
├── .gitignore              # Files to be ignored by Git
├── README.md               # This file
└── requirements.txt        # Frozen Python dependencies
```

## Contributing

Contributions are welcome! If you have ideas for new features, environments, or improvements, feel free to open an issue to discuss it or submit a pull request.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/my-new-feature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/my-new-feature`).
6.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) team for the excellent simulation environment.
- The developers of [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) for their robust RL implementations.
