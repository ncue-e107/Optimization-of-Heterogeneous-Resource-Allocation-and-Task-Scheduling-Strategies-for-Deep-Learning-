# 🧠 Optimization of Heterogeneous Resource Allocation and Task Scheduling Strategies for Deep Learning Automatic Hyperparameter Tuning System Based on Ray Tune

![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)

This repository contains the official implementation of our paper on **hyperparameter tuning with heterogeneous resource scheduling** using **Ray Tune**. To ensure full reproducibility, we provide benchmark configurations, environment setup steps, and scripts for running all major experiments.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Reproducibility Checklist](#reproducibility-checklist)
- [How to Run](#how-to-run)
- [Benchmark Mapping](#benchmark-mapping)
- [Configurations](#configurations)
- [Components](#components)
- [Example Workflow](#example-workflow)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 📝 Overview

This project implements an advanced hyperparameter tuning system for deep learning models, focusing on optimizing **heterogeneous resource allocation** and **task scheduling strategies**. It supports **ResNet-18** and **ResNet-50** models, trained on **CIFAR-10** and **CIFAR-100**, and uses **Population-Based Training (PBT)** for evolving hyperparameters.

---

## ✨ Features

- ✅ Optimized for heterogeneous CPU/GPU cluster environments
- ✅ Advanced dynamic task scheduling
- ✅ Ray Tune backend for distributed experiment management
- ✅ Population-Based Training with checkpointing and mutation
- ✅ Real-time logs and training analytics
- ✅ Supports ResNet-18 and ResNet-50

---

## 🛠 Installation

```bash
git clone https://github.com/ncue-e107/Optimization-of-Heterogeneous-Resource-Allocation-and-Task-Scheduling-Strategies-for-Deep-Learning-.git
cd Optimization-of-Heterogeneous-Resource-Allocation-and-Task-Scheduling-Strategies-for-Deep-Learning-.git
pip install -r "python code/requirements.txt"

# (Optional) Create the default directory for saving program outputs and datasets:
mkdir -p ~/Documents/workspace/tune_population_based/
```

---

## ✅ Reproducibility Checklist

| Component             | Provided |
|----------------------|----------|
| Open-source code      | ✅ Yes |
| Dataset used          | ✅ CIFAR-10/100 (torchvision) |
| Training script       | ✅ `main.py` |
| Evaluation metrics    | ✅ Accuracy, training time |
| Multi-run average     | ✅ Available via --exp_times flag |
| Hardware reported     | ✅ Multi-node cluster, varying CPU/GPU |
| Checkpoints/logs      | ✅ `results/`, `Running_Results.txt` |

---

## 🚀 How to Run

```bash
python /python code/main.py --exp_times 3
```

This will run the experiment 3 times using the default configuration. Logs and results will be saved in `results/`.

---

## 📊 Benchmark Mapping

| Description                      | Script/Configuration              | Notes |
|----------------------------------|-----------------------------------|-------|
| Baseline (Ray PBT)              | [ray-project/ray:tune/examples/pbt_ppo_example.py](https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/pbt_ppo_example.py) | Official Ray PBT baseline |
| Ours (Heterogeneous PBT)        | `main.py` with dynamic placement | Includes resource-aware tuning |
| Reproducibility Evaluation      | `--exp_times N`| Multi-run logging enabled |

---

## ⚙️ Configurations

| Parameter               | Description                                    | Default Value |
|-------------------------|------------------------------------------------|---------------|
| `HYPER_NUM`             | Number of hyperparameter configurations        | 20            |
| `BATCH_SIZE`            | Batch size for training                        | 512           |
| `STOP_ITER`             | Stop training after this many iterations       | 1000          |
| `STOP_ACC`              | Stop training if accuracy exceeds this value   | 0.8           |
| `INTERVAL_REPORT`       | Interval for reporting progress (in seconds)   | 300           |
| `INTERVAL_CHECK`        | Checkpoint interval (in iterations)            | 50            |

---

## 🧩 Components

The system is modularized into five key components, each responsible for a specific function within the heterogeneous-aware hyperparameter tuning framework:

- **Task Scheduler (`create_new_trial`)**: Dynamically assigns hyperparameter trials to available resources based on scheduling policies and system load.

- **Resource Allocator (`set_placement_group`)**: Initializes and maintains a pool of CPU/GPU resource bundles for each trial, respecting heterogeneous hardware profiles across nodes.

- **PBT Core (`mutation`, `quantile`, `explore`)**: Handles population-based training logic, including performance evaluation, cloning, and mutation of hyperparameters.

- **Checkpoint Handler (`report_before_trial_end`)**: Periodically saves and restores model/optimizer states to support continued training and PBT mutation.

- **Logger/Reporter (`Reporter`)**: Periodically reports training progress, current accuracy, resource usage, and hyperparameter status to the terminal in real time.

---

## 🖇️ Example Workflow

**Start Ray Cluster**:

- Launch Ray using `ray start` or via cluster launcher with all nodes (e.g., `ray://192.168.50.35:10001`).
**Configure Resources**:
- Define CPU/GPU availability for each node in the `RESOURCE_ALLOCATION` dictionary in `main.py`.

**Run Experiments**:

- Use the following command to run the experiment multiple times:

     ```bash
     python /python code/main.py --exp_times 3
     ```

- This triggers multiple tuning runs for both ResNet-18 (CIFAR-10) and ResNet-50 (CIFAR-100).

**Monitor Training Progress**:

- The `Reporter` class logs training status, resource usage, accuracy, and trial states in real time.

**Evaluate Results**:

- Summary metrics are written to `Running_Results.txt`, and detailed accuracy logs are stored in `results/<timestamp>/`.

**Compare with Ray PBT Baseline**:

- Use official Ray PBT code for baseline comparison: [Ray PBT PPO Example](https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/pbt_ppo_example.py)

---

## Future Improvements

- **Support for Additional Models**:
  - Extend to support transformers and other architectures for broader application.

- **Integration with Advanced Schedulers**:
  - Incorporate more sophisticated scheduling algorithms for task prioritization.

- **Enhanced Visualization**:
  - Add graphical dashboards for progress monitoring and resource usage analytics.

- **Cloud Integration**:
  - Adapt the system for seamless deployment on cloud platforms like AWS, GCP, or Azure.

---

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License** (CC BY-NC 4.0).

### Permissions

- **Share**: You may copy and redistribute the material in any medium or format.
- **Adapt**: You may remix, transform, and build upon the material for non-commercial purposes.

### Restrictions

- **NonCommercial**: The material may not be used for commercial purposes.
- **Attribution Required**: Proper credit must be given, a link to the license must be provided, and any changes made must be indicated.

For more details, refer to the full license: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Acknowledgements

- [Python Ray](https://github.com/ray-project/ray)
- [PyTorch](https://github.com/pytorch/pytorch)
