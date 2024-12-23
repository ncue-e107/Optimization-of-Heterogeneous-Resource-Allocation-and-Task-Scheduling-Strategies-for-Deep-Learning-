# Optimization of Heterogeneous Resource Allocation and Task Scheduling Strategies for Deep Learning Automatic Hyperparameter Tuning System Based on Ray Tune

This project implements an advanced hyperparameter tuning system for deep learning models, focusing on optimizing **heterogeneous resource allocation** and **task scheduling strategies**. It uses **Ray Tune** to efficiently manage distributed workloads and performs **Population-Based Training (PBT)** to evolve hyperparameters dynamically.

The system supports **ResNet-18** and **ResNet-50** models, trained on **CIFAR-10** and **CIFAR-100** datasets, with automatic task scheduling and resource allocation strategies tailored for multi-node clusters with varying CPU and GPU resources.

---

## Features

1. **Optimized Resource Allocation:**
   - Adapts to heterogeneous clusters by balancing task distribution across nodes with different CPU and GPU configurations.
   
2. **Advanced Task Scheduling:**
   - Dynamically schedules training jobs based on available resources and trial progress.
   
3. **Population-Based Training (PBT):**
   - Evolves hyperparameters through exploration (mutation) and exploitation (copying successful configurations).

4. **Distributed Training with Ray Tune:**
   - Efficiently distributes workloads across multiple nodes for scalable hyperparameter tuning.

5. **Real-time Reporting and Analytics:**
   - Monitors resource usage, training progress, and hyperparameter configurations in real-time.

6. **Model Support:**
   - Supports **ResNet-18** and **ResNet-50**, customizable for other models and datasets.

7. **Checkpointing and Recovery:**
   - Periodically saves training state for resumption and supports fault-tolerance.

---

## Requirements

Install the dependencies using:
```bash
pip install -r /python code/requirements.txt
```

---

## Usage

### 1. Configure the Environment
- Update the `RESOURCE_ALLOCATION` dictionary in the script to reflect the available CPU and GPU resources for your cluster.

### 2. Run the System
Execute the script with:
```bash
python python code/main.py --exp_times <number_of_experiments>
```
Replace `<number_of_experiments>` with the desired number of repetitions for the experiment.

### 3. Monitor Training
The system provides real-time logs of:
- Resource usage (CPU/GPU)
- Hyperparameter configurations
- Training progress (accuracy and iterations)

### 4. Analyze Results
Output logs and checkpoints are saved in the `results/` directory. Summary statistics, including average runtime and accuracy, are logged in `Running_Results.txt`.

---

## Configurations

| Parameter               | Description                                    | Default Value |
|-------------------------|------------------------------------------------|---------------|
| `HYPER_NUM`             | Number of hyperparameter configurations        | 20            |
| `BATCH_SIZE`            | Batch size for training                        | 512           |
| `STOP_ITER`             | Stop training after this many iterations       | 1000          |
| `STOP_ACC`              | Stop training if accuracy exceeds this value   | 0.8           |
| `INTERVAL_REPORT`       | Interval for reporting progress (in seconds)   | 300           |
| `INTERVAL_CHECK`        | Checkpoint interval (in iterations)            | 50            |

---

## Key Components

1. **Task Scheduler and Resource Allocator**:
   - Dynamically assigns tasks to nodes based on available resources and workload efficiency.

2. **Population-Based Training (PBT)**:
   - Evolves hyperparameters such as learning rate and momentum to optimize model performance.

3. **Checkpoints and Recovery**:
   - Saves model and optimizer states periodically to allow recovery and efficient mutation.

4. **Real-time Reporter**:
   - Provides updates on training status, resource utilization, and hyperparameter states.

---

## Example Workflow

1. Start the Ray cluster with connected nodes configured for heterogeneous resources.
2. Customize resource allocation and hyperparameter mutation ranges in the script.
3. Execute the script to initiate the hyperparameter tuning system.
4. Monitor real-time logs in the terminal.
5. Analyze the training results and model performance from the output directory.

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

This project is licensed under the MIT License. See `LICENSE` for more details.
