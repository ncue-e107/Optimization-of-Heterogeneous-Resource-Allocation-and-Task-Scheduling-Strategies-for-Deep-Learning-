# Population-Based Training with Ray for ResNet Models

This repository implements a **Population-Based Training (PBT)** system for hyperparameter optimization of **ResNet-18** and **ResNet-50** models on the **CIFAR-10** and **CIFAR-100** datasets. The solution leverages **Ray**, a distributed computing framework, to manage resources and perform hyperparameter exploration and exploitation efficiently.

---

## Features

1. **Population-Based Training (PBT):**
   - Adaptive hyperparameter tuning using exploration (mutation of hyperparameters) and exploitation (copying successful configurations).
   - Tracks model performance to evolve hyperparameters dynamically during training.

2. **Distributed Training:**
   - Implements a resource-aware system using **Ray** to distribute training workloads across multiple nodes.
   - Customizable CPU and GPU resource allocation for each machine in the cluster.

3. **Model Architectures:**
   - Supports **ResNet-18** (CIFAR-10) and **ResNet-50** (CIFAR-100) architectures with modified fully connected layers for classification tasks.

4. **Checkpointing:**
   - Periodically saves model and optimizer states, enabling efficient resumption during hyperparameter tuning or training interruptions.

5. **Real-time Reporting:**
   - Provides detailed, real-time insights into trial progress, resource utilization, and hyperparameter states during training.

6. **Configurable Parameters:**
   - Flexible configurations for the number of trials, stopping criteria, and hyperparameter mutation ranges.

7. **Result Logging:**
   - Logs training runtime, resource usage, and achieved accuracy into structured output files for post-analysis.

---

## Requirements

- Python 3.8+
- PyTorch
- Ray
- torchvision
- matplotlib
- numpy
- argparse

Install the required packages with:
```bash
pip install torch torchvision ray matplotlib numpy
```

---

## Usage

### 1. Configure Resource Allocation
Update the `RESOURCE_ALLOCATION` dictionary in the script to define the CPU and GPU resources for each node in your cluster:
```python
RESOURCE_ALLOCATION = {
    "CPU": {
        "192.168.50.34": 16,
        "192.168.50.35": 12,
        "192.168.50.36": 4,
        ...
    },
    "GPU": {
        "192.168.50.35": 1
    }
}
```

### 2. Run the Training
Execute the script with:
```bash
python script_name.py --exp_times <number_of_experiments>
```
Replace `<number_of_experiments>` with the desired number of repetitions for the experiment.

### 3. Monitor Progress
The script provides real-time updates on:
- Hyperparameter configurations
- Resource usage (CPU and GPU)
- Accuracy and iteration progress

### 4. Analyze Results
Output logs and accuracy checkpoints are saved in a structured directory under `results/`. Training statistics, including runtime and accuracy, are appended to `Running_Results.txt`.

---

## Configurable Parameters

| Parameter               | Description                                    | Default Value |
|-------------------------|------------------------------------------------|---------------|
| `HYPER_NUM`             | Number of hyperparameter configurations        | 20            |
| `BATCH_SIZE`            | Batch size for training                        | 512           |
| `STOP_ITER`             | Stop training after this many iterations       | 1000          |
| `STOP_ACC`              | Stop training if accuracy exceeds this value   | 0.8           |
| `INTERVAL_REPORT`       | Interval for reporting progress (in seconds)   | 300           |
| `INTERVAL_CHECK`        | Checkpoint interval (in iterations)            | 50            |

To customize these parameters, modify their values in the script or create a configuration file.

---

## Code Structure

### Key Components
1. **Data Loaders**:
   - `get_data_loader`: Dynamically creates data loaders for CIFAR-10 and CIFAR-100 datasets.

2. **Training Utilities**:
   - `train`: Executes one iteration of training.
   - `test`: Evaluates the model on test data to calculate accuracy.

3. **Population-Based Training**:
   - `Tuner`: Manages hyperparameter configurations, resource scheduling, and checkpointing.
   - `mutation`: Handles hyperparameter exploration and exploitation.

4. **Distributed Training**:
   - `Trial`: Executes the training process for a given hyperparameter configuration.
   - `Reporter`: Provides real-time status updates on trials and resources.

### Output
- **Real-time Logs**: Detailed progress and status updates in the terminal.
- **Results Directory**: Contains per-trial accuracy logs and checkpoints.

---

## Example Workflow

1. Start the Ray cluster, ensuring all nodes are correctly configured and connected.
2. Configure resource allocation and desired hyperparameter ranges in the script.
3. Run the script to initiate PBT.
4. Monitor progress in the terminal.
5. Analyze the results logged in `Running_Results.txt` and the `results/` directory.

---

## Future Improvements

- **Support for Additional Models**:
  - Extend support to other architectures like EfficientNet or Vision Transformers.
  
- **Dynamic Resource Scaling**:
  - Implement adaptive resource scaling based on training load and node availability.

- **Advanced Visualization**:
  - Enhance reporting with graphical dashboards using tools like TensorBoard or Matplotlib.

- **Integration with Ray Tune**:
  - Leverage Ray Tune's built-in PBT capabilities for streamlined experimentation.

---

## License

This project is licensed under the MIT License. See `LICENSE` for more details.
