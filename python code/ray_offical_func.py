import torch
import torch.optim as optim

import ray
from ray import tune, air
from ray.air import session

from ray.train import Checkpoint
import ray.cloudpickle as pickle
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import CLIReporter
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

import argparse
import os
import time

# 自訂 Callback，用來計算基於 iteration 的 iter_staleness
# from ray.tune.callback import Callback

# class IterStalenessCallback(Callback):
#     def on_trial_result(self, iteration, trials, trial, result, **info):
#         training_iters = [t.last_result.get("training_iteration", 0)
#                           for t in trials if t.last_result and "training_iteration" in t.last_result]
#         max_iter = max(training_iters) if training_iters else 0
#         current_iter = result.get("training_iteration", 0)
#         ckpt_interval = 50
#         iter_staleness = int((max_iter - current_iter) / ckpt_interval)

#         # 加入 debug print
#         # print(f"[IterStalenessCallback] Trial {trial.trial_id}: "
#         #       f"training_iters = {training_iters}, max_iter = {max_iter}, "
#         #       f"current_iter = {current_iter}, iter_staleness = {iter_staleness}")

#         result["iter_staleness"] = iter_staleness


# 參數設定
HEAD_NODE_IP = "192.168.50.35"  # 頭節點 IP
HYPER_NUM = 20  # 超參數數量
BATCH_SIZE = 512  # 每個 iteration 的 batch size
STOP_ITER = 2000  # 共訓練多少個 iteration
STOP_ACC = 0.8  # 達到該準確率停止訓練
INTERVAL_REPORT = 300  # terminal 顯示進度間隔
INTERVAL_CHECK = 50  # 每隔多少 iteration 儲存一次 checkpoint
PER_CPU = 12
PER_GPU = 1


# 建立資料讀取器
def get_data_loader(
    model_type,
    batch_size=64,
    data_dir="~/Documents/workspace/tune_population_based/data",
):
    data_dir = os.path.expanduser(data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    if model_type == "resnet-18":
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root=data_dir, train=True, download=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root=data_dir, train=False, download=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=False,
        )
    elif model_type == "resnet-50":
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(
                root=data_dir, train=True, download=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(
                root=data_dir, train=False, download=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=False,
        )
    return train_loader, test_loader


# 模型訓練函式
def train(model, optimizer, train_loader, device=None):
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        break  # 每個 iteration 只跑一個 batch


# 測試模型
def test(model, test_loader, device=None):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total


def training(config):
    # 初始化基於準確率的 staleness 相關變數
    best_acc_so_far = 0.0
    # staleness = 0
    # max_staleness = 0
    step = 1
    model_type = config.get("model_type", "resnet-18")
    train_loader, test_loader = get_data_loader(
        model_type, config.get("batch_size", BATCH_SIZE)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 建立模型與 optimizer
    if model_type == "resnet-18":
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif model_type == "resnet-50":
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get("lr", 0.01),
        momentum=config.get("momentum", 0.9),
    )

    # 若有 checkpoint，則從中恢復
    if session.get_checkpoint():
        checkpoint_obj = session.get_checkpoint()
        with checkpoint_obj.as_directory() as checkpoint_dir:
            ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pkl")
            with open(ckpt_path, "rb") as f:
                checkpoint_dict = pickle.load(f)
        model.load_state_dict(checkpoint_dict["model_state_dict"])
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            if "lr" in config:
                param_group["lr"] = config["lr"]
            if "momentum" in config:
                param_group["momentum"] = config["momentum"]
        last_step = checkpoint_dict["step"]
        step = last_step + 1
        best_acc_so_far = checkpoint_dict.get("best_acc_so_far", 0.0)
        # staleness = checkpoint_dict.get("staleness", 0)
        # max_staleness = checkpoint_dict.get("max_staleness", 0)

    while True:
        train(model, optimizer, train_loader, device)

        # 每隔 checkpoint_interval 進行測試與 checkpoint 儲存
        if step % config["checkpoint_interval"] == 0:
            acc = test(model, test_loader, device)
            # 更新基於準確率的 staleness
            # if acc > best_acc_so_far:
            #     best_acc_so_far = acc
            #     staleness = 0
            # else:
            #     staleness += 1
            #     if staleness > max_staleness:
            #         max_staleness = staleness

            # 將 checkpoint 存入持久化目錄（不刪除檔案）
            checkpoint_dir = os.path.join(
                config.get(
                    "persistent_checkpoint_dir",
                    os.path.expanduser(
                        "~/Documents/workspace/tune_population_based/persistent_checkpoints"
                    ),
                ),
                f"trial_{step}",
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_file = os.path.join(checkpoint_dir, "checkpoint.pkl")
            with open(ckpt_file, "wb") as f:
                pickle.dump(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_acc_so_far": best_acc_so_far,
                        # "staleness": staleness,
                        # "max_staleness": max_staleness,
                    },
                    f,
                )
            checkpoint = Checkpoint.from_directory(checkpoint_dir)

            # 報告時上傳 metrics，包含 training_iteration 與 checkpoint_interval
            session.report(
                {
                    "accuracy": acc,
                    "lr": config["lr"],
                    # "staleness": staleness,
                    # "max_staleness": max_staleness,
                    "training_iteration": step,
                    "checkpoint_interval": config["checkpoint_interval"],
                },
                checkpoint=checkpoint,
            )
        else:
            session.report(
                {
                    "accuracy": 0,
                    "lr": config["lr"],
                    # "staleness": staleness,
                    # "max_staleness": max_staleness,
                    "training_iteration": step,
                    "checkpoint_interval": config["checkpoint_interval"],
                },
                checkpoint=None,
            )
        step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_times", type=int, default=1)  # 重複實驗次數
    args = parser.parse_args()

    # 這裡不設定 storage_path 讓 Tune 不自動複製 checkpoint
    result_dir_path = os.path.expanduser(
        "~/Documents/workspace/tune_population_based/poly/nfs"
    )
    dir_path = os.path.expanduser("~/Documents/workspace/tune_population_based/poly")
    runtime_env = {
        "working_dir": os.path.dirname(os.path.abspath(__file__)),
        "excludes": [
            "data/",
            "my_model/",
            "ray_results/",
            "pytorch-cifar/",
            "results/",
            "nfs/",
            "checkpoints/",
            "persistent_checkpoints/",
        ],
    }

    with open(os.path.join(dir_path, "Running_Results.txt"), "a+") as out_result:
        out_result.write("+---------------+---------------+\n")
        out_result.write(f"{time.ctime()}  <<Ray Tune Results - {__file__}>> \n")
        out_result.write(f"Trial_num = {HYPER_NUM} \n")
        out_result.write(f"Stop iteration = {STOP_ITER} \n")
        out_result.write(f"Stop accuracy = {STOP_ACC} \n")
        out_result.write(f"Checkpoint interval = {INTERVAL_CHECK} \n")
        out_result.write(f"Batch size = {BATCH_SIZE} \n")
        out_result.write(f"Each Trial resource : cpu = {PER_CPU}, gpu = {PER_GPU} \n")

    # model_types = ["resnet-18", "resnet-50"]
    model_types = ["resnet-50"]
    for x in range(3):
        for model in model_types:
            with open(
                os.path.join(dir_path, "Running_Results.txt"), "a+"
            ) as out_result:
                out_result.write(f"model_type: {model} \n")
            avg_run_time = 0
            total_run_time = []
            avg_accuracy = 0
            # max_staleness_overall = 0
            # max_iter_staleness_overall = 0  # 全局最大 iter_staleness

            for _ in range(args.exp_times):
                if ray.is_initialized():
                    ray.shutdown()
                ray.init(
                    address="ray://" + HEAD_NODE_IP + ":10001", runtime_env=runtime_env
                )

                scheduler = PopulationBasedTraining(
                    time_attr="training_iteration",
                    perturbation_interval=INTERVAL_CHECK,
                    metric="accuracy",
                    mode="max",
                    hyperparam_mutations={
                        "lr": tune.uniform(0.0001, 1),
                        "momentum": [0.8, 0.9, 0.99],
                    },
                )

                reporter = CLIReporter(
                    max_report_frequency=INTERVAL_REPORT,
                    metric_columns=["training_iteration", "accuracy", "lr"],
                )

                # 將 callbacks 直接放入 Tuner 建構子參數中
                tuner = tune.Tuner(
                    tune.with_resources(
                        training, resources={"cpu": PER_CPU, "gpu": PER_GPU}
                    ),
                    run_config=air.RunConfig(
                        stop={"accuracy": STOP_ACC, "training_iteration": STOP_ITER},
                        verbose=1,
                        checkpoint_config=air.CheckpointConfig(
                            num_to_keep=3,
                        ),
                        storage_path=result_dir_path,
                        progress_reporter=reporter,
                        # callbacks=[IterStalenessCallback()]
                    ),
                    tune_config=tune.TuneConfig(
                        scheduler=scheduler,
                        num_samples=HYPER_NUM * (x + 1),
                    ),
                    param_space={
                        "lr": tune.uniform(0.001, 1),
                        "momentum": tune.uniform(0.001, 1),
                        "checkpoint_interval": INTERVAL_CHECK,
                        "model_type": model,
                        "persistent_checkpoint_dir": os.path.join(
                            dir_path, "persistent_checkpoints"
                        ),
                    },
                    # callbacks=[IterStalenessCallback()]
                )
                start_time = time.time()
                results_grid = tuner.fit()
                run_time = time.time() - start_time
                total_run_time.append(run_time)
                avg_run_time += run_time

                best_result = results_grid.get_best_result(
                    metric="accuracy", mode="max"
                )
                avg_accuracy += best_result.metrics["accuracy"]

                # 統計最大 max_staleness 與 iter_staleness
                # for result in results_grid:
                #     staleness_val = result.metrics.get("max_staleness", 0)
                #     if staleness_val > max_staleness_overall:
                #         max_staleness_overall = staleness_val
                #     iter_staleness_val = result.metrics.get("iter_staleness", 0)
                #     if iter_staleness_val > max_iter_staleness_overall:
                #         max_iter_staleness_overall = iter_staleness_val

                ray.shutdown()
                time.sleep(10)

            with open(
                os.path.join(dir_path, "Running_Results.txt"), "a+"
            ) as out_result:
                out_result.write(f"Trial num : {HYPER_NUM * (x + 1)} \n")
                out_result.write(
                    f"Avg_total_runtime : {avg_run_time / args.exp_times} \n"
                )
                out_result.write(f"Avg_accuracy : {avg_accuracy / args.exp_times} \n")
                out_result.write(f"total_run_time : {total_run_time} \n\n")
                # out_result.write(f"Max_staleness : {max_staleness_overall} \n")
                # out_result.write(f"Max_iter_staleness : {max_iter_staleness_overall} \n\n")
