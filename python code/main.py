import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import ray

from numpy import random
import time
import copy
import os
import heapq
import json
import math
import argparse


HEAD_NODE_IP = "192.168.50.35"  # 頭節點IP
HYPER_NUM = 20  # 超參數數量
BATCH_SIZE = 512  # 訓練一個interation的batch size
STOP_ITER = 1000  # 共訓練幾個iteration
STOP_ACC = 0.8  # 訓練到準確率停止
INTERVAL_REPORT = 300  # 間隔多久在ternimal中顯示執行過程
INTERVAL_CHECK = 50
RESOURCE_ALLOCATION = {  # 每台電腦的資源分配
    "CPU": {
        "192.168.50.34": 16,
        "192.168.50.35": 12,
        "192.168.50.36": 4,
        "192.168.50.37": 4,
        "192.168.50.38": 4,
        "192.168.50.39": 20,
        "192.168.50.43": 12,
        "192.168.50.44": 12,
    },
    "GPU": {"192.168.50.35": 1},
}
# TEST_SIZE = 25


# 建立data_loader
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


# 模型訓練
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
        break


def test(model, test_loader, device):
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


# 用來顯示當前所有Trail的狀態 (在command中顯示)
@ray.remote(num_cpus=0.1, resources={"node:" + HEAD_NODE_IP: 0.1})
def Reporter(tuner, max_report_frequency=5, hyper_num=1):
    start_time = ray.get(tuner.get_start_time.remote())
    resource = ray.get(tuner.get_resource.remote())
    while True:
        hypers, accuracy, state, perturbs, running_trial_num = ray.get(
            tuner.get_for_reporter.remote()
        )
        m, s = divmod(time.time() - start_time, 60)
        h, m = divmod(m, 60)
        if "CPU" in ray.available_resources():
            unused_cpu_num = ray.available_resources()["CPU"]
        else:
            unused_cpu_num = 0
        if "GPU" in ray.available_resources():
            unused_gpu_num = ray.available_resources()["GPU"]
        else:
            unused_gpu_num = 0

        print("== Status ==")
        print(
            f"Current Time : {time.ctime()} (runnung for {str(int(h)).zfill(2)}:{str(int(m)).zfill(2)}:{str(int(s)).zfill(2)})"
        )
        print(f"Unused Resource : {unused_cpu_num} CPUs and {unused_gpu_num} GPUs")
        print(f"PBT : {perturbs} perturbs")
        print(f"Total hypers : {hyper_num} ( {running_trial_num} is training )")
        print(
            "+--------------+------------+------------+---------------------+------------+------------+------------+----------+-------+-----------------+"
        )
        print(
            "| Hyper name   |   Status   |  CPU / GPU |                  IP |         lr |   momentum | batch_size |      acc |  iter |   total time (s)|"
        )
        print(
            "+--------------+------------+------------+---------------------+------------+------------+------------+----------+-------+-----------------+"
        )
        for i, (hyper, acc, sta) in enumerate(zip(hypers, accuracy, state)):
            if sta["resource_id"] == -2:
                status = "TERNIMAL"
                cpus_per_trial = 0
                gpus_per_trial = 0
                ip = "node:0.0.0.0"
            elif sta["resource_id"] == -1:
                status = "PENDING"
                cpus_per_trial = 0
                gpus_per_trial = 0
                ip = "node:0.0.0.0"
            else:
                status = "RUNNING"
                cpus_per_trial = resource[sta["resource_id"]]["CPU"]
                gpus_per_trial = resource[sta["resource_id"]]["GPU"]
                ip = resource[sta["resource_id"]]["node"]

            print(
                f"| hyper_{str(i).zfill(5)}  |  {status:^8}  | {cpus_per_trial:>4.1f} / {gpus_per_trial:<3.1f} | {ip:>19} | {hyper['lr']:10.6f} | {hyper['momentum']:10.6f} | {hyper['batch_size']:>11}| {acc:8.4f} | {sta['iteration']:>5} | {sta['run_time']:15.6f} | "
            )
        print(
            "+--------------+------------+------------+---------------------+------------+------------+------------+----------+-------+-----------------+"
        )
        time.sleep(max_report_frequency)


@ray.remote(num_cpus=0.2, resources={"node:" + HEAD_NODE_IP: 0.1})
class Tuner(object):
    """
    Tuner : 控制所有Trial進程, 創建與接收Trial的結果。

    Args:
        hyper_num : 共建立多少hyper組合
        batch_siz : 一個iteration訓練的batch大小
        stop_acc : 訓練停止的accuracy條件
        stop_iteration : 訓練停止的iteration條件
        checkpoint_interval : 多少iteration存一個checkpoint
        trials_state : 存每筆hyper訓練使用的資源id，以及訓練花費時間
        resource : 存這樣分配(RESOURCE_ALLOCATION)總共有多少組資源
        avaliable_resource : 存目前可用資源 (正在使用和效能太差就不會在裡面)
        trials_scheduler : 存要訓練的hyper_id (到達終止條件的就不會在裡面)
        running_trial_num : 正在執行訓練的trial數量
        min_run_one_interval_time : 執行一個interval最少需要的時間 (當計算每個資源能力的基礎)

    """

    def __init__(
        self,
        hyper_num=1,
        model_type="resnet-18",
        resource_allocation=None,
        stop_acc=1,
        stop_iteration=0,
        checkpoint_interval=5,
        hyperparam_mutations=None,
        path=None,
    ):
        self.start_time = time.time()
        self.tuner = None
        self.hyper_num = hyper_num
        self.model_type = model_type
        self.stop_acc = stop_acc
        self.stop_iteration = stop_iteration
        self.checkpoint_interval = checkpoint_interval
        self.hyperparam_mutations = hyperparam_mutations
        self.path = path

        self.trials_scheduler = []
        self.hypers = []
        self.trials_state = []
        self.checkpoints = []
        self.last_checkpoint = [0] * hyper_num
        self.perturbs = 0
        self.trial_acc_list = [0] * hyper_num
        self.resource = []
        self.avaliable_resource = []

        self.running_trial_num = 0
        self.min_run_one_interval_time = 9999
        self.max_iter = 0
        self.max_acc = -1
        self.last_run_interval = 9999

        self.initialize_all_config()
        self.set_placement_group(resource_allocation)

    # 初始化每組hyper的值與checkpoint
    def initialize_all_config(self):
        if self.model_type == "resnet-18":
            # 建立model
            model = models.resnet18()
            # 修改全连接层的输出
            model.fc = nn.Linear(model.fc.in_features, 10)
        elif self.model_type == "resnet-50":
            # 建立model
            model = models.resnet50()
            # 修改全连接层的输出
            model.fc = nn.Linear(model.fc.in_features, 100)
        optimizer = optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
        )

        for i in range(self.hyper_num):
            hyper = {
                "lr": random.uniform(0.001, 1),
                "momentum": random.uniform(0.001, 1),
                "batch_size": 512,  # int(random.choice([32, 64, 128, 256, 512])),
                "model_type": self.model_type,
            }
            trial_state = {
                "resource_id": -1,
                "run_time": 0,
                "iteration": 0,
            }
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "checkpoint_interval": self.checkpoint_interval,
            }
            self.trials_scheduler.append(i)
            self.hypers.append(hyper)
            self.trials_state.append(trial_state)
            self.checkpoints.append(checkpoint)

    # 分配每個節點在訓練時能使用的資源數
    def set_placement_group(self, resource_allocation):
        # print(ray.available_resources())
        for nodes in ray.nodes():
            if (
                "CPU" in nodes["Resources"]
                and nodes["NodeManagerAddress"] in resource_allocation["CPU"]
            ):
                if nodes["NodeManagerAddress"] == HEAD_NODE_IP:
                    sub = 1
                else:
                    sub = 0
                sum = nodes["Resources"]["CPU"]
                while int(
                    sum / (resource_allocation["CPU"][nodes["NodeManagerAddress"]])
                ):
                    self.resource.append(
                        {
                            "CPU": resource_allocation["CPU"][
                                nodes["NodeManagerAddress"]
                            ]
                            - sub,
                            "GPU": 0,
                            "node": "node:" + nodes["NodeManagerAddress"],
                            "calculate_ability": 0,
                            "Used_count": 0,
                        }
                    )
                    print(self.resource[-1])
                    sub = 0
                    sum -= resource_allocation["CPU"][nodes["NodeManagerAddress"]]
                    # print("CPU", resource_allocation["CPU"][nodes['NodeManagerAddress']])
            if (
                "GPU" in nodes["Resources"]
                and nodes["NodeManagerAddress"] in resource_allocation["GPU"]
            ):
                sum = nodes["Resources"]["GPU"]
                while int(
                    sum / resource_allocation["GPU"][nodes["NodeManagerAddress"]]
                ):
                    self.resource.append(
                        {
                            "CPU": 0,
                            "GPU": resource_allocation["GPU"][
                                nodes["NodeManagerAddress"]
                            ],
                            "node": "node:" + nodes["NodeManagerAddress"],
                            "calculate_ability": 0,
                            "Used_count": 0,
                        }
                    )
                    print(self.resource[-1])
                    sum -= resource_allocation["GPU"][nodes["NodeManagerAddress"]]
                    # print("GPU", resource_allocation["GPU"][nodes['NodeManagerAddress']])

        for i in range(len(self.resource)):
            self.avaliable_resource.append(i)

    # 建立新的trial
    def create_new_trial(self):
        # self.trials_scheduler.sort(key = lambda t: self.trials_state[t]["iteration"])

        remaining_generations = 0
        for trial_state in self.trials_state:
            if trial_state["resource_id"] < 0:
                remaining_generations += math.ceil(
                    (self.max_iter - trial_state["iteration"])
                    / self.checkpoint_interval
                )
        cc = 1.0
        if len(self.trials_scheduler):
            while True:
                if not len(self.avaliable_resource):
                    break

                resource_id = self.avaliable_resource.pop(0)
                if (
                    self.resource[resource_id]["calculate_ability"] == 0
                    or self.resource[resource_id]["calculate_ability"]
                    > remaining_generations
                ):  # 讓很差的資源訓練進度最快的
                    if (
                        self.stop_iteration
                        and self.resource[resource_id]["calculate_ability"]
                        > self.last_run_interval
                    ):
                        continue
                    if self.stop_acc and self.max_acc >= self.stop_acc:
                        continue
                    id = self.trials_scheduler.pop(-1)
                    if (
                        self.stop_iteration - self.trials_state[id]["iteration"]
                    ) < self.checkpoint_interval / 2:
                        self.checkpoints[id]["checkpoint_interval"] = (
                            self.stop_iteration - self.trials_state[id]["iteration"]
                        )
                    else:
                        self.checkpoints[id]["checkpoint_interval"] = int(
                            self.checkpoint_interval / 2
                        )
                    cc = 0.5
                else:
                    id = self.trials_scheduler.pop(0)
                    if (
                        self.stop_iteration - self.trials_state[id]["iteration"]
                    ) < self.checkpoint_interval:
                        self.checkpoints[id]["checkpoint_interval"] = (
                            self.stop_iteration - self.trials_state[id]["iteration"]
                        )
                    else:
                        self.checkpoints[id]["checkpoint_interval"] = (
                            self.checkpoint_interval
                        )

                # Training Remote
                Trial.options(
                    num_cpus=self.resource[resource_id]["CPU"],
                    num_gpus=self.resource[resource_id]["GPU"],
                    resources={self.resource[resource_id]["node"]: 0.1},
                ).remote(self.tuner, id, self.hypers[id], self.checkpoints[id])

                self.running_trial_num += 1
                self.resource[resource_id]["Used_count"] += cc
                self.trials_state[id]["resource_id"] = resource_id

                break

    # 處理訓練完要結束的trial
    def report_before_trial_end(self, id, accuracy, run_time, checkpoint):
        self.trial_acc_list[id] = accuracy

        if (
            checkpoint["checkpoint_interval"] >= self.checkpoint_interval
        ):  # 檢查是否達到突變時間間隔
            mutation.remote(
                self.tuner,
                id,
                self.hypers,
                self.trial_acc_list,
                self.last_checkpoint,
                self.hyperparam_mutations,
            )

        resource_id = self.trials_state[id]["resource_id"]
        self.trials_state[id]["resource_id"] = -1  # -1 : 表示沒有要訓練(釋放資源)

        self.trials_state[id]["run_time"] += run_time
        self.trials_state[id]["iteration"] += checkpoint["checkpoint_interval"]
        self.checkpoints[id] = checkpoint

        save_acc_to_json.remote(
            id, accuracy, self.trials_state[id]["iteration"], self.path
        )

        if self.resource[resource_id]["Used_count"] == 0.5:
            self.min_run_one_interval_time = min(
                self.min_run_one_interval_time, run_time
            )
            calculate_ability = math.ceil(run_time / self.min_run_one_interval_time)
            for resource in self.resource:
                if resource["calculate_ability"]:
                    self.resource[resource_id]["calculate_ability"] += int(
                        calculate_ability / resource["calculate_ability"]
                    )
            self.resource[resource_id]["calculate_ability"] += calculate_ability
            print(self.resource[resource_id])

        if self.trials_state[id]["iteration"] > self.max_iter:
            self.max_iter = self.trials_state[id]["iteration"]
            self.last_run_interval = int(
                (self.stop_iteration - self.max_iter)
                / self.checkpoint_interval
                * self.hyper_num
            )

        if accuracy > self.max_acc:
            self.max_acc = accuracy

        check = 0
        if self.stop_iteration:
            if self.trials_state[id]["iteration"] < self.stop_iteration:
                check += 1
            else:
                check = -9  # 結束條件達成

        if self.stop_acc != 1:
            if accuracy < self.stop_acc:
                check += 1
            else:
                check = -9  # 結束條件達成
        if check > 0:  # 把還需要訓練的id放到scheduler
            # self.trials_scheduler.append(id)
            # self.trials_scheduler = sorted(self.trials_scheduler, reverse=False)
            self.insert_trial(id)
        elif check < 0:
            self.trials_state[id]["resource_id"] = -2  # -2 : 表示訓練結束了
        else:
            print("No end condition!!")  # 沒有設定結束條件
            exit(0)

        self.running_trial_num -= 1
        self.avaliable_resource.append(resource_id)
        self.create_new_trial()  # 創建新的訓練進程

    def insert_trial(self, trial_id):
        new_iteration = self.trials_state[trial_id]["iteration"]

        left, right = 0, len(self.trials_scheduler)
        while left < right:
            mid = (left + right) // 2
            if (
                self.trials_state[self.trials_scheduler[mid]]["iteration"]
                <= new_iteration
            ):
                left = mid + 1
            else:
                right = mid
        self.trials_scheduler.insert(left, trial_id)

    # 查看是否全部都訓練完
    def is_finish(self):
        if len(self.trials_scheduler) + self.running_trial_num == 0:
            return True
        else:
            return False

    # 設定head的指標
    def set_head(self, tuner):
        self.tuner = tuner

        for _ in range(len(self.resource)):
            self.create_new_trial()

    def set_after_mutation(self, id, chosed_id, new_hyper, last_checkpoint):
        self.last_checkpoint[id] = last_checkpoint

        if new_hyper:
            self.perturbs += 1
            self.hypers[id] = new_hyper
            self.checkpoints[id] = copy.deepcopy(self.checkpoints[chosed_id])

            # if self.trials_state[id]["iteration"] > self.trials_state[chosed_id]["iteration"]:
            #     if self.trials_state[id]["iteration"] == self.stop_iteration:
            #         self.trials_scheduler.append(id)
            #         self.trials_scheduler = sorted(self.trials_scheduler, reverse=False)
            #     self.trials_state[id]["iteration"] = self.trials_state[chosed_id]["iteration"]

    def get_for_reporter(self):
        return (
            self.hypers,
            self.trial_acc_list,
            self.trials_state,
            self.perturbs,
            self.running_trial_num,
        )

    def get_start_time(self):
        return self.start_time

    def get_resource(self):
        return self.resource

    def get_best_accuracy(self):
        max_list = list(
            map(self.trial_acc_list.index, heapq.nlargest(1, self.trial_acc_list))
        )
        return max_list[0], self.trial_acc_list[max_list[0]], self.perturbs


# 突變
@ray.remote(num_cpus=0.1, resources={"node:" + HEAD_NODE_IP: 0.1})
def mutation(
    tuner,
    id,
    hypers,
    accuracys,
    last_checkpoint,
    hyperparam_mutations,
    resample_posibility=0.25,
    quantile_fraction=0.25,
):
    lower_quantile, upper_quantile = quantile(accuracys, quantile_fraction)
    if id in upper_quantile:
        last_checkpoint[id] = 1
    else:
        last_checkpoint[id] = 0

    new_hyper = None
    chosed_id = None

    if id in lower_quantile:  # 是否表現很差
        print("--- Exploit ---")
        chosed_id = random.choice(upper_quantile)  # 選出一個優秀的Trial
        print(
            f"Cloning  hyper_{str(chosed_id).zfill(5)} (score : {accuracys[chosed_id]}) to hyper_{str(id).zfill(5)} (score : {accuracys[id]}) \n"
        )
        if last_checkpoint[chosed_id] == 0:
            print(
                f"Hyper_{str(chosed_id).zfill(5)} don't have checkpoint, skip exploit for  hyper_{str(id).zfill(5)}!!"
            )
        else:
            new_hyper = explore(
                id, hypers[chosed_id], hyperparam_mutations, resample_posibility
            )  # 突變

    tuner.set_after_mutation.remote(id, chosed_id, new_hyper, last_checkpoint[id])


# 找出標線優秀跟差的
def quantile(accuracys, quantile_fraction):
    trials = []
    for id, acc in enumerate(accuracys):
        if acc != 0:
            trials.append(id)

    if len(trials) <= 1:
        return [], []

    trials.sort(key=lambda t: accuracys[t])

    # 計算num_trials_in_quantile
    num_trials_in_quantile = int(math.ceil(len(trials) * quantile_fraction))
    if num_trials_in_quantile > len(trials) / 2:
        num_trials_in_quantile = int(math.floor(len(trials) / 2))

    return (trials[:num_trials_in_quantile], trials[-num_trials_in_quantile:])


# 探索新的hyper
def explore(id, hyper, hyperparam_mutations, resample_posibility):
    new_hyper = hyper
    print(f"--- Explore the hyperparameters on  hyper_{str(id).zfill(5)} ---")
    for key, distribution in hyperparam_mutations.items():
        print(f"{key} : {hyper[key]} --- ", end="")
        if isinstance(distribution, list):
            if random.random() < resample_posibility or hyper[key] not in distribution:
                new_hyper[key] = random.choice(distribution)
                print(f"(resample)  --> {new_hyper[key]}")
            else:
                shift = random.choice([-1, 1])
                old_idx = distribution.index(hyper[key])
                new_idx = old_idx + shift
                new_idx = min(max(new_idx, 0), len(distribution) - 1)
                new_hyper[key] = distribution[new_idx]
                print(
                    f"(shift {'left' if shift == -1 else 'right'}) --> {new_hyper[key]}"
                )
        elif isinstance(distribution, tuple):
            if random.random() < resample_posibility:
                new_hyper[key] = random.uniform(distribution[0], distribution[1])
                print(f"(resample)  --> {new_hyper[key]}")
            else:
                mul = random.choice([0.8, 1.2])
                new_hyper[key] = hyper[key] * mul
                print(f"(* {mul})  --> {new_hyper[key]}")
    print()
    return new_hyper


@ray.remote(num_cpus=0.1, resources={"node:" + HEAD_NODE_IP: 0.1})
def save_acc_to_json(id, acc, iter, path):
    jsonFile = open(path + "/" + str(id) + "-accuracy.json", "a")
    data = {
        "iteration": iter,
        "accuracy": acc,
    }
    w = json.dumps(data)  # 產生要寫入的資料
    jsonFile.write(w)  # 寫入資料
    jsonFile.write("\n")  # 寫入資料
    jsonFile.close()


# 會被分配一個hyper，設計訓練與data傳接
@ray.remote
def Trial(tuner, id, hyper, checkpoint):
    start_time = time.time()
    # 建立dataloader
    model_type = hyper.get("model_type", "resnet-18")
    train_loader, test_loader = get_data_loader(
        model_type, hyper.get("batch_size", 512)
    )  # 建立訓練資料的loader

    if model_type == "resnet-18":
        # 建立model
        model = models.resnet18()
        # 修改全连接层的输出
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif model_type == "resnet-50":
        # 建立model
        model = models.resnet50()
        # 修改全连接层的输出
        model.fc = nn.Linear(model.fc.in_features, 100)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        for k, v in checkpoint["model_state_dict"].items():
            checkpoint["model_state_dict"][k] = v.cuda()
        # Convert to CPU
        for state in checkpoint["optimizer_state_dict"]["state"].values():
            for k, v in state.items():
                state[k] = v.cuda()
    else:
        device = torch.device("cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = optim.SGD(
        model.parameters(),
        lr=hyper.get("lr", 0.01),
        momentum=hyper.get("momentum", 0.9),
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    checkpoint["model_state_dict"] = None
    checkpoint["optimizer_state_dict"] = None
    for param_group in optimizer.param_groups:
        if "lr" in hyper:
            param_group["lr"] = hyper["lr"]
        if "momentum" in hyper:
            param_group["momentum"] = hyper["momentum"]

    # 開始訓練
    for _ in range(checkpoint["checkpoint_interval"]):
        train(model, optimizer, train_loader, device)

    # 獲得accuracy
    acc = test(model, test_loader, device)

    # 開始儲存結果
    run_time = time.time() - start_time
    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if (
        torch.cuda.is_available()
    ):  # 如果是GPU，把model與optimizer放入cpu，因為Checkpoint在CPU
        # Convert to CPU
        for k, v in checkpoint["model_state_dict"].items():
            checkpoint["model_state_dict"][k] = v.cpu()
        # Convert to CPU
        for state in checkpoint["optimizer_state_dict"]["state"].values():
            for k, v in state.items():
                state[k] = v.cpu()

    tuner.report_before_trial_end.remote(
        id, acc, run_time, checkpoint
    )  # 把結果傳回Tuner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_times", type=int, default=1)  # 重複實驗幾次
    args = parser.parse_args()

    dir_path = os.path.expanduser("~/Documents/workspace/tune_population_based/")

    # Ray 初始化
    runtime_env = {
        "working_dir": dir_path,
        "excludes": ["data/", "my_model/", "ray_results/", "pytorch-cifar/"],
    }

    # 輸出使用資源
    out_result = open(dir_path + "Running_Results.txt", "a+")
    out_result.write("+---------------+---------------+\n")
    out_result.write(f"{time.ctime()}  <<Our Results - {__file__}>> \n")
    out_result.write(f"Hyper_num = {HYPER_NUM} \n")
    out_result.write(f"Stop iteration = {STOP_ITER} \n")
    out_result.write(f"Stop accuracy = {STOP_ACC} \n")
    out_result.write(f"Checkpoint interval = {INTERVAL_CHECK} \n")
    out_result.write(f"Batch size = {BATCH_SIZE} \n")
    out_result.write(f"Resource allocation: {RESOURCE_ALLOCATION} \n")
    out_result.close()

    model_types = ["resnet-18", "resnet-50"]
    # model_types = ["resnet-50"]

    for model in model_types:
        out_result = open(dir_path + "Running_Results.txt", "a+")
        out_result.write(f"model_type: {model} \n")
        out_result.close()

        avg_run_time = 0
        avg_accuracy = 0

        for _ in range(args.exp_times):
            if ray.is_initialized():
                ray.shutdown()
            ray.init(
                address="ray://" + HEAD_NODE_IP + ":10001", runtime_env=runtime_env
            )
            # ray.init(address="auto", runtime_env=runtime_env)
            print(ray.available_resources())
            tt = time.ctime()
            tt_tmp = tt.split()
            json_path = (
                dir_path
                + "results/"
                + tt_tmp[-1]
                + "-"
                + tt_tmp[-4]
                + "-"
                + tt_tmp[-3]
                + "-"
                + tt_tmp[-2]
                + "/"
            )
            os.makedirs(json_path)
            print(f"{json_path = }")

            # 建立Tuner
            tuner_head = Tuner.remote(
                hyper_num=HYPER_NUM,
                model_type=model,
                resource_allocation=RESOURCE_ALLOCATION,
                stop_acc=STOP_ACC,
                stop_iteration=STOP_ITER,
                checkpoint_interval=INTERVAL_CHECK,
                path=json_path,
                hyperparam_mutations={
                    "lr": (0.0001, 1),
                    "momentum": [0.8, 0.9, 0.99],
                    # "batch_size" : [32, 64, 128, 256, 512],
                },
            )

            tuner_head.set_head.remote(tuner_head)

            # 建立Reporter
            Reporter.remote(
                tuner_head,
                max_report_frequency=INTERVAL_REPORT,
                hyper_num=HYPER_NUM,
            )

            # 卡在這裡，直到全部訓練結束
            while not ray.get(tuner_head.is_finish.remote()):
                time.sleep(1)

            max_acc_index, max_acc, perturbs = ray.get(
                tuner_head.get_best_accuracy.remote()
            )

            start_time = ray.get(tuner_head.get_start_time.remote())
            avg_run_time += time.time() - start_time
            avg_accuracy += max_acc
            resource = ray.get(tuner_head.get_resource.remote())

            # 輸出結果
            out_result = open(dir_path + "Running_Results.txt", "a+")
            out_result.write(f"Resource results: {resource} \n")
            out_result.close()
            ray.shutdown()
            time.sleep(10)

        # 輸出結果
        out_result = open(dir_path + "Running_Results.txt", "a+")
        out_result.write(f"Avg_total_runtime : {avg_run_time / args.exp_times} \n")
        out_result.write(f"Avg_accuracy : {avg_accuracy / args.exp_times} \n\n")
        out_result.close()
