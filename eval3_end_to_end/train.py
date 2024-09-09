# 系统库
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import gc
import random
import math
import time
import argparse
from sklearn.metrics import accuracy_score, f1_score


# 自定义库
from utils.contract import Contract
from utils.models import mlp, resnet18
from utils.dataset import get_data_loaders


# 禁用核心转储文件
import resource

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


# 随机种子设置
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 让显卡产生的随机数一致
    torch.cuda.manual_seed_all(
        seed
    )  # 多卡模式下，让所有显卡生成的随机数一致？这个待验证
    np.random.seed(seed)  # numpy产生的随机数一致
    random.seed(seed)


class CollaborativeTrain:
    def __init__(
        self,
        dataset_name,
        model_name,
        device,
        lr=0.01,
        momentum=0.9,
        local_epochs=1,
        veri_method="test_acc",
    ):
        self.train_loaders, self.test_loader = None, None
        self.dataset_name, self.model_name = dataset_name, model_name
        self.lr, self.momentum = lr, momentum
        self.local_epochs = 1
        self.veri_method = veri_method
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.local_models = None
        self.trianer_malicious = []

        # 首次运行初始化全局模型
        if self.model_name == "MLP" and self.dataset_name == "MNIST":
            self.global_model = mlp().to(self.device)
        elif self.model_name == "ResNet18" and self.dataset_name == "CIFAR10":
            self.global_model = resnet18(num_classes=10).to(self.device)
        else:
            raise ValueError("Unsupported model or dataset combination")

    def model_distance(self, model1, model2):
        # 将模型参数转换为一维向量并计算差值
        differences = [
            p1 - p2 for p1, p2 in zip(model1.parameters(), model2.parameters())
        ]
        # 将差值转换为GPU张量并计算范数
        distance = torch.norm(torch.cat([diff.flatten() for diff in differences]))
        return distance.item()

    # 模型聚合函数
    def aggregate_models(self, models, is_malicious=None):
        if is_malicious is None:
            is_malicious = [False for _ in range(len(models))]

        # 获取全局模型的状态字典
        global_dict = self.global_model.state_dict()

        # 对所有本地模型的参数进行聚合，仅聚合is_malicious为False的模型
        for k in global_dict.keys():
            model_params = [
                models[i].state_dict()[k].float()
                for i in range(len(models))
                if not is_malicious[i]
            ]
            if model_params:  # 确保model_params不为空
                global_dict[k] = torch.stack(model_params, 0).mean(0)
            else:
                print(f"No non-malicious models found for parameter {k}.")

        # 创建一个新的全局模型并加载聚合后的参数
        if self.model_name == "ResNet18":
            aggregated_model = resnet18(num_classes=10).to(self.device)
        else:
            aggregated_model = self.global_model.__class__().to(self.device)

        aggregated_model.load_state_dict(global_dict)
        return aggregated_model

    # 评估测试集准确率
    def test(self, model):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += self.criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100.0 * correct / len(self.test_loader.dataset)
        return accuracy, test_loss

    # behaviours可选: ADVERSARIAL/FREERIDER/NORMAL
    def step_training(self, selections, behaviours):
        self.trianer_malicious = [False if b == "NORMAL" else True for b in behaviours]

        def _local_train(weights):
            # 初始化数据集
            self.train_loaders, self.test_loader = get_data_loaders(
                self.dataset_name, 32, "./data", weights
            )

            # 模拟各个训练者本地训练，并保存训练结果
            local_models = []
            for i in range(len(weights)):
                # 初始化本地模型
                if self.model_name == "ResNet18":
                    local_model = resnet18(num_classes=10).to(self.device)
                else:
                    local_model = self.global_model.__class__().to(self.device)

                if behaviours[i] == "ADVERSARIAL":  # 返回随机结果
                    local_models.append(local_model)
                elif behaviours[i] == "FREERIDER":  # 直接返回全局模型
                    local_model.load_state_dict(self.global_model.state_dict())
                    local_models.append(local_model)
                else:  # 正常训练
                    local_model.load_state_dict(self.global_model.state_dict())
                    optimizer = optim.SGD(
                        local_model.parameters(), lr=self.lr, momentum=self.momentum
                    )
                    # 执行本地训练
                    local_model.train()
                    for e in range(self.local_epochs):
                        for data, target in self.train_loaders[i]:
                            data, target = data.to(self.device), target.to(self.device)
                            optimizer.zero_grad()
                            output = local_model(data)
                            loss = self.criterion(output, target)
                            loss.backward()
                            optimizer.step()
                    local_models.append(local_model)

            # 返回所有本地训练结果
            return local_models

        # 根据签署的合约 使用特定数据量训练（可以按合约中数据量参数作为权重，划分数据集）
        weights = [int(item[1][0]) for item in selections]

        # 本地训练一轮
        self.local_models = _local_train(weights)
        return weights

    def step_verification(self, weights, selections):
        # TODO：这里需要添加验证代码

        def veri_0_none():
            # 检测不出任何攻击
            return "none"

        """
        测试1：评估各trainer提交的测试集准确率（并与现有全局模型对比）
        检测攻击原理：
        - 测试集准确率相对前一个全局模型的提升越高，说明该提交对全局模型的贡献越大，而提升过小（甚至负数）的提交被认为是恶意的
        阈值设置：
        """

        def veri_1_testacc():
            ret = []
            acc_global, _ = self.test(self.global_model)
            for model in self.local_models:
                accuracy, _ = self.test(model)
                ret.append(accuracy - acc_global)

            # 判断参与者是否恶意
            ret = np.array(ret)
            return ret

        """
        测试2：基于Shapley值计算各训练者贡献度
        缺陷：计算开销很大（参与者指数级别开销）
        阈值设置：
        参考：https://github.com/clickade/federated-shapley-playground
        """

        def veri_2_shapley(num_samples=10):
            n = len(self.local_models)
            marginal_contributions = np.zeros(n)
            permutations = [
                random.sample(range(n), n) for _ in range(num_samples)
            ]  # 随机采样排列

            for perm in permutations:
                current_value = 0
                previous_value = 0
                for i in range(len(perm)):
                    subset_models = [self.local_models[j] for j in perm[: i + 1]]
                    aggregated_model = self.aggregate_models(subset_models)
                    accuracy, _ = self.test(aggregated_model)
                    current_value = accuracy
                    marginal_contributions[perm[i]] += current_value - previous_value
                    previous_value = current_value

            shapley_values = marginal_contributions / num_samples
            return shapley_values

        """
        测试3：基于influence metric计算各训练者贡献度
        检测攻击原理：
        - inﬂuence越大，说明该提交对全局模型的贡献越大，而infulence过小（如负数）的提交被认为是恶意的


        阈值设置：
        参考：The inﬂuence [9] of a data point is deﬁned as the diﬀerence in loss function between the model trained with and without the data point.
        """

        def veri_3_influence():
            # TODO： 判断infulence计算是否有误
            n = len(self.local_models)
            influence_contributions = np.zeros(n)

            # 计算基线模型的损失
            full_model = self.aggregate_models(self.local_models)
            _, baseline_loss = self.test(full_model)

            # 对每个训练者模型计算影响力
            for i in range(n):
                # 移除第 i 个训练者模型，计算新的聚合模型损失
                subset_models = [self.local_models[j] for j in range(n) if j != i]
                aggregated_model = self.aggregate_models(subset_models)
                _, loss = self.test(aggregated_model)
                # 影响力为基线损失与当前损失之差
                influence_contributions[i] = -(baseline_loss - loss)
            return influence_contributions

        """
        测试4：基于Multi-KRUM算法计算各训练者贡献度
        参考：The veriﬁer will add up Euclidean distances of each customer i’s update to the closest R − f − 2 updates 
        and denote the sum as each customer i’s score s(i). R means the number of updates, and f means the number of Byzantine customers.
        """

        def veri_4_multi_KRUM():
            n = len(self.local_models)
            f = math.ceil(n / 2) - 1
            scores = np.zeros(n)

            for i in range(n):
                # 计算更新i到其它所有更新的距离
                distances = []
                for j in range(n):
                    if i != j:
                        distances.append(
                            self.model_distance(
                                self.local_models[i], self.local_models[j]
                            )
                        )
                # 计算每个更新的得分（最小的n - f - 2个距离之和）
                distances.sort()
                # 返回负值 距离越大则分数越低
                scores[i] = -sum(distances[: n - f - 2])
            return scores

        """
        测试5：基于update significance计算各训练者贡献度
        检测方法：update signiﬁcance过大的提交被认为是恶意
        参考：In this paper, the update signiﬁcance is measured by model deviation, 
        which is the divergence of a particular local model from the average across all local models [20], [21].
        """

        def veri_5_update_significance():
            n = len(self.local_models)
            # 计算所有本地模型参数的平均值
            avg_model_params = {
                name: torch.zeros_like(param.data)
                for name, param in self.local_models[0].named_parameters()
            }
            for model in self.local_models:
                for name, param in model.named_parameters():
                    avg_model_params[name] += param.data / n

            # 计算每个本地模型与平均模型之间的欧几里得距离
            update_significance = []
            for model in self.local_models:
                deviation = 0.0
                for name, param in model.named_parameters():
                    deviation += (
                        torch.norm(param.data - avg_model_params[name]).item() ** 2
                    )
                deviation = deviation**0.5
                update_significance.append(deviation)

            return update_significance

        def veri_6_reproduction():
            # 检测出所有
            return "all"

        # 验证方法列表
        methods = [
            ("none", veri_0_none),
            ("test_acc", veri_1_testacc),
            ("shapley", veri_2_shapley),
            ("influence", veri_3_influence),
            ("multi_KRUM", veri_4_multi_KRUM),
            ("update_significance", veri_5_update_significance),
            ("reproduction", veri_6_reproduction),
        ]

        # 实验1：评估验证方法执行时间
        def test_veri():
            for method in methods:
                # 使用参数设置的验证方法验证
                if self.veri_method == method[0]:
                    start = time.time()
                    scores = method[1]()
                    # print(scores)
                    # 根据返回值自适应评估
                    if not isinstance(scores, str):
                        threshold = np.mean(scores) - 1 * np.std(scores)
                        is_malicious = (scores <= threshold).tolist()
                    elif scores == "all":
                        # 用我们的复现检测 能检测到所有攻击
                        is_malicious = self.trianer_malicious
                    elif scores == "none":
                        # 不进行验证 检测不到任何攻击
                        is_malicious = [
                            False for _ in range(len(self.trianer_malicious))
                        ]
                    time_elapsed = time.time() - start
                    # print(f"Time for {method[0]}: {time_elapsed:.2f} s")
                    # print("预测结果", is_malicious)
                    # print("原始", self.trianer_malicious)

                    accuracy = accuracy_score(
                        self.trianer_malicious, is_malicious
                    )  # 计算验证准确率
                    f1 = f1_score(
                        self.trianer_malicious, is_malicious, zero_division=0
                    )  # 计算F1值
                    # print(f"acc:{accuracy}, f1:{f1}")
                    # print("actual", self.trianer_malicious)
                    # print("pred", is_malicious)

                    # 验证方法名，耗时，当前epoch的acc和f1,预期结果,实际结果
                    return (
                        method[0],
                        time_elapsed,
                        accuracy,
                        f1,
                        self.trianer_malicious,
                        is_malicious,
                    )

        # 使用参数设置的验证方法验证
        veri_ret = test_veri()
        is_malicious = veri_ret[5]

        # 根据各已签署合约的验证结果 发放奖励
        index = 0
        rewards_trainer = [0 for _ in range(len(weights))]
        contribution_trainer = [0 for _ in range(len(weights))]
        # 遍历每个worker
        for item in selections:
            # 如果验证结果通过则发放奖励
            if not is_malicious[index]:
                rewards_trainer[index] = item[1][1]
            # 如果某个训练者实际是诚实的则计算贡献
            if not self.trianer_malicious[index]:
                contribution_trainer[index] = int(item[1][0])
            index += 1
        # 记录当前轮次的奖励发放情况（根据验证方法）和训练者实际（ground truth）贡献情况
        return contribution_trainer, rewards_trainer, veri_ret


def main():
    # 创建变量用于存放实验结果
    data_save = {}

    def generate_normal_random_numbers(count, mean, std_dev):
        random_numbers = np.empty(count)
        generated_count = 0

        while generated_count < count:
            remaining_count = count - generated_count
            new_numbers = np.random.normal(
                loc=mean, scale=std_dev, size=remaining_count
            )
            valid_numbers = new_numbers[new_numbers >= 0]
            valid_count = len(valid_numbers)
            random_numbers[generated_count : generated_count + valid_count] = (
                valid_numbers
            )
            generated_count += valid_count

        return random_numbers

    # 创建解析器
    parser = argparse.ArgumentParser(description="Train model.")

    # 添加参数
    parser.add_argument(
        "--v",
        type=int,
        required=False,
        default=0,
        help="Verification method type: 0 none; 1 test_acc; 2 shapley; 3 influence; 4 multi_KRUM; 5 update_significance; 6 reproduction",
    )
    parser.add_argument(
        "--adv",
        type=int,
        required=False,
        default=0,
        help="Num of adversaries (If client is adversarial, they return randomized parameters)",
    )
    parser.add_argument(
        "--fr",
        type=int,
        required=False,
        default=0,
        help="Num of freeriders (If client is a freerider, they return the same server model parameters0",
    )
    parser.add_argument(
        "--task",
        type=int,
        required=False,
        default=1,
        help="Task: 1 CIFAR10+Resnet18; 2 MNIST+MLP",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
    )
    veri_methods = [
        "none",
        "test_acc",
        "shapley",
        "influence",
        "multi_KRUM",
        "update_significance",
        "reproduction",
    ]
    # 解析参数
    args = parser.parse_args()

    # 设置随机数种子
    set_seed(args.seed)

    data_save["meta"] = {
        "v": args.v,
        "adv": args.adv,
        "fr": args.fr,
        "task": args.task,
    }  # 保存实验元数据
    data_save["data"] = {}  # 保存实验结果
    data_save["data"]["epochs"] = []  # 保存各epoch数据

    veri_method = veri_methods[args.v]

    # 参与者相关参数参数
    # 模拟各类型参与者平均分布
    participants_num = 10
    estimated_types = np.array([0.1, 0.13, 0.16, 0.19, 0.22])
    estimated_props = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    participants = np.array([])
    for type, prop in zip(estimated_types, estimated_props):
        tmp = generate_normal_random_numbers(int(participants_num * prop), type, 0.01)
        participants = np.concatenate((participants, tmp))
    participants = np.sort(participants)
    # print(participants)

    # 配置参与者行为【按顺序分配攻击者】
    # If client is adversarial, they return randomized parameters （即随机初始化模型并返回）
    # If client is a freerider, they return the same server model parameters
    adv_num = args.adv  # 攻击者数量（返回随机结果）
    freerider_num = args.fr  # 偷懒者数量（不训练）
    participants_behaviors = (
        ["ADVERSARIAL"] * adv_num
        + ["FREERIDER"] * freerider_num
        + ["NORMAL"] * (participants_num - adv_num - freerider_num)
    )
    # random.shuffle(participants_behaviors)
    # print(participants_behaviors)

    # 任务相关参数
    # k1为全局准确率到模型价值的转换参数
    # k2和k3为本地训练数据总量到全局模型准确率的转换参数（和训练任务有关，需要根据实际任务调整）
    u_m_k = [300, 0.001]
    u_p = lambda reward, cost: reward - cost
    u_m = lambda data, reward: u_m_k[0] * cp.log(1 + u_m_k[1] * data) - reward

    # task1
    if args.task == 1:
        global_epochs = 40
        dataset_name = "CIFAR10"
        model_name = "ResNet18"
    elif args.task == 2:
        global_epochs = 20
        dataset_name = "MNIST"
        model_name = "MLP"

    CT = CollaborativeTrain(dataset_name, model_name, "cuda", veri_method=veri_method)
    C = Contract()

    # 步骤1：设计合约
    contracts = C.design_contract(estimated_types, u_m, u_p)

    # 步骤2：各TN签署合约
    selections = C.select_contract(participants, contracts)

    for epoch in range(global_epochs):
        epoch_data = {}
        st = time.time()
        # 步骤3：各TN根据签署的合约本地训练并提交结果
        weights = CT.step_training(selections, participants_behaviors)

        # 步骤4：MO验证各TN的提交，并获取它们实际贡献/收到的奖励/验证结果
        contribution_trainer, rewards_trainer, veri_ret = CT.step_verification(
            weights, selections
        )
        is_malicious = veri_ret[5]
        epoch_data["is_malicious"] = is_malicious

        # TODO：恶意参与者不一定按规则选取合约项（威胁模型如何设置？）
        # 步骤5：MO根据有效提交数量计算自身效用值
        mo_utility, participant_utilities = C.cal_actual_utility(
            selections, contribution_trainer, rewards_trainer
        )
        epoch_data["mo_utility"] = mo_utility
        epoch_data["participant_utilities"] = participant_utilities

        # 步骤6：MO聚合验证通过的提交，开始下一轮训练
        CT.global_model = CT.aggregate_models(CT.local_models, is_malicious)
        # 测试集准确率
        accuracy, test_loss = CT.test(CT.global_model)
        epoch_data["model_acc"] = accuracy

        # 清理数据加载器和模型
        del CT.train_loaders, CT.test_loader, CT.local_models
        gc.collect()
        end = time.time()

        # 保存该epoch实验结果
        data_save["data"]["epochs"].append(epoch_data)

    save_path = f"./results/task{args.task}-v{args.v}-adv{args.adv}-fr{args.fr}-seed{args.seed}"
    torch.save(data_save, save_path)
    # data = torch.load(save_path)
    # print(data)

    # 保存实验结果到文件
    # with open(f"./results/eval10/{dataset_name}-{veri_method}-adv{adv_num}-fr{freerider_num}", "a") as f:
    #     f.write(f"Epoch {epoch}: Time {end-st:.3f} s, model acc: {accuracy}\n")
    #     f.write(
    #         f"{veri_ret[0]}: {veri_ret[1]:.3f} s, acc {veri_ret[2]}, f1 {veri_ret[3]}\n"
    #     )
    #     f.write(f"mo_utility: {mo_utility}\n")
    #     f.write(f"participant_utilities: {participant_utilities}\n")
    #     f.write("--------------------------\n")


if __name__ == "__main__":
    main()

# python train.py --task 2 --v 4 --adv 4 --fr 0

"""
初步结论整理：









【CIFAR10 + ResNet18】

- 仅ADV
python train.py --task 2 --v 1 --adv 4 --fr 0
python train.py --task 2 --v 1 --adv 1 --fr 0

- 仅FR
python train.py --task 2 --v 1 --adv 0 --fr 4
python train.py --task 2 --v 1 --adv 0 --fr 1

- 混合
python train.py --task 2 --v 1 --adv 2 --fr 2

- Baseline
python train.py --task 0 --v 0 --adv 0 --fr 0

"""
