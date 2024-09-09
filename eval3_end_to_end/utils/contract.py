# 论文【3】中个人贡献 对 模型拥有者收益的转化是线性的！
# （即：多个用户对模型拥有者的贡献 不存在边际收益递减）
# 因此，可针对每个用户类型单独求解最优的合约项

# TODO: 该版本虽然可获取合理的合约，但是 在设计合约时没有考虑 各用户类型的比例（需要考虑吗？）

import cvxpy as cp


class Contract:
    def __init__(self):
        pass

    def _enforce_monotonicity(self, contracts):
        n = len(contracts)

        # 迭代调整合同以确保单调性
        def bunch_and_iron(contracts):
            for i in range(n - 1):
                if contracts[i][0] > contracts[i + 1][0]:
                    contracts[i + 1] = (
                        contracts[i][0],
                        max(contracts[i][1], contracts[i + 1][1]),
                    )
                if contracts[i][1] > contracts[i + 1][1]:
                    contracts[i + 1] = (
                        max(contracts[i][0], contracts[i + 1][0]),
                        contracts[i][1],
                    )
            return contracts

        # 递归地检查和调整合同
        def recursive_adjustment(contracts):
            adjusted = bunch_and_iron(contracts)
            if adjusted == contracts:
                return adjusted
            else:
                return recursive_adjustment(adjusted)

        return recursive_adjustment(contracts)

    def _optimize_contract(self, eta, J_list, eta_list):
        def calculate_reward(J_list, eta_list):
            M = len(J_list)
            R_list = [0] * M  # 初始化最优奖励列表

            # 从成本最高的雇员开始计算
            for m in range(M - 1, -1, -1):
                if m == M - 1:  # 成本最高的雇员
                    R_list[m] = eta_list[m] * J_list[m]
                else:  # 其他雇员
                    R_list[m] = (
                        R_list[m + 1]
                        - eta_list[m] * J_list[m + 1]
                        + eta_list[m] * J_list[m]
                    )
            return R_list[0]

        J_m = cp.Variable(integer=True)  # 定义决策变量
        R_star = calculate_reward(
            [J_m] + J_list, eta_list
        )  # 使用公式获取对于贡献度要求J_m效用最大的奖励
        objective = cp.Maximize(self.u_m(J_m, R_star))  # 定义优化目标

        # IR约束仅对当前类型的参与者生效
        ir_constraint = self.u_p(R_star, eta * J_m) >= 0
        constraints = [J_m >= 1, ir_constraint]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS_BB)

        J_star = max(J_m.value, 1) if prob.status == cp.OPTIMAL else None
        R_star = (
            calculate_reward([J_star] + J_list, eta_list)
            if prob.status == cp.OPTIMAL
            else None
        )

        # if DEBUG:
        #     print(eta, f"优化结果：J_star={J_star}, R_star={R_star}, status={prob.status}")
        return J_star, R_star

    def _verify_ic(self, contracts, estimated_types):
        for eta in estimated_types:
            best_utility = max(
                [self.u_p(contract[1], eta * contract[0]) for contract in contracts]
            )
            chosen_contract = max(
                contracts, key=lambda contract: self.u_p(contract[1], eta * contract[0])
            )
            chosen_utility = self.u_p(chosen_contract[1], eta * chosen_contract[0])
            if chosen_utility != best_utility:
                print(f"类型 {eta} 的参与者选择的合约项并不是最优的")
                return False
        return True

    def design_contract(self, estimated_types, u_m, u_p):
        self.u_m = u_m
        self.u_p = u_p

        contracts = []
        J_list = []  # 目前已经优化完成的贡献度要求（随着优化过程 从左侧添加元素）
        eta_list = []  # 目前已经优化完成的成本值（随着优化过程 从左侧添加元素）

        # 从成本最高的雇员类型开始进行优化
        for eta in reversed(estimated_types):
            eta_list = [eta] + eta_list
            J_star, R_star = self._optimize_contract(eta, J_list, eta_list)
            if J_star and R_star:
                J_list = [J_star] + J_list
                contracts.append((J_star, R_star))
            else:
                return False
        contracts = self._enforce_monotonicity(contracts)

        # 验证合约正确性
        if self._verify_ic(contracts, estimated_types):
            return contracts
        else:
            return False

    def select_contract(self, participants, contracts, method="best"):
        selections = []
        for cost in participants:
            if (
                method == "best"
            ):  # 所有雇员选取能使自己净收益最大化的合约项（即为自己类型设计的合约项）
                best_contract = max(
                    contracts,
                    key=lambda contract: self.u_p(contract[1], cost * contract[0]),
                )
                selections.append((cost, best_contract))
            elif (
                method == "uniform"
            ):  # 所有雇员选取为成本最高类型设计的合约项（确保收益不为负）
                selected_contract = contracts[0]
                selections.append((cost, selected_contract))
            else:
                raise ValueError("Invalid selection method. Choose 'best' or 'random'.")
        return selections

    # def evaluate_effect_with_attacker(
    #     self, selections, attack_ratio=0, attack_succ_rate=0
    # ):
    #     # 【所有】雇员按实际成本选取合约项；但在计算【恶意】雇员效用时，将获取的奖励乗【攻击成功率】比例，且成本为0（模拟一种极端情况？）
    #     # 雇主在计算效用时，将所有【恶意】雇员的贡献项设为0（无论攻击是否成功）
    #     # （乐观假设攻击成功的提交虽然无贡献，但也不会额外降低模型性能；实际情况雇主效用可能更低）

    #     attacker_utilities_all = []
    #     honest_utilities_all = []
    #     model_owner_utility = 0

    #     for cost, num, contract in selections:
    #         # 计算每一类雇员中的恶意雇员数量
    #         num_attackers_in_class = int(num * attack_ratio)
    #         num_honest_in_class = num - num_attackers_in_class

    #         # 恶意雇员效用
    #         attacker_utilities = num_attackers_in_class * (
    #             attack_succ_rate * contract[1]
    #         )
    #         attacker_utilities_all.append(attacker_utilities)

    #         # 诚实雇员效用
    #         honest_utilities = num_honest_in_class * self.u_p(
    #             contract[1], cost * contract[0]
    #         )
    #         honest_utilities_all.append(honest_utilities)

    #         # 模型拥有者效用 (恶意雇员贡献为0)
    #         model_owner_utility += (
    #             num_honest_in_class * self.u_m(contract[0], contract[1])
    #             + num_attackers_in_class * self.u_m(0, contract[1])
    #         ).value

    #     # 计算平均效用
    #     total_attackers = sum([int(num * attack_ratio) for _, num, _ in selections])
    #     total_honest = sum([num - int(num * attack_ratio) for _, num, _ in selections])

    #     avg_attacker_utility = (
    #         sum(attacker_utilities_all) / total_attackers if total_attackers > 0 else 0
    #     )
    #     avg_honest_utility = (
    #         sum(honest_utilities_all) / total_honest if total_honest > 0 else 0
    #     )

    #     return avg_attacker_utility, avg_honest_utility, model_owner_utility

    def evaluate_effect(self, selections, estimated_types, contracts):
        # 1.每一个雇员的效用
        participant_utilities = [
            self.u_p(contract[1], cost * contract[0]) for cost, contract in selections
        ]

        # 2.模型拥有者效用
        model_owner_utility = sum(
            [self.u_m(contract[0], contract[1]) for cost, contract in selections]
        ).value

        # 3.设计时考虑的各雇员类型的效用
        estimated_utilities = {}
        for eta in estimated_types:
            utilities = [
                self.u_p(contract[1], eta * contract[0]) for contract in contracts
            ]
            estimated_utilities[eta] = utilities

        return (
            participant_utilities,
            model_owner_utility,
            estimated_utilities,
        )

    # 计算MO和参与者的实际效用
    # 参数：训练者实际贡献、实际奖励、选取的合约项
    def cal_actual_utility(
        self, selections, contribution_trainer=None, rewards_trainer=None
    ):
        # 解包
        costs = []
        for cost, contract in selections:
            costs.append(cost * contract[0])

        if contribution_trainer is None:
            contribution_trainer = []
            rewards_trainer = []
            for _, contract in selections:
                contribution_trainer.append(contract[0])
                rewards_trainer.append(contract[1])

        # 计算实际效用值
        mo_utility = 0
        participant_utilities = []
        for (cost, data, reward) in zip(costs, contribution_trainer, rewards_trainer):
            # 计算模型拥有者效用
            mo_utility += self.u_m(data, reward).value
            # 计算参与者效用（实际收到的奖励-实际付出的效用）
            participant_utilities.append(self.u_p(reward, cost))
        return mo_utility, participant_utilities
