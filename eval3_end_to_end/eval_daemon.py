import subprocess
import psutil
import time
from datetime import datetime
import argparse


def get_running_python_processes_count():
    count = 0
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = process.info["cmdline"]
            if cmdline and "train.py" in cmdline:
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return count


def run_script_in_background(script_name, paras):
    command = f"nohup /root/miniconda3/envs/py38/bin/python {script_name} {paras} > /dev/null 2>&1 &"
    subprocess.Popen(command, shell=True)


def stop_all_python_processes():
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "train.py" in process.info["cmdline"]:
                print(f"Terminating process {process.info['pid']}...")
                process.terminate()
                process.wait(timeout=5)  # 等待进程终止
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass


def start_eval():
    # 配置实验参数组
    # tasks = [2, 1]
    # maricious_nums = [(4, 0), (1, 0), (0, 4), (0, 1), (2, 2)]
    # seeds = [43, 44, 45, 46]
    # paras = []
    # for seed in seeds:
    #     for task in tasks:
    #         # 添加baseline（不验证+不攻击）
    #         paras.append((task, 0, 0, 0, seed))
    #         # 添加各种验证方法和攻击强度的组合
    #         for veri_method in [0, 1, 2, 3, 4, 5, 6]:
    #             for maricious_num in maricious_nums:
    #                 paras.append(
    #                     (task, veri_method, maricious_num[0], maricious_num[1], seed)
    #                 )

    tasks = [2, 1]
    maricious_nums = [(0, 0)]
    seeds = [42]
    paras = []
    for seed in seeds:
        for task in tasks:
            # 添加各种验证方法和攻击强度的组合
            for veri_method in [1, 2, 3, 4, 5, 6]:
                for maricious_num in maricious_nums:
                    paras.append(
                        (task, veri_method, maricious_num[0], maricious_num[1], seed)
                    )

    # 运行所有实验
    # 允许同时运行的最大进程数
    max_running_processes = 8
    for para in paras:
        # 等待直到当前运行的 train.py 进程数量少于最大允许数量
        while get_running_python_processes_count() >= max_running_processes:
            time.sleep(5)  # 等待 5 秒钟再检查

        # 启动新的 train.py 脚本
        run_script_in_background(
            "train.py", f"--task {para[0]} --v {para[1]} --adv {para[2]} --fr {para[3]} --seed {para[4]}"
        )
        # 获取当前日期和时间
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{formatted_datetime}: 启动 train.py 脚本，参数: --task {para[0]} --v {para[1]} --adv {para[2]} --fr {para[3]} --seed {para[4]}"
        )
        time.sleep(1)

    print("所有脚本启动命令已发出。")


def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description="Train model.")

    # 添加参数
    parser.add_argument(
        "--m",
        type=int,
        required=False,
        default=0,
        help="Mode: 0 start; 1 stop",
    )

    # 解析参数
    args = parser.parse_args()
    if args.m == 0:
        start_eval()
    else:
        stop_all_python_processes()


if __name__ == "__main__":
    main()

# nohup python eval_daemon.py > /dev/null 2>&1 &
