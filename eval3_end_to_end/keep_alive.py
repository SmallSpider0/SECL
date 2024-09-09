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
            if cmdline and "train_keep_alive.py" in cmdline:
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return count


def run_script_in_background(script_name, paras=""):
    command = f"nohup /root/miniconda3/envs/py38/bin/python {script_name} {paras} > /dev/null 2>&1 &"
    print(command)
    subprocess.Popen(command, shell=True)


def stop_all_python_processes():
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "train_keep_alive.py" in process.info["cmdline"]:
                print(f"Terminating process {process.info['pid']}...")
                process.terminate()
                process.wait(timeout=5)  # 等待进程终止
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass


def start_eval():
    # 允许同时运行的最大进程数
    max_running_processes = 1
    while True:
        # 等待直到当前运行的 train.py 进程数量少于最大允许数量
        while get_running_python_processes_count() >= max_running_processes:
            time.sleep(5)  # 等待 5 秒钟再检查

        # 启动新的 train.py 脚本
        run_script_in_background(
            "train_keep_alive.py"
        )
        time.sleep(1)


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
