import psutil
import os

def get_process_memory(unit="MB"):
    """
    获取当前Python进程的物理内存占用
    :param unit: 单位，支持B/KB/MB/GB，默认MB
    :return: 内存占用数值（保留2位小数）
    """
    unit_map = {"B":1, "KB":1024, "MB":1024**2, "GB":1024**3}
    process = psutil.Process(os.getpid())  # 获取当前进程对象
    mem_rss = process.memory_info().rss  # 物理内存占用（RSS，实际使用的内存）
    return round(mem_rss / unit_map[unit], 2)