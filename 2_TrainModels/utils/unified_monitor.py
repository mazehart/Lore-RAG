import torch
import os
import pynvml
import psutil
import time
import threading
from typing import List, Tuple, Optional


class UnifiedMonitor:
    """统一监视器类，同时监控GPU显存、CPU和内存使用情况"""
    
    def __init__(self, device_id="0"):
        """
        初始化统一监视器
        
        Args:
            device_id: CUDA设备ID，默认为"0"
        """
        self.device_id = device_id
        self.gpu_handle = None
        self.monitoring = False
        self.monitor_thread = None
        self.pid = os.getpid()
        
        # GPU监控数据
        self.gpu_memory_records = []
        
        # 资源监控数据
        self.cpu_records = []
        self.memory_records = []
        
        self.start_time = None
        
    def setup(self):
        """设置CUDA、NVML和资源监控环境"""
        # GPU设置
        print(f"原始CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}")
        os.environ["CUDA_VISIBLE_DEVICES"] = self.device_id
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用，请确保已分配GPU资源！")
        
        print(f'CUDA是否可用: {torch.cuda.is_available()}')
        print(f'当前CUDA设备: {torch.cuda.current_device()}')
        print(f'GPU名称: {torch.cuda.get_device_name(0)}')
        
        # 初始化NVML
        pynvml.nvmlInit()
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', str(self.device_id))
        first_visible = str(visible).split(',')[0].strip()
        nvml_index = int(first_visible) if first_visible != '' else 0
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
        print(f'NVML初始化成功，绑定物理GPU索引: {nvml_index}')
        
        # 资源信息
        print(f'CPU核心数: {psutil.cpu_count()}')
        print(f'总内存: {psutil.virtual_memory().total / 1024**3:.2f} GB')
        print(f'统一监控初始化成功')
    
    def start(self, interval=0.01):
        """
        开始统一监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if self.monitoring:
            print("警告: 统一监控已经在运行中")
            return
        
        self.monitoring = True
        self.gpu_memory_records = []
        self.cpu_records = []
        self.memory_records = []
        self.start_time = time.time()
        
        def monitor_loop():
            """监控循环"""
            while self.monitoring:
                current_time = time.time()
                
                # GPU显存监控（仅记录当前进程在所有GPU上的显存占用总和，单位MB）
                used_by_self = 0
                device_count = pynvml.nvmlDeviceGetCount()
                for idx in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses_v2(handle)
                    for p in procs:
                        if p.pid == self.pid:
                            used_by_self += int(p.usedGpuMemory)
                gpu_memory_mb = used_by_self / 1024**2
                
                # CPU监控
                cpu_percent = psutil.cpu_percent(interval=None)
                
                # 进程内存监控（RSS，基于当前进程）
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024**2  # 转换为MB
                
                # 记录所有数据
                self.gpu_memory_records.append((current_time, gpu_memory_mb))
                self.cpu_records.append((current_time, cpu_percent))
                self.memory_records.append((current_time, memory_mb))
                
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.start()
        print(f"统一监控已开始")
    
    def end(self) -> Tuple[List[float], List[float], List[float], float]:
        """
        结束统一监控
        
        Returns:
            Tuple[List[float], List[float], List[float], float]: (GPU显存列表, CPU列表, 内存列表, 运行时间)
        """
        if not self.monitoring:
            print("警告: 统一监控未在运行")
            return [], [], [], 0.0
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        end_time = time.time()
        run_time = end_time - self.start_time
        
        # 提取所有监控数据
        gpu_memory_list = [record[1] for record in self.gpu_memory_records]
        cpu_list = [record[1] for record in self.cpu_records]
        memory_list = [record[1] for record in self.memory_records]
        
        # 清除历史缓存
        self.gpu_memory_records = []
        self.cpu_records = []
        self.memory_records = []
        self.start_time = None
        
        print(f"统一监控已结束，运行时间: {run_time:.2f} 秒，记录点数: {len(gpu_memory_list)}")
        
        return gpu_memory_list, cpu_list, memory_list, run_time
    
    def __del__(self):
        """析构函数，确保资源被清理"""
        if self.monitoring:
            self.end()
        pynvml.nvmlShutdown()
