import subprocess as sp
import os
# from tensorflow.config import experimental
import torch

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    print(memory_free_values)

# get_gpu_memory()
# print(experimental.get_memory_info('DEVICE_NAME'))

print(torch.cuda.get_device_properties(0).total_memory)
print(torch.cuda.get_device_properties(1).total_memory)

'''

addqueue -q cmbgpu --gpus 1 --gputype rtx3090with24gb -g '1GPU' /usr/bin/python3 ./check_GPU_memory.py
addqueue -q cmbgpu --gpus 2 --gputype rtx3090with24gb -g '2GPU' /usr/bin/python3 ./check_GPU_memory.py

addqueue -q cmbgpu --gpus 1 -n 1x2 -s --gputype rtx3090with24gb -g '1GPU1x2' /usr/bin/python3 ./check_GPU_memory.py
addqueue -q cmbgpu --gpus 1 -n 2x1 -s --gputype rtx3090with24gb -g '1GPU2x1' /usr/bin/python3 ./check_GPU_memory.py
addqueue -q cmbgpu --gpus 1 -n 2x2 -s --gputype rtx3090with24gb -g '1GPU2x2' /usr/bin/python3 ./check_GPU_memory.py

'''