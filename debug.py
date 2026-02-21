import torch
import sys
import subprocess

print("=== SYSTEM & PYTHON INFO ===")
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
print(f"PyTorch Built with CUDA: {torch.version.cuda}")

print("\n=== GPU HARDWARE INFO ===")
if torch.cuda.is_available():
    print(f"CUDA is available: YES")
    print(f"GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"Device [{i}]: {props.name}")
        print(f"Compute Capability: {props.major}.{props.minor}")
else:
    print("CUDA is available: NO")

print("\n=== PYTORCH CAPABILITIES ===")
print(f"Compiled Architectures (sm_XX): {torch.cuda.get_arch_list()}")

print("\n=== DRIVER INFO ===")
try:
    smi = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']
    ).decode().strip()
    # If multiple GPUs, it might return multiple lines, just grab the first
    print(f"NVIDIA Driver Version: {smi.split()[0]}")
except Exception as e:
    print(f"Could not read driver: {e}")