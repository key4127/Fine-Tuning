import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    
# 检查NCCL相关
print(f"NCCL可用: {torch.cuda.nccl.is_available() if hasattr(torch.cuda, 'nccl') else 'NCCL模块不存在'}")