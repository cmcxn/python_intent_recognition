import torch
print(torch.__version__)       # 应该没有 +cpu 后缀
print(torch.version.cuda)      # 应该显示 12.1
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # 应该显示 RTX A4000 或 Tesla P4
