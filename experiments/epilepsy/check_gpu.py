import platform, torch
print(platform.python_version(), torch.__version__, torch.version.cuda,
      torch.cuda.is_available(), torch.cuda.get_device_name(0))
