import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"Is CUDA available? {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Run a quick calculation on the GPU
    device = torch.device("cuda")
    a = torch.ones(3, 3).to(device)
    b = torch.ones(3, 3).to(device)
    print(f"Calculation Result:\n{a + b}")
else:
    print("CUDA is NOT available. Check your drivers and PyTorch version.")