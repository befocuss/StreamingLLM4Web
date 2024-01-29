import torch

sample = torch.ones(1,40,583,128)
final = sample[:,:,-417:583,:]
print(final.size())