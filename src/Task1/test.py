import torch

v = torch.Tensor([[1, 2, 1],
                  [4, 5, 6]])
r = torch.squeeze(v)
print(r.size())

t = torch.ones(20, 10, 2, 1)
r = torch.squeeze(t, 1)
print(r.size())