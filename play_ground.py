import torch
import numpy as np

seed = 55
np.random.seed(seed)
torch.manual_seed(seed)

T = 10
x = torch.randn(2, 32, 32, 3)
batch_size = x.shape[0]
t = torch.randint(low=0, high=T, size=(batch_size,))
print(t)

beta = torch.linspace(1e-4, 0.02, T)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, 0)

print('beta, alpha, alpha_bar')
print(beta)
print(alpha)
print(alpha_bar)


abar = torch.gather(alpha_bar, 0, t).view((batch_size, 1, 1, 1))
print(abar)
