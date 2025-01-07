# example HORN network with manual weights
# Felix Effenberger, July 21, 2023

import torch
import matplotlib.pyplot as plt

from model import HORN

torch.manual_seed(1) # set random seed
torch.set_grad_enabled(False) # no gradient computation

# 1-dim time series as input
num_input = 1
num_hidden = 32
num_output = 10

# hyperparameters
h = 1.0
alpha = 0.04 # excitability
omega_base = 2. * torch.pi / 28. # natural frequency
gamma_base = 0.01 # damping

# oscillation parameters - varying across units
# uncomment to enable
omega_min = 0.5 * omega_base
omega_max = 2.0 * omega_base
omega = torch.rand(num_hidden) * (omega_max - omega_min) + omega_min
gamma_min = 0.5 * gamma_base
gamma_max = 2.0 * gamma_base
gamma = torch.rand(num_hidden) * (gamma_max - gamma_min) + gamma_min

# construct heterogeneous HORN
model = HORN(num_input, num_hidden, num_output, h, alpha, omega, gamma)
model.eval()

# set input weights
model.i2h.weight[:, :] = torch.randn(num_hidden, 1)
model.i2h.bias[:] = 0

# set recurrent weights
model.h2h.weight[:, :] = torch.randn(num_hidden, num_hidden) * 0.1
model.i2h.bias[:] = 0

# show weight matrix - diagonal terms are feedback connections
plt.matshow(model.h2h.weight)
plt.colorbar()
plt.title('W_hh')

# compute dynamics for 1000 time steps
domain = torch.arange(1000)

# stimulus shape: batch x input dimension x input length, i.e. 1 x 1 x 1000
stimulus = torch.sin(domain * torch.pi * 2 / 60).unsqueeze(0).unsqueeze(0)
stimulus[0, 0, 500:] = 0 # not stimulus for last 500 time steps
stimulus = stimulus.permute(2, 0, 1) # shape: time steps x batch x input dimension

# run model dynamics
random_init = 1.0 # set to None for x_0 and y_0 = 0
out = model.forward(stimulus, random_init = random_init, record = True)
x_t = out['rec_x_t'] # get amplitudes - shape batch x time steps x unit
y_t = out['rec_y_t'] # get velocities

# show amplitude dynamics
plt.figure()
for i in range(num_hidden):
    # plot amplitude dyn for unit i
    plt.plot(domain, x_t[0, :, i])
# plot stimulus
plt.plot(domain, stimulus[:, 0, 0], color = 'k', linewidth = 2)
plt.xlabel('time')
plt.ylabel('amplitude')

plt.show()
