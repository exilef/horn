# HORN network module
# Felix Effenberger, July 21, 2023

import math
import torch

# HORN model
class HORN(torch.nn.Module):

    def __init__(self, num_input, num_nodes, num_output, h, alpha, omega, gamma):
        super().__init__()

        self.num_input = num_input
        self.num_nodes = num_nodes
        self.num_output = num_output

        # hyperparameters h, alpha, omega, gamma
        self.h = h
        self.alpha = alpha
        self.omega = omega
        self.gamma = gamma

        # precompute omega^2 for DHO equation
        self.omega_factor = self.omega * self.omega

        # precompute 2 * gamma for DHO equation
        self.gamma_factor = 2.0 * self.gamma

        # precompute recurrent gain factor
        self.gain_rec = 1. / math.sqrt(self.num_nodes)

        # input, recurrent and output layers
        self.i2h = torch.nn.Linear(num_input, num_nodes)
        self.h2h = torch.nn.Linear(num_nodes, num_nodes)
        self.h2o = torch.nn.Linear(num_nodes, num_output)

    def dynamics_step(self, x_t, y_t, input_t):
        # one discrete dynamics step based on sympletic Euler integration

        # 1. integrate y_t
        y_t = y_t + self.h * (
            # input (forcing) on y_t
            self.alpha * torch.tanh(
                self.i2h(input_t) # external input
                + self.gain_rec * self.h2h(y_t) # recurrent input from network
            )
            - self.omega_factor * x_t # natural frequency term
            - self.gamma_factor * y_t # damping term
        )

        # 2. integrate x_t with updated y_t, no input here
        x_t = x_t + self.h * y_t

        # return updated x_t, y_t
        return x_t, y_t

    def forward(self, batch, random_init = None, record = False):
        # batch has shape: (time steps, batch size, ...)
        batch_size = batch.size(1)
        num_timesteps = batch.size(0)

        ret = {}

        if record:
            # record dynamics of x_t, y_t
            rec_x_t = torch.zeros(batch_size, num_timesteps, self.num_nodes)
            rec_y_t = torch.zeros(batch_size, num_timesteps, self.num_nodes)
            ret['rec_x_t'] = rec_x_t
            ret['rec_y_t'] = rec_y_t

        # initial conditions for variables x, y for each DHO node
        if not random_init is None:
            # Gauss random initial condition
            x_0 = torch.randn(batch_size, self.num_nodes) * random_init
            y_0 = torch.randn(batch_size, self.num_nodes) * random_init
        else:
            # initial condition x=y=0
            x_0 = torch.zeros(batch_size, self.num_nodes)
            y_0 = torch.zeros(batch_size, self.num_nodes)

        # make x_t, y_t autograd variables for automatic differentiation
        x_t = torch.autograd.Variable(x_0)
        y_t = torch.autograd.Variable(y_0)

        # loop over all time steps, feeding one pixel per time step
        for t in range(num_timesteps):
            # compute one dynamics step and update x_t, y_t
            x_t, y_t = self.dynamics_step(x_t, y_t, batch[t])

            if record:
                rec_x_t[:, t, :] = x_t
                rec_y_t[:, t, :] = y_t

        # linear readout at last time step
        output = self.h2o(x_t)

        ret['output'] = output
        return ret
