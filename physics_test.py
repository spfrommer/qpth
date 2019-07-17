import torch
import torch.nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from qpth.qp import QPFunction
from qpth.qp import QPSolvers

import numpy as np

import pdb

prev_vels = torch.tensor([[1.0]])
next_vels = torch.tensor([[2.0]])
us = torch.tensor([[2.0]])

class PhysicsNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = Parameter(torch.tensor([1.0]))

    def forward(self, vk):
        mu = self.mu
        mu = torch.tensor([1.0])

        beta = next_vels - vk - us

        G = torch.tensor([[1.0, -1, 1], [-1, 1, 1], [-1, -1, 0]])
        
        f = torch.matmul(torch.tensor([[1.0, 1, 0], [-1, -1, 0], [0, 0, 1]]),
                torch.cat((vk[0], us[0], mu)))
        
        # For prediction error
        A = torch.tensor([[1.0, -1, 0], [-1, 1, 0], [0, 0, 0]])
        b = torch.tensor([-2 * beta, 2 * beta, 0])

        a1 = 0
        a2 = 1

        Q = 2 * a1 * A + 2 * a2 * G
        p = a1 * b + a2 * f
        
        # Constrain lambda to be >= 0
        R = -torch.eye(3)
        h = torch.zeros((1, 3))

        # Constrain G lambda + f >= 0
        R = torch.cat((R, -G))
        h = torch.cat((h.transpose(0, 1), f.unsqueeze(1)))
        h = h.transpose(0, 1)
        
        z = QPFunction(check_Q_spd=False)(Q, p, R, h, 
                torch.tensor([]), torch.tensor([]))
        z = torch.tensor([[0.0, 1, 2]])
        assert(torch.all(torch.matmul(R, z.transpose(0, 1)) \
                        <= (h.transpose(0, 1) + torch.ones(h.shape) * 1e-7)))

        lcp_slack = torch.matmul(G, z.transpose(0, 1)).transpose(0, 1) + f

        cost = 0.5 * torch.matmul(z, torch.matmul(Q, z.transpose(0, 1))) \
                + torch.matmul(p, z.transpose(0, 1)) + a1 * beta**2
        pdb.set_trace()
        return cost

# print(QPFunction()(2.0 * torch.tensor([[1.0]]), torch.tensor([1.0]), 
                   # torch.tensor([[1.0]]), torch.tensor([5.0]), 
                   # torch.tensor([]), torch.tensor([])))

net = PhysicsNet()

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

for epoch in range(10000):
    # Zero the gradients
    optimizer.zero_grad()

    error = net(prev_vels)

    loss = torch.norm(error, 2)
    print('epoch: {}, loss: {:0.4f}, mu: {:0.4f}'.format(
        epoch, loss.item(), net.mu.item()))
    
    # perform a backward pass (backpropagation)
    loss.backward()
    
    # Update the parameters
    # optimizer.step()
    for p in net.parameters():
        if p.requires_grad:
            p.data.add_(0.01, -p.grad.data)
    
    # Needed to recreate the backwards graph
    # TODO: fix this properly
    #net.lcp_solver = LCPFunction()
