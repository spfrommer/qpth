import torch
import torch.nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from qpth.qp import QPFunction

import numpy as np
from numpy import linalg as LA
from cvxopt import matrix
from cvxopt import solvers

import pdb

prev_vels = torch.tensor([[1.0]])
next_vels = torch.tensor([[2.0]])
us = torch.tensor([[2.0]])

class PhysicsNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = Parameter(torch.tensor([0.2]))

    def forward(self, vk):
        mu = self.mu

        beta = next_vels - vk - us

        G = torch.tensor([[1.0, -1, 1], [-1, 1, 1], [-1, -1, 0]])
        
        f = torch.matmul(torch.tensor([[1.0, 1, 0], [-1, -1, 0], [0, 0, 1]]),
                torch.cat((vk[0], us[0], mu)))
        
        A = torch.tensor([[1.0, -1, 0], [-1, 1, 0], [0, 0, 0]])
        b = torch.tensor([-2 * beta, 2 * beta, 0])

        a1 = 1
        a2 = 1

        Q = 2 * a1 * A + a2 * G
        # I should need this?? but doesn't give right answers
        #Q = 2 * Q
        # Need some transposes here maybe
        p = a1 * b + a2 * f
        
        # Constrain lambda to be >= 0
        R = -torch.eye(3)
        h = torch.zeros((1, 3))

        # Constrain G lambda + f >= 0
        R = torch.cat((R, -G))
        h = torch.cat((h.transpose(0, 1), f.unsqueeze(1)))
        h = h.transpose(0, 1)
        
        z = QPFunction(check_Q_spd=False)(Q, p, R, h, torch.tensor([]), torch.tensor([]))

        lcp_slack = torch.matmul(G, z.transpose(0, 1)).transpose(0, 1) + f
        # Cost should have 0.5 in front of quadratic term
        # But again doesn't work for some reason
        cost = 0.5 * torch.matmul(z, torch.matmul(Q, z.transpose(0, 1))) \
                + torch.matmul(p, z.transpose(0, 1)) + a1 * beta**2

        error = beta - z[0][0] + z[0][1]
        return error

# print(QPFunction()(2.0 * torch.tensor([[1.0]]), torch.tensor([1.0]), 
                   # torch.tensor([[1.0]]), torch.tensor([5.0]), 
                   # torch.tensor([]), torch.tensor([])))

net = PhysicsNet()

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.5)

for epoch in range(500):
    # Zero the gradients
    optimizer.zero_grad()

    error = net(prev_vels)

    loss = torch.norm(error, 2)
    print('epoch: ', epoch,' loss: ', loss.item(), ' mu: ', net.mu.item())
    
    # perform a backward pass (backpropagation)
    loss.backward()
    
    # Update the parameters
    # optimizer.step()
    for p in net.parameters():
        if p.requires_grad:
            p.data.add_(0.1, -p.grad.data)
    
    # Needed to recreate the backwards graph
    # TODO: fix this properly
    #net.lcp_solver = LCPFunction()
