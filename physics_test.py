import torch
import torch.nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from qpth.qp import QPFunction
from qpth.qp import QPSolvers

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

import pdb

class PhysicsNet(torch.nn.Module):
    def __init__(self, startmu):
        super().__init__()
        self.mu = Parameter(torch.tensor([startmu]))

    def forward(self, data):
        n = data.shape[0]
        vs = torch.unsqueeze(data[:, 0], 1)
        vnexts = torch.unsqueeze(data[:, 1], 1)
        us = torch.unsqueeze(data[:, 2], 1)

        mu = self.mu
        #mu = torch.tensor([1.0])

        beta = vnexts - vs - us

        G = torch.tensor([[1.0, -1, 1], [-1, 1, 1], [-1, -1, 0]])

        Gpad = torch.tensor([ [1.0, -1, 1, 0, 0, 0],
                              [-1, 1, 1, 0, 0, 0],
                              [-1, -1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]])
        
        fmats = torch.tensor([[1.0, 1, 0], 
                              [-1, -1, 0],
                              [0, 0, 1]]).unsqueeze(0).repeat(n, 1, 1)
        fvecs = torch.cat((vs, us, mu * torch.ones(vs.shape)), 1).unsqueeze(2)
        f = torch.bmm(fmats, fvecs)
        batch_zeros = torch.zeros(f.shape)
        fpad = torch.cat((f, batch_zeros), 1)
        
        # For prediction error
        A = torch.tensor([[1.0, -1, 0, 0, 0, 0],
                          [-1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]]).repeat(n, 1, 1)
        beta_zeros = torch.zeros(beta.shape)
        b = torch.cat((-2 * beta, 2 * beta, beta_zeros, beta_zeros, beta_zeros, beta_zeros), 1).unsqueeze(2)
        
        slack_penalty = torch.tensor([0.0, 0, 0, 1, 1, 1]).repeat(n, 1).unsqueeze(2)

        a1 = 1
        a2 = 1
        a3 = 1

        Q = 2 * a1 * A + 2 * a2 * Gpad
        pdb.set_trace()
        p = a1 * b + a2 * fpad + a3 * slack_penalty
        
        # Constrain lambda and slacks to be >= 0
        R = -torch.eye(6)
        h = torch.zeros((1, 6))

        # Constrain G lambda + f >= 0
        #R = torch.cat((R, -G))
        # Should not have second negative here?
        R = torch.cat((R, -torch.cat((G, -torch.eye(3)), 1)))
        #R = torch.cat((R, -torch.cat((G, torch.zeros(3,3)), 1)))
        h = torch.cat((h.transpose(0, 1), f.unsqueeze(1)))
        h = h.transpose(0, 1)

        Qmod = 0.5 * (Q + Q.transpose(0, 1)) + 0.001 * torch.eye(6)
        
        z = QPFunction(check_Q_spd=False)(Qmod, p, R, h, 
                torch.tensor([]), torch.tensor([]))

        #print(self.scipy_optimize(0.5 * (Q + Q.transpose(0, 1)), p, R, h))
        #assert(torch.all(torch.matmul(R, z.transpose(0, 1)) \
        #                <= (h.transpose(0, 1) + torch.ones(h.shape) * 1e-5)))

        lcp_slack = torch.matmul(Gpad, z.transpose(0, 1)).transpose(0, 1) + fpad
        #print(z[0])

        cost = 0.5 * torch.matmul(z, torch.matmul(Qmod, z.transpose(0, 1))) \
                + torch.matmul(p, z.transpose(0, 1)) + a1 * beta**2
        return cost

    def scipy_optimize(self, Q, p, R, h):
        Qnp = Q.numpy()
        pnp = p.numpy()
        Rnp = R.numpy()
        hnp = h.numpy()

        def fun(z):
            return 0.5 * np.matmul(z, np.matmul(Q, z)) + np.matmul(p, z)
        
        linear_constraint = LinearConstraint(Rnp, 
                [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], hnp[0])
        res = minimize(fun, [1, 1, 1], method='trust-constr', constraints=[linear_constraint])

        return res.x



#print(QPFunction()(torch.tensor([[1.0]]), torch.tensor([1.0]), 
#                torch.tensor([[1.0]]), torch.tensor([5.0]), 
#                torch.tensor([]), torch.tensor([])))
#print(net.scipy_optimize(torch.tensor([[1.0]]), torch.tensor([[1.0]]), 
#                torch.tensor([1.0]), torch.tensor([[5.0]])))

# Previous vel, next vel, u
data = torch.tensor([[1.0, 2.0, 2.0],
                     [2.0, 2.0, 1.0]])

evolutions = []
#for startmu in np.linspace(0.1, 10, num=30):
for startmu in [4.]:
    net = PhysicsNet(startmu)

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

    evolution = []
    for epoch in range(100):
        # Zero the gradients
        optimizer.zero_grad()

        error = net(data)

        loss = torch.norm(error, 2)
        evolution.append([epoch, net.mu.item()])
        print('epoch: {}, loss: {:0.4f}, mu: {:0.4f}'.format(
            epoch, loss.item(), net.mu.item()))
        
        # perform a backward pass (backpropagation)
        loss.backward()
        
        # Update the parameters
        #optimizer.step()
        for p in net.parameters():
            if p.requires_grad:
                p.data.add_(0.1, -p.grad.data)
                
    evolutions.append(evolution)

evolutions_array = np.array(evolutions)
np.save('soft_evolution', evolutions_array)
