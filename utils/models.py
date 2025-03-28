import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim



class SvmPrimal(nn.Module):
    def __init__(self, c=100, lr=0.001):
        super().__init__()

        self.c = c
        self.linear = nn.Linear(2, 1)
        
        
        # define loss, optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)

    def forward(self, x):
        x = self.linear(x)
        return x

    def get_loss(self, labels, predict, weight):
        loss = self.c * torch.mean(torch.clamp(1 - labels * predict, min=0))
        loss +=  (weight.t() @ weight) / 2.0

        return loss

    def train_step(self, inputs, labels):
        output = self(inputs)
        output = output.squeeze()

        weight = self.linear.weight.squeeze()

        input_loss = self.get_loss(labels, output, weight)

        input_loss.backward()

        self.optimizer.step()

        return input_loss.item()



class SvmDual(nn.Module):
    def __init__(self, x_train, y_train, C=100, lr=0.001, rbf=True, sigma=1):
        super().__init__()

        self.C = C

        self.sigma = sigma

        self.if_rbf = rbf

        self.init_svm(x_train, y_train)
        
        self.lr = lr

        

        

    def init_svm(self, x_train, y_train):
         # learned layers
        self.alpha = nn.Linear(x_train.shape[0], 1, bias=False).weight.flatten()
        self.b = 0

        # fixed members
        self.x_train = torch.tensor(x_train)
        self.y_train = torch.tensor(y_train)

        '''
        self.inner_min = torch.tensor(0).type(torch.float32)
        self.inner_range = torch.tensor(1).type(torch.float32)
        print("init inner min, range: ", self.inner_min, self.inner_range)

        self.inner_mat = self.inner_prod(self.x_train, self.x_train)
        print("inner mat max min: ", self.inner_mat.max(), self.inner_mat.min())

        self.inner_min = self.inner_mat.min()
        self.inner_range = self.inner_mat.max() - self.inner_min

        print("init inner min, range: ", self.inner_min, self.inner_range)
        '''
        
        # need apply normalized inner product, re compute inner mat again
        self.inner_mat = self.inner_prod(self.x_train, self.x_train)
        #print("After normalized, inner mat max min: ", self.inner_mat.max(), self.inner_mat.min())
        #print("After standarization, inner mat mean, std: ", self.inner_mat.mean(), self.inner_mat.std())

        

        self.yi_yj_xij = torch.outer(self.y_train, self.y_train) * self.inner_mat

        self.ones = torch.ones(self.x_train.shape[0])

    # fix on rbf
    def inner_prod(self, x1, x2):
        #if self.if_rbf:
        inner = self.kernel(x1, x2)
        #else:
            #inner = (torch.matmul(x1, x2.T) - self.inner_min) / self.inner_range
        return inner

    # non linear kernel
    def kernel(self, x1, x2):
        print("shape: ", x1.shape, x2.shape)
        # here rbf
        norm_2 = torch.linalg.norm(x1[:, None] - x2[None, :], axis=2) ** 2

        norm_2_sig = (-(1/self.sigma) ** 2) * norm_2

        return torch.exp(norm_2_sig)

    def forward(self, x):
        x_mid = self.inner_prod(self.x_train, x)

        alpha_mid = self.alpha * self.y_train
        # here shape of alpha_mid (m, ) is same as (1, m), so alpha_mid(m, ) able to multiply with x_mid (m, n)

        # or to clarify, set actual shape of alpha_mid
        alpha_mid = alpha_mid[None, :]
        y_pred = torch.matmul( alpha_mid , x_mid ) + self.b

        return y_pred.unsqueeze(1)

    def hyplane_b(self):
        sup_v = ( (self.alpha > 0) & (self.alpha < self.C) ).nonzero(as_tuple=True)
        x_mid = self.inner_prod(self.x_train, self.x_train[sup_v])
        
        #x_mid = self.y_train[sup_v]*x_mid

        #alpha_sup = self.alpha[sup_v]
        alpha_y = self.alpha * self.y_train
        alpha_y = alpha_y[None, :]
        alpha_x = torch.matmul(alpha_y, x_mid)
        
        b_i = self.y_train[sup_v] - alpha_x.flatten()

        print("sup v: ", sup_v)
        print("b_i: ", b_i)
        print("y picked: ", self.y_train[sup_v])
        print("support alpha: ", self.alpha[sup_v])
        print("alpha > 0: ", self.alpha[self.alpha>0])
        self.b = b_i.mean()


    def get_loss(self):
        loss =  - self.alpha.sum() + 0.5 * torch.sum( torch.outer(self.alpha, self.alpha) * self.yi_yj_xij )

        return loss

    def update_gradient(self):
        #print("yi_yj_xij * self.alpha: ", torch.matmul( self.yi_yj_xij, self.alpha))
        gradient =  - self.ones + torch.matmul( self.yi_yj_xij, self.alpha)

        self.alpha = self.alpha - self.lr * gradient

    def train_step(self):

        # loss, projected gradient
        epoch_loss = self.get_loss()
        self.update_gradient()
        
        # 2nd half of projected gradient
        # deal with constraint
        self.alpha = torch.clamp(self.alpha, min=0, max=self.C)

        return epoch_loss.item()


