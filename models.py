import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from utils import compute_n_params

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class LinearReg(nn.Module):
    def __init__(self, d):
        super(LinearReg, self).__init__()
        self.fc1 = nn.Linear(d, 1) 
        
    def forward(self, x):
        out = self.fc1(x)
        return out

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        # self.bn1 = torch.nn.BatchNorm1d(H)
        # self.bn2 = torch.nn.BatchNorm1d(H)
        # self.bn3 = torch.nn.BatchNorm1d(H)
        # self.bn4 = torch.nn.BatchNorm1d(H)
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary (differentiable) operations on Tensors.
        """
        x = F.relu(self.linear1(x))
        #x = self.bn1(x)
        x = F.relu(self.linear2(x))
        #x = self.bn2(x)
        x = F.relu(self.linear3(x))
        #x = self.bn3(x)
        x = F.relu(self.linear4(x))
        #x = self.bn4(x)
        y_pred = self.linear5(x)
        return y_pred

    
class LinearNet(torch.nn.Module):
    def __init__(self, D_in, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(LinearNet, self).__init__()
        self.linear = torch.nn.Linear(D_in, D_out)


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary (differentiable) operations on Tensors.
        """
        y_pred = self.linear(x)
        return y_pred

    

class NLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(NLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        layer = []
        layer.append(self.linear1)
       

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary (differentiable) operations on Tensors.
        """
        y_pred = self.linear(x)
        return y_pred


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, ngf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class SmallCNN(nn.Module):
    def __init__(self, nc1=15, nc2=20):
        super(SmallCNN, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(1, nc1, 3),
            nn.ReLU(inplace=True),
            # state size. (ndf) x 32 x 32
            #nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.Conv2d(nc1, nc2, 3),
            #nn.BatchNorm2d(nc2),
            nn.ReLU(inplace=True),
            # state size. (ndf*2) x 16 x 16
            #nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.Conv2d(nc2, 10, 3),
        )
        print(f'The network has {compute_n_params(self)} parameters.')
        weights_init(self)

    def logits(self, input):
        logit = self.main(input).view(-1, 10)
        return logit

    def forward(self, input):
        logit = self.logits(input)
        scores = F.log_softmax(logit, dim=1)
        return scores

    def fisher_rao(self, args, input, labels):
        logit = self.main(input).view(-1, 10)
        scores = F.log_softmax(logit, dim=1)
        # torch.dot(logits, scores) - scores[labels]
        res = (logit*scores).sum(-1) - scores[(torch.arange(labels.numel()), labels)]
        # loss = F.nll_loss(scores, labels, size_average=False)
        # grad_f_loss = torch.autograd.grad(outputs=loss, inputs=logit,\
        #                       grad_outputs=torch.ones(loss.size()).to(args.device),\
        #                       only_inputs=True)[0]
        # fisher_rao = ((loss*grad_f_loss).sum(1)**2).sum()
        fisher_rao = (res**2).sum()
        return fisher_rao

class LogReg(nn.Module):
    def __init__(self):
        super(LogReg, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(49, 10)
        )
        print(f'The network has {compute_n_params(self)} parameters.')
        weights_init(self)

    def logits(self, input):
        input = input.view(input.size(0), -1)
        output = self.main(input).view(-1, 10)
        return output

    def forward(self, input):
        output = self.logits(input)
        scores = F.log_softmax(output, dim=1)
        return scores

    def fisher_rao(self, args, input, labels):
        input = input.view(input.size(0), -1)
        logit = self.main(input).view(-1, 10)
        scores = F.log_softmax(logit, dim=1)
        loss = F.nll_loss(scores, labels, size_average=False)
        grad_f_loss = torch.autograd.grad(outputs=loss, inputs=logit,\
                              grad_outputs=torch.ones(loss.size()).to(args.device),\
                              only_inputs=True)[0]
        fisher_rao = ((loss*grad_f_loss).sum(1)**2).sum()
        return fisher_rao

class MLP(nn.Module):
    def __init__(self, hsize=30, activation='relu'):
        super(MLP, self).__init__()
        if activation == 'relu':
            act = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            act = nn.Tanh()
        else:
            raise NotImplementedError

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Linear(49, hsize),
            act,
            nn.Linear(hsize, 10)
        )
        print(f'The network has {compute_n_params(self)} parameters.')
        weights_init(self)

    def logits(self, input):
        input = input.view(input.size(0), -1)
        output = self.main(input).view(-1, 10)
        return output


    def forward(self, input):
        output = self.logits(input)
        scores = F.log_softmax(output, dim=1)
        return scores

    def fisher_rao(self, args, input, labels):
        input = input.view(input.size(0), -1)
        logit = self.main(input).view(-1, 10)
        scores = F.log_softmax(logit, dim=1)
        loss = F.nll_loss(scores, labels, size_average=False)
        grad_f_loss = torch.autograd.grad(outputs=loss, inputs=logit,\
                              grad_outputs=torch.ones(loss.size()).to(args.device),\
                              only_inputs=True)[0]
        fisher_rao = ((loss*grad_f_loss).sum(1)**2).sum()
        return fisher_rao

class SmallCNN_BN(nn.Module):
    def __init__(self, nc1=15, nc2=20):
        super(SmallCNN_BN, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(1, nc1, 3),
            nn.ReLU(inplace=True),
            # state size. (ndf) x 32 x 32
            #nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.Conv2d(nc1, nc2, 3),
            nn.BatchNorm2d(nc2),
            nn.ReLU(inplace=True),
            # state size. (ndf*2) x 16 x 16
            #nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.Conv2d(nc2, 10, 3),
        )
        print(f'The network has {compute_n_params(self)} parameters.')
        weights_init(self)

    def logits(self, input):
        logit = self.main(input).view(-1, 10)
        return logit

    def forward(self, input):
        logit = self.logits(input)
        scores = F.log_softmax(logit, dim=1)
        return scores

    def fisher_rao(self, args, input, labels):
        logit = self.main(input).view(-1, 10)
        scores = F.log_softmax(logit, dim=1)
        loss = F.nll_loss(scores, labels, size_average=False)
        grad_f_loss = torch.autograd.grad(outputs=loss, inputs=logit,\
                              grad_outputs=torch.ones(loss.size()).to(args.device),\
                              only_inputs=True)[0]
        fisher_rao = ((loss*grad_f_loss).sum(1)**2).sum()
        return fisher_rao

class BigMLP(nn.Module):
    def __init__(self, hsize=30):
        super(BigMLP, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Linear(49, hsize),
            nn.ReLU(inplace=True),
            nn.Linear(hsize, hsize),
            nn.ReLU(inplace=True),
            nn.Linear(hsize, 10)
        )
        print(f'The network has {compute_n_params(self)} parameters.')
        weights_init(self)

    def logits(self, input):
        input = input.view(input.size(0), -1)
        output = self.main(input).view(-1, 10)
        return output

    def forward(self, input):
        output = self.logits(input)
        scores = F.log_softmax(output, dim=1)
        return scores

    def fisher_rao(self, args, input, labels):
        input = input.view(input.size(0), -1)
        logit = self.main(input).view(-1, 10)
        scores = F.log_softmax(logit, dim=1)
        loss = F.nll_loss(scores, labels, size_average=False)
        grad_f_loss = torch.autograd.grad(outputs=loss, inputs=logit,\
                              grad_outputs=torch.ones(loss.size()).to(args.device),\
                              only_inputs=True)[0]
        fisher_rao = ((loss*grad_f_loss).sum(1)**2).sum()
        return fisher_rao
