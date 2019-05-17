import argparse
import os
import ipdb
import torch
import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
import torch.distributions as dist
from torch.utils import data
from models import LinearReg
import numpy as np
import random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboardX import SummaryWriter
import datetime
from itertools import count

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--data_dim', type=int, default=50, metavar='N',
                    help='dimensionality of X')
parser.add_argument('--N', type=int, default=1500, metavar='N',
                    help='dataset size')
parser.add_argument('--steps', type=int, default=3500, metavar='N',
                    help='number of optim steps performed')
parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                    help='log every n steps')
parser.add_argument('--learning_rate', type=float, default=5e-3, metavar='N',
                    help='learning rate')
parser.add_argument('--lanbda', type=float, default=0e-3, metavar='N',
                    help='regularization')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()

seed = np.random.randint(0, 1000)
torch.manual_seed(seed)

device = torch.device("cuda" if use_cuda else "cpu")

# # Linear regression

# In[26]:
now = datetime.datetime.now()
outf = str(now.month)+'_'+str(now.day)+'/'
outf += str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)
outf += f'_d={args.data_dim}_N={args.N}_m={args.batch_size}_lr={args.learning_rate}_lanbda={args.lanbda}'
folder = './log/linear_reg/'+outf
writer = SummaryWriter(folder)
directory = folder+'/im/'
if not os.path.exists(directory):
        os.makedirs(directory)


# Declaring data distribution
#Cx = torch.ones(args.data_dim, ) # gaussian isotropic case
# Train and test sets
# X_tr = X_dist.sample((args.N,))
# X_te = X_dist.sample((args.N,))
X_tr = torch.randn(args.N, args.data_dim)
X_te = torch.randn(args.N, args.data_dim)
# Optimal parameter
w_true = torch.ones(args.data_dim, 1)

# Additive noise on ouput
rand_mat = torch.eye(1)#torch.randn(p,p)
sigma_noise = 1*rand_mat.t()@rand_mat #Identity in this case
Y_tr = X_tr@w_true+ torch.randn(args.N, 1)@sigma_noise
Y_te = X_te@w_true + torch.randn(args.N, 1)@sigma_noise

print(X_tr.size())


H_tr = (X_tr.t()@X_tr+args.lanbda*torch.eye(args.data_dim)).to(device)
H_tr_1 = torch.inverse(H_tr)
H_te = (X_te.t()@X_te+args.lanbda*torch.eye(args.data_dim)).to(device)
H_te_1 = torch.inverse(H_te)

model = LinearReg(args.data_dim)
model = model.to(device)

def frob_sim(A,B):
    normA = torch.sqrt(torch.trace(A.t()@A))
    normB = torch.sqrt(torch.trace(B.t()@B))
    prodAB = torch.trace(A.t()@B)
    return prodAB/(normA*normB)

# Estimate cov matrix (not recentering)
def estimate_C(model, loader):
#     w = list(model.parameters)[0].t()
    C = torch.zeros(args.data_dim, args.data_dim).to(device)
    grad_grad = torch.zeros(args.data_dim, args.data_dim).to(device)
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        grad = data.t()@(model(data)-target)/args.batch_size + args.lanbda*list(model.parameters())[0]
        C += grad@grad.t()/len(loader)
    return C.detach()

# Estimate gradient norm^2
def grad_norm2(model):
    tot = 0
    for p in model.parameters():
        tot += (p.grad**2).sum().item()
    return tot

########
# Making datasets
train_data = data.TensorDataset(X_tr, Y_tr)
test_data = data.TensorDataset(X_te, Y_te)
train_loader = data.DataLoader(dataset=train_data,
                                           batch_size=args.batch_size,
                                           shuffle=True)
test_loader = data.DataLoader(dataset=test_data,
                                           batch_size=args.batch_size,
                                           shuffle=True)
#######


criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,\
        momentum=0., weight_decay=args.lanbda)

def test(X, model):
    model.eval()
    total_loss = 0
    for data, target in X:
        data, target = data.to(device), target.to(device)
        pred = model(data)
        loss = criterion(pred, target)
        total_loss += loss.item()/len(X)
    return total_loss


def train(model, T):
    model.train()
    results = {}
    results['trH1C'] = []
    results['gap'] = []
    results['test_loss'] = []
    t = 0
    for i in count(0): # iter ad infinitam over epochs
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            t += 1
            pred = model(data)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % args.log_interval == 0:
                print(f'Logging step {t}')
                g2 = grad_norm2(model)
                loss_tr = test(train_loader, model)
                loss_te = test(test_loader, model)
                C_tr = estimate_C(model, train_loader)
                C_te = estimate_C(model, test_loader)

                w = list(model.parameters())[0]
                writer.add_scalar(f'loss/w-w_opt', ((w-w_true.to(device))**2).sum(), t)


                writer.add_scalar(f'loss/train', loss_tr, t)
                writer.add_scalar(f'loss/test', loss_te, t)
                writer.add_scalar(f'loss/gap', loss_te-loss_tr, t)
                writer.add_scalar(f'grad/norm2', g2, t)
                writer.add_scalar(f'information/trHC/train',
                        torch.trace(H_tr@C_tr), t)
                writer.add_scalar(f'information/trHC/test',
                        torch.trace(H_te@C_te), t)
                writer.add_scalar(f'information/trH_1C/train',
                        torch.trace(H_tr_1@C_tr), t)
                writer.add_scalar(f'information/trH_1C/test',
                        torch.trace(H_te_1@C_te), t)
                writer.add_scalar(f'information/cosHC/train',
                        frob_sim(H_tr, C_tr), t)
                writer.add_scalar(f'information/cosHC/test',
                        frob_sim(H_te, C_te), t)
                writer.add_scalar(f'information/L2/train',
                        torch.sum((H_tr-C_tr)**2), t)
                writer.add_scalar(f'information/L2/test',
                        torch.sum((H_te-C_te)**2), t)
                writer.add_scalar(f'information/norm_frob/H_tr',
                        torch.trace(H_tr.t()@H_tr), t)
                writer.add_scalar(f'information/norm_frob/H_te',
                        torch.trace(H_te.t()@H_te), t)
                writer.add_scalar(f'information/norm_frob/C_tr',
                        torch.trace(C_tr.t()@C_tr), t)
                writer.add_scalar(f'information/norm_frob/C_te',
                        torch.trace(C_te.t()@C_te), t)
                results['trH1C'].append(torch.trace(H_te_1@C_te).item())
                results['gap'].append((loss_te-loss_tr))
                results['test_loss'].append(loss_te)
            if t == args.steps:
                return results

                ### add writer 



results = train(model, args.steps)

def plot_read(x, y, label):
    fig = plt.figure()
    plt.plot(x, y, 'o-', label=label)
    fig.savefig(f'{folder}/im/{label}.png', bbox_inches='tight')   # save the figure to file
    plt.close('all')

    im = plt.imread(f'{folder}/im/{label}.png')
    writer.add_image(f'{label}', im, 1)

plot_read(results['gap'], results['trH1C'], 'x_gap-y_trH1C')
plot_read(results['test_loss'], results['trH1C'], 'x_losstest-y_trH1C')
