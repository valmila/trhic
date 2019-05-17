from __future__ import print_function
import ipdb
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
from copy import deepcopy
from models import MLP, CNN
import datetime

def evaluate_loss(model, device, train_loader, one_batch=False):
    batch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)
        if one_batch:
            batch_loss += loss.item()
            break
        batch_loss += loss.item()/len(train_loader)
    return batch_loss

def evaluate_drift_var(model, optimizer, device, train_loader, epoch, writer, args):
    print(f'evaluating {epoch}')
    current_state = deepcopy(model.state_dict())
    current_opt = deepcopy(optimizer.state_dict())
    # ==== batch ===
    # do the batch gradient update
    prev_loss = 0
    optimizer.zero_grad()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)/len(train_loader)
        loss.backward()
        prev_loss += loss.item()
    optimizer.step()
    print('batch step done')
    # get gradient norm squared
    #grad_batch = [p.grad for p in model.parameters()]
    muTmu = 0
    for p in model.parameters():
        muTmu += torch.sum(p.grad**2).item()

    # Get the loss for batch updated model
    batch_loss = evaluate_loss(model, device, train_loader, True)
    print('batch evaluation done')

    # ==== stochastic ===
    model.load_state_dict(current_state)
    optimizer.load_state_dict(current_opt)
    # Do one SGD update
    stoch_losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        # for ps, pb in zip(model.parameters(), grad_batch):
        #     ps.grad = ps.grad - args.mu*pb
        optimizer.step()

        # Compute posterior loss
        stoch_loss = evaluate_loss(model, device, train_loader, True)
        stoch_losses.append(stoch_loss)
        # Revert back
        model.load_state_dict(current_state)
        optimizer.load_state_dict(current_opt)

    stoch_losses = np.array(stoch_losses)
    ELs_Lb = stoch_losses.mean() - batch_loss
    ELs_Lt = stoch_losses.mean() - prev_loss
    Lb_Lt = batch_loss - prev_loss
    writer.add_scalar('ito/E_Ls-Lb', ELs_Lb, epoch)
    writer.add_scalar('ito/E_Ls-Lt', ELs_Lt, epoch)
    writer.add_scalar('ito/Lb-Lt', Lb_Lt, epoch)
    writer.add_scalar('ito/std_Ls', stoch_losses.std(), epoch)
    writer.add_scalar('ito/var_Ls', stoch_losses.var(), epoch)
    writer.add_scalar('ito/E_Ls', stoch_losses.mean(), epoch)
    writer.add_scalar('ito/Lb', batch_loss, epoch)
    log_d = -(stoch_losses.mean() - batch_loss)**2/(2*stoch_losses.var())
    writer.add_scalar('ito/log_density', log_d, epoch)
    writer.add_scalar('ito/grad_norm2', muTmu, epoch)
    writer.add_scalar('ito/var_Ls_ratio_grad_norm2', stoch_losses.var()/muTmu, epoch)
    writer.add_scalar('ito/E_Ls-Lb_ratio_E_Ls-Lb+Lb-Lt', ELs_Lb/(np.abs(ELs_Lb)+np.abs(Lb_Lt)), epoch)

    return batch_loss, stoch_losses

def train(args, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        if batch_idx % args.log_interval == 0:
            step = epoch*len(train_loader) + batch_idx
            evaluate_drift_var(model, optimizer, device, train_loader, step, writer, args)

        loss.backward()
        # adding the crazy gradient thing
        # model_batch = deepcopy(model)
        # for batch_idx, (data, target) in enumerate(train_loader):
        #     data, target = data.to(device), target.to(device)
        #     output = model_batch(data)
        #     loss = F.nll_loss(output, target)/len(train_loader)
        #     loss.backward()
        #grad_batch = [p.grad for p in model_batch.parameters()]
        # for ps, pb in zip(model.parameters(), grad_batch):
        #     ps.grad = ps.grad - args.mu*pb

        optimizer.step()

        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, epoch, writer, flag='test'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\n'+flag+' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    writer.add_scalar(f'losses/{flag}', test_loss, epoch)
    writer.add_scalar(f'accuracy/{flag}', correct / len(test_loader.dataset), epoch)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--mu', type=float, default=0., metavar='MU',
                        help='noise magnification')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--net', type=str, default='CNN', metavar='S',
                        help='type of network')
    parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    now = datetime.datetime.now()
    outf = str(now.month)+'_'+str(now.day)+'/'
    outf += str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)
    outf += f'_net_{args.net}_batchsize={args.batch_size}_lr={args.lr}_mom_{args.momentum}_mu_{args.mu}'
    writer = SummaryWriter('./log/'+outf)

    if args.net == 'MLP':
        model = MLP().to(device)
    elif args.net == 'CNN':
        model = CNN().to(device)
    else:
        raise NotImplementedError
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, epoch, writer)
        test(args, model, device, test_loader, epoch, writer)
        test(args, model, device, train_loader, epoch, writer, flag='train')


if __name__ == '__main__':
    main()
