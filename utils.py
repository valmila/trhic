import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as udata
# import torchvision
# import torchvision.transforms as transforms
import ipdb

from kfac import KFAC



from hessian_eigenthings import compute_hessian_eigenthings


def calculate_low_rank_tic(args, loader, model, rank=1):
    loss = torch.nn.functional.nll_loss

    num_eigenthings = 1  # compute top 20 eigenvalues/eigenvectors
    eigenvals, eigenvecs = compute_hessian_eigenthings(model, loader
                                                       ,
                                                       loss, num_eigenthings, use_gpu=torch.cuda.is_available())
    low_rank_tic = 0
    n_examples = 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(args.device), y.to(args.device)
        for t in range(x.size(0)):
    #         print(x[t].unsqueeze(0).size())
            y_pred = model(x[t].unsqueeze(0))
            loss = F.nll_loss(y_pred, y[t].unsqueeze(0), reduction='sum')

            grads = torch.autograd.grad(loss, model.parameters())


            grads = torch.cat([g.view(-1) for g in grads]).numpy()

            low_rank_tic += (1/eigenvals)@(eigenvecs@grads)**2
            #optimizer.zero_grad(
    #             cov += torch.ger(grads, grads).detach()
        n_examples += x.size(0)
        if n_examples> args.estim_size:
            break

#     print(res)
    return low_rank_tic/n_examples
    

def calculate_kfac_tic(args, loader, model, eps=0.1):
    prec = KFAC(model, eps)
    kfac_tic = 0
    n_param = sum(p.numel() for p in model.parameters())
#     cov = torch.zeros(n_param, n_param).to(args.device)
    n_examples = 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(args.device), y.to(args.device)
        for t in range(x.size(0)):
    #         print(x[t].unsqueeze(0).size())
            y_pred = model(x[t].unsqueeze(0))
            loss = F.nll_loss(y_pred, y[t].unsqueeze(0), reduction='sum')

            grads = torch.autograd.grad(loss, model.parameters())
            
            for j in range(len(grads)):
                kfac_fisher = prec.params[j//2]['params'][j%2]
#                 kfac_fisher[kfac_fisher > 1] = 0
                kfac_tic += torch.sum(grads[j]*grads[j]*kfac_fisher)
            
#             grads = torch.cat([g.view(-1) for g in grads])
            #optimizer.zero_grad()
#             cov += torch.ger(grads, grads).detach()
        n_examples += x.size(0)
        if n_examples> args.estim_size:
            break
#     print(res)
    return kfac_tic.item()/n_examples



def calculate_hessian(args, loader, model):
    loss = 0
    n_examples = 0
    for i, (data, labels) in enumerate(loader):
        data, labels = data.to(args.device), labels.to(args.device)
        y_pred = model(data)
        loss += F.nll_loss(y_pred, labels, reduction='sum')
        n_examples += data.size(0)
        if n_examples> args.estim_size:
            break
    temp = []
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
    grads = torch.cat([g.view(-1) for g in grads])

    for i in range(grads.numel()):
        grad2 = torch.autograd.grad(grads[i], model.parameters(), retain_graph=True)
        grad2 = torch.cat([g.contiguous().view(-1) for g in grad2])
        temp.append(grad2.cpu().numpy())
    return np.array(temp)/n_examples

def frob_sim(A,B):
    normA = np.sqrt(np.trace(A.T@A))
    normB = np.sqrt(np.trace(B.T@B))
    prodAB = np.trace(A.T@B)
    return prodAB/(normA*normB)


def calculate_covariance(args, loader, model):
    n_param = sum(p.numel() for p in model.parameters())
    cov = torch.zeros(n_param, n_param).to(args.device)
    n_examples = 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(args.device), y.to(args.device)
        for t in range(x.size(0)):
    #         print(x[t].unsqueeze(0).size())
            y_pred = model(x[t].unsqueeze(0))
            loss = F.nll_loss(y_pred, y[t].unsqueeze(0), reduction='sum')

            grads = torch.autograd.grad(loss, model.parameters())
            grads = torch.cat([g.view(-1) for g in grads])
            #optimizer.zero_grad()
            cov += torch.ger(grads, grads).detach()
        n_examples += x.size(0)
        if n_examples> args.estim_size:
            break

    return np.array(cov.cpu())/n_examples

def calculate_gaussnewton(args, loader, model):
    n_param = sum(p.numel() for p in model.parameters())
    #cov = torch.zeros(n_param, n_param).to(args.device)
    cov = np.zeros((n_param, n_param))
    n_examples = 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(args.device), y.to(args.device)
        for t in range(x.size(0)):
    #         print(x[t].unsqueeze(0).size())
            logits = model.logits(x[t].unsqueeze(0)).view(-1, 10)

            loss = nn.CrossEntropyLoss(reduction='sum')(logits, y[t].unsqueeze(0))

            ######=====###
            hess_loss = []
            grads = torch.autograd.grad(loss, logits, create_graph=True, retain_graph=True)
            grads = torch.cat([g.view(-1) for g in grads])

            for i in range(grads.numel()):
                grad2 = torch.autograd.grad(grads[i], logits, retain_graph=True)
                grad2 = torch.cat([g.contiguous().view(-1) for g in grad2])
                hess_loss.append(grad2.cpu().numpy())
            hess_loss = np.array(hess_loss)

            #model(x[t].unsqueeze(0))
            #loss = F.nll_loss(y_pred.detach(), y[t].unsqueeze(0), size_average=False)

            grad_logit = []
            for i in range(logits.numel()):
                grads = torch.autograd.grad(logits.squeeze()[i], model.parameters(), retain_graph=True)
                grads = torch.cat([g.view(-1) for g in grads])
                grad_logit.append(grads.cpu().numpy())
            grad_logit = np.array(grad_logit)
            #optimizer.zero_grad()
            cov += grad_logit.T@hess_loss@grad_logit
            #cov += torch.ger(grads, grads).detach()
        n_examples += x.size(0)
        if n_examples> args.estim_size:
            break
        return np.array(cov)/n_examples


def calculate_fisher(args, loader, model):
    n_param = sum(p.numel() for p in model.parameters())
    fish = torch.zeros(n_param, n_param).to(args.device)
    n_examples = 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(args.device), y.to(args.device)
        for t in range(x.size(0)):
    #         print(x[t].unsqueeze(0).size())
            y_pred = model(x[t].unsqueeze(0))
            y_fisher = torch.multinomial(F.softmax(y_pred, dim=-1), 1).squeeze().detach()
            y_fisher = y_fisher.view(*y[t].unsqueeze(0).size())
            loss = F.nll_loss(y_pred, y_fisher, reduction='sum')

            grads = torch.autograd.grad(loss, model.parameters())
            grads = torch.cat([g.view(-1) for g in grads])
            #optimizer.zero_grad()
            fish += torch.ger(grads, grads).detach()
        n_examples += x.size(0)
        if n_examples> args.estim_size:
            break

    return np.array(fish.cpu())/n_examples


# def calculate_aristide_fisher(args, loader, model):
#     n_param = sum(p.numel() for p in model.parameters())
#     fish = torch.zeros(n_param, n_param).to(args.device)
#     n_examples = 0
#     for i, (x, y) in enumerate(loader):
#         x, y = x.to(args.device), y.to(args.device)
#         for t in range(x.size(0)):
#     #         print(x[t].unsqueeze(0).size())
#             scores = model(x[t].unsqueeze(0))
#             y_fisher = torch.multinomial(F.softmax(y_pred, dim=-1), 1).squeeze().detach()
#             y_fisher = y_fisher.view(*y[t].unsqueeze(0).size())
#             loss = F.nll_loss(y_pred, y_fisher, reduction='sum')

#             grads = torch.autograd.grad(loss, model.parameters())
#             grads = torch.cat([g.view(-1) for g in grads])
#             #optimizer.zero_grad()
#             fish += torch.ger(grads, grads).detach()
#         n_examples += x.size(0)
#         if n_examples> args.estim_size:
#             break

#     return np.array(fish.cpu())/n_examples

def pr(M):
    sigma_vals = np.abs(np.linalg.eigvalsh(M))
    return np.sum(sigma_vals)**2/np.sum(sigma_vals**2)

def log_mat(hess, t, label, writer):
    eig = np.linalg.eigvalsh(hess).ravel()
    writer.add_histogram(label, eig, t)
    writer.add_scalar('max_eigen/'+label,np.max(eig), t)
    writer.add_scalar('trace/'+label, np.trace(hess), t)
    writer.add_scalar('pr/'+label, pr(hess) ,t)
    print(f'trace({label})={np.trace(hess)}')
    print(f'Logged {label}')
    return np.max(eig), np.trace(hess), pr(hess)

def stat_mat(hess):
    return None, np.trace(hess), None
    eig = np.linalg.eigvalsh(hess).ravel()
    return np.max(eig), np.trace(hess), pr(hess)


# we should divide by the number of samples we evalute on rather that 
def compute_loss(args, loader, model):
    loss = 0
    n_examples = 0
    for data, labels in loader:
        n_examples += data.size(0)
        data, labels = data.to(args.device), labels.to(args.device)
        y_pred = model(data)
        loss += F.nll_loss(y_pred, labels, reduction='sum')
        #if n_examples> args.estim_size:
            #break
    return loss/n_examples

def compute_fisher_rao(args, loader, model):
    all_fr = 0
    for data, labels in loader:
        data, labels = data.to(args.device), labels.to(args.device)
        fr = model.fisher_rao(args, data, labels)
        all_fr += fr
    return all_fr

def compute_n_params(model):
    n_param = sum(p.numel() for p in model.parameters())
    return n_param


def compute_sensitivity(args, loader, model):
    sensitivity = 0
    n_examples = 0
    for data, labels in loader:
        data, labels = data.to(args.device), labels.to(args.device)
        data.requires_grad=True
        y_pred = model(data)
        loss = F.nll_loss(y_pred, labels, reduction='sum')
        loss.backward()
        sensitivity += (data.grad**2).sum()
        n_examples += data.size(0)
        if n_examples> args.estim_size:
            break
    return sensitivity/n_examples

def inv_rank_k(lanbda, P, cov, k):
    inv_lanbda = np.zeros_like(lanbda)
    inv_lanbda[-k:] = 1/lanbda[-k:]
    return P@np.diag(inv_lanbda)@P.T@cov

def compute_fisher_rao_norm(args, loader, model):
    # do E[ || theta^T grad||^2]
    theta = parameters_to_vector(model.parameters())
    fr_norm = 0
    n_examples = 0

    for i, (data, labels) in enumerate(loader):
        data, labels = data.to(args.device), labels.to(args.device)
        y_pred = model(data)
        loss = F.nll_loss(y_pred, labels, size_average=False)
        n_examples += data.size(0)

        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
        grads = torch.cat([g.view(-1) for g in grads]).cpu()

        fr_norm += (torch.dot(theta, grads)**2).detach()

        if n_examples> args.estim_size:
            break
    return fr_norm/n_examples

def parameters_to_vector(parameters):
    r"""Convert parameters to one vector

    Arguments:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        #param_device = _check_param_device(param, param_device)

        vec.append(param.cpu().view(-1))
    return torch.cat(vec)

def compute_L2_norm(args, model):
    # do E[ || theta^T grad||^2]
    theta = parameters_to_vector(model.parameters())
    return torch.sum(theta**2).item()
