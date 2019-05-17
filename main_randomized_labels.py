from comet_ml import Experiment
import numpy as np
from args import get_args
import random
from get_loader import get_loader

from models import *
from utils import *

from tensorboardX import SummaryWriter
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np

args = get_args()
experiment = Experiment(api_key='98Hhyb58cThYVpxaOvbL3Yu8S', project_name='trhic-4datasets', workspace='valthom')

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print(device)
N, N_te = min(args.dataset_size, 60000), min(args.dataset_size, 10000)
args.dataset_size = N
args.test_size = N_te
experiment.log_multiple_params(vars(args))
experiment.disable_mp()

args.device=device
now = datetime.datetime.now()

train_loader, test_loader = get_loader(args)


if args.model == 'cnn':
    model = SmallCNN().to(device)
elif args.model == 'cnn_bn':
    model = SmallCNN_BN().to(device)
elif args.model == 'logreg':
    model = LogReg().to(device)
elif args.model == 'mlp':
    model = MLP().to(device)
elif args.model == 'big_mlp':
    model = BigMLP().to(device)
else:
    print('No model recognized')


optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
#optimizer = torch.optim.Adam(model.parameters(), lr = lr)
n_param = sum(p.numel() for p in model.parameters())

folder =\
f'./log/{now.month}_{now.day}/{now.isoformat()}_lr={args.lr}_param={n_param}_N={N}_bs={args.batch_size}_mu={args.momentum}_model={args.model}_n_estim={args.estim_size}'
writer = SummaryWriter(log_dir=folder)
# for t in range(50000):
t = 0
# model.train()
log_step = 75000
while t <= 750000:
    for data, labels in train_loader:
        data, labels = data.to(args.device), labels.to(args.device)
        y_pred = model(data)

        loss = F.nll_loss(y_pred, labels)
        if t % log_step == 0:
            model.eval()
            #print(t, loss.item())
            loss_tr = compute_loss(args, train_loader, model)
            loss_te = compute_loss(args, test_loader, model)
            gap = (loss_te-loss_tr).item()

            fr_tr = compute_fisher_rao(args, train_loader, model)
            fr_te = compute_fisher_rao(args, test_loader, model)
            #writer.add_scalar( t)
            #writer.add_scalar( t)

            sens_tr =  compute_sensitivity(args, train_loader, model)
            sens_te =  compute_sensitivity(args, test_loader, model)
            dic = { 'loss/tr': loss_tr.item(),\
            'loss/te': loss_te.item(),\
            'loss/gap': gap,\
            'fisher_rao/tr': fr_tr.item(),\
            'fisher_rao/te': fr_te.item(),\
            'sensitivity/tr': sens_tr.item(),\
            'sensitivity/te': sens_te.item()}
            for n, m in dic.items():
                writer.add_scalar(n, m, t)
                experiment.log_metric(n, m, t)
            model.train()

        if t % log_step  == 0:
            model.eval()
            print(t, 'gap:', gap)
            # # Gauss Newton
            # Gtr = calculate_gaussnewton(args, train_loader, model)
            # Gte = calculate_gaussnewton(args, test_loader, model)
            # print('Computed Gauss Newton')
            # _, trGtr, _ = stat_mat(Gtr)
            # _, trGte, _ = stat_mat(Gte)

            # fisher
            Ftr = calculate_fisher(args, train_loader, model)
            Fte = calculate_fisher(args, test_loader, model)
            print('Computed fisher')
            _, trFtr, _ = stat_mat(Ftr)
            _, trFte, _ = stat_mat(Fte)
            # hessian
            Htr = calculate_hessian(args, train_loader, model)
            Hte = calculate_hessian(args, test_loader, model)
            print('Computed hessians')

            _, trHtr, _ = stat_mat(Htr)
            _, trHte, _ = stat_mat(Hte)
            # covariance
            Ctr = calculate_covariance(args, train_loader, model)
            Cte = calculate_covariance(args, test_loader, model)
            print('Computed covariances')
            _, trCtr, _ = stat_mat(Ctr)
            _, trCte, _ = stat_mat(Cte)

            #h1c = np.linalg.pinv(hess, rcond=1e-4)@cov/N
            dic_hick = {}
            if args.mode == 'lstsq':
                for reg in [1e-6, 1e-4, 1e-2, 0.1, 1e-0]:
                    H1Ctr = np.linalg.lstsq(Htr+reg*np.eye(Htr.shape[0]), Ctr, rcond=args.rcond)[0]/N
                    H1Cte = np.linalg.lstsq(Hte+reg*np.eye(Hte.shape[0]), Cte, rcond=args.rcond)[0]/N_te
                    dic_hick[f'H1Ctr_{reg}'] = H1Ctr
                    dic_hick[f'H1Cte_{reg}'] = H1Cte

                    G1Ctr = np.linalg.lstsq(Gtr+reg*np.eye(Gtr.shape[0]), Ctr, rcond=args.rcond)[0]/N
                    G1Cte = np.linalg.lstsq(Gte+reg*np.eye(Gte.shape[0]), Cte, rcond=args.rcond)[0]/N_te
                    # dic_hick[f'G1Ctr_{reg}'] = G1Ctr
                    # dic_hick[f'G1Cte_{reg}'] = G1Cte

            elif args.mode == 'inv':
                H1Ctr = np.linalg.pinv(Htr, rcond=args.rcond)@Ctr/N
                H1Cte = np.linalg.pinv(Hte, rcond=args.rcond)@Cte/N_te
                dic_hick[f'H1Ctr'] = H1Ctr
                dic_hick[f'H1Cte'] = H1Cte

                F1Ctr = np.linalg.pinv(Ftr, rcond=args.rcond)@Ctr/N
                F1Cte = np.linalg.pinv(Fte, rcond=args.rcond)@Cte/N_te
                dic_hick[f'F1Ctr'] = F1Ctr
                dic_hick[f'F1Cte'] = F1Cte

            #     G1Ctr = np.linalg.pinv(Gtr, rcond=args.rcond)@Ctr/N
            #     G1Cte = np.linalg.pinv(Gte, rcond=args.rcond)@Cte/N_te
            #     dic_hick[f'G1Ctr'] = G1Ctr
            #     dic_hick[f'G1Cte'] = G1Cte
            #
            # elif args.mode == 'rankk':
            #     lanbdatr, Ptr = np.linalg.eigh(Htr)
            #     lanbdate, Pte = np.linalg.eigh(Hte)
            #     for k in [1, 2, 5, 10, 20, 30, 100]:
            #         dic_hick[f'H1Ctr_{k}'] = inv_rank_k(lanbdatr, Ptr, Ctr, k)/N
            #         dic_hick[f'H1Cte_{k}'] = inv_rank_k(lanbdate, Pte, Cte, k)/N_te
            #
            #     lanbdatr, Ptr = np.linalg.eigh(Gtr)
            #     lanbdate, Pte = np.linalg.eigh(Gte)
            #     for k in [1, 2, 5, 10, 20, 30, 100]:
            #         dic_hick[f'G1Ctr_{k}'] = inv_rank_k(lanbdatr, Ptr, Ctr, k)/N
            #         dic_hick[f'G1Cte_{k}'] = inv_rank_k(lanbdate, Pte, Cte, k)/N_te
            else:
                raise NotImplementedError

            #h1c = np.linalg.inv(hess + 1e-4*np.eye(n_param))@cov/N
            #h1c_test = np.linalg.pinv(hess_test, rcond=1e-4)@cov_test/N_te
            #h1c = np.linalg.pinv(hess)@cov/N
            #h1c_test = np.linalg.inv(hess_test+1e-4np.eye(n_param))@cov_test/N_te



            all_scalars = {}
            dic_mat = {'Htr': Htr, \
                       'Ctr': Ctr, \
                       'Ftr': Ftr, \
                       #'Gtr': Gtr, \
                       #'H1Ctr': H1Ctr, \
                       'Hte': Hte, \
                       'Cte': Cte, \
                       'Fte': Fte, \
                       #'Gte': Gte, \
                       #'H1Cte': H1Cte
                       }
            dic_mat.update(dic_hick)
            for name, mat in dic_mat.items():
                _, trace, _ = stat_mat(mat)
                #all_scalars['maxeig/'+name] = maxeig
                all_scalars['trace/'+name] = trace
                #all_scalars['pr/'+name] = part
            print(all_scalars)


            #all_scalars['ratio/maxeigC_o_Htr'] = maxeigCtr/maxeigHtr
            #all_scalars['ratio/maxeigC_o_Hte'] = maxeigCte/maxeigHte
            all_scalars['ratio/trace_C_o_Htr'] = trCtr/trHtr
            all_scalars['ratio/trace_C_o_Hte'] = trCte/trHte
            all_scalars['ratio/trace_C_o_Ftr'] = trCtr/trFtr
            all_scalars['ratio/trace_C_o_Fte'] = trCte/trFte
            # all_scalars['ratio/trace_C_o_Gtr'] = trCtr/trGtr
            # all_scalars['ratio/trace_C_o_Gte'] = trCte/trGte

            all_scalars['sim_frob/Hte_Htr'] = frob_sim(Htr, Hte).item()
            all_scalars['sim_frob/Cte_Ctr'] = frob_sim(Ctr, Cte).item()
            all_scalars['sim_frob/Ctr_Htr'] = frob_sim(Ctr, Htr).item()
            all_scalars['sim_frob/Cte_Hte'] = frob_sim(Cte, Hte).item()
            all_scalars['sim_frob/Fte_Ftr'] = frob_sim(Ftr, Fte).item()
            all_scalars['sim_frob/Ftr_Htr'] = frob_sim(Ftr, Htr).item()
            all_scalars['sim_frob/Fte_Hte'] = frob_sim(Fte, Hte).item()
            all_scalars['sim_frob/Ctr_Ftr'] = frob_sim(Ftr, Ctr).item()
            all_scalars['sim_frob/Cte_Fte'] = frob_sim(Fte, Cte).item()
            # all_scalars['sim_frob/Gtr_Htr'] = frob_sim(Gtr, Htr).item()
            # all_scalars['sim_frob/Gte_Hte'] = frob_sim(Gte, Hte).item()
            # all_scalars['sim_frob/Ctr_Gtr'] = frob_sim(Gtr, Ctr).item()
            # all_scalars['sim_frob/Cte_Gte'] = frob_sim(Gte, Cte).item()
            # all_scalars['sim_frob/Ftr_Gtr'] = frob_sim(Ftr, Gtr).item()
            # all_scalars['sim_frob/Fte_Gte'] = frob_sim(Fte, Gte).item()

            all_scalars['unsim_L2/Htr_Hte'] = ((Htr-     Hte)**2).sum()
            all_scalars['unsim_L2/Ctr_Cte'] = ((Ctr-       Cte)**2).sum()
            all_scalars['unsim_L2/Ctr_Htr'] = ((Ctr-           Htr)**2).sum()
            all_scalars['unsim_L2/Cte_Hte'] = ((Cte- Hte)**2).sum()
            all_scalars['unsim_L2/Fte_Hte'] = ((Fte- Hte)**2).sum()
            all_scalars['unsim_L2/Cte_Fte'] = ((Cte- Fte)**2).sum()
            all_scalars['unsim_L2/Ftr_Htr'] = ((Ftr- Htr)**2).sum()
            all_scalars['unsim_L2/Ctr_Ftr'] = ((Ctr- Ftr)**2).sum()

            all_scalars['trace/HCtr'] = np.trace(Htr@Ctr)
            all_scalars['trace/HCte'] = np.trace(Hte@Cte)

            all_scalars['trace/HHtr'] = np.trace(Htr@Htr) 
            all_scalars['trace/HHte'] = np.trace(Hte@Hte) 

            all_scalars['trace/CCtr'] = np.trace(Ctr@Ctr) 
            all_scalars['trace/CCte'] = np.trace(Cte@Cte) 

            all_scalars['trace/FFtr'] = np.trace(Ftr@Ftr) 
            all_scalars['trace/FFte'] = np.trace(Fte@Fte) 

            all_scalars['ratio/trace_HC_o_HHtr'] = np.trace(Htr@Ctr)/np.trace(Htr@Htr) 
            all_scalars['ratio/trace_HC_o_HHte'] = np.trace(Hte@Cte)/np.trace(Hte@Hte) 
            for n, m in all_scalars.items():
                writer.add_scalar(n, m, t)
                experiment.log_metric(n, m, t)

            model.train()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t += 1
model.eval()

writer.export_scalars_to_json(f"{folder}/all_scalars.json")
writer.close()
