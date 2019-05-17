import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--dataset_size', type=int, default=60000, help='number of examples used')
    parser.add_argument('--dataset', type=str, default='mnist', help='number of examples used')
    parser.add_argument('--estim_size', type=int, default=5000, help='Number of samples to estimate H and C')
    parser.add_argument('--model', type=str, default='cnn', help='Model to use')
    parser.add_argument('--activation', type=str, default='relu', help='activation to use')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--momentum', type=float, default=0.9, help='beta1 for adam. default=0.5')
    parser.add_argument('--mode', type=str, default='inv', help='How to compute TrH1C. With inv of lstsq')
    parser.add_argument('--rcond', type=float, default=1e-8, help='Condition number')
    parser.add_argument('--corrupt', type=float, default=0e-8, help='Condition number')
    #parser.add_argument('--reg', type=float, default=1e-2, help='Condition number')
    parser.add_argument('--hidden_size', type=int, default=30, help='number of examples used')
    parser.add_argument('--cuda', default=True, help='enables cuda')
    cuda = torch.cuda.is_available()

    opt = parser.parse_args()
    print(opt)
    return opt

