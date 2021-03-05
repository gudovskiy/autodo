from __future__ import print_function
import argparse, os, sys, random, time, datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#
import torch


def get_args():
    parser = argparse.ArgumentParser(description='Get AutoDO Results')
    parser.add_argument('--data', default='./local_data', type=str, metavar='NAME',
                        help='folder to save all data')
    parser.add_argument('--dataset', default='MNIST', type=str,
                        help='dataset MNIST/CIFAR10/CIFAR100/SVHN/SVHN_extra/ImageNet')
    parser.add_argument('-ir', '--imbalance-ratio', type=int, default=1, metavar='N',
                        help='ratio of [1:C/2] to [C/2+1:C] labels in the training dataset drawn from uniform distribution')
    parser.add_argument('-sr', '--subsample-ratio', type=float, default=1.0, metavar='N',
                        help='ratio of selected to total labels in the training dataset drawn from uniform distribution')
    parser.add_argument('-nr', '--noise-ratio', type=float, default=0.0, metavar='N',
                        help='ratio of noisy (randomly flipped) labels (default: 0.0)')
    parser.add_argument('--overfit', action='store_true', default=False,
                        help='ablation: estimate DA on test data (default: False)')
    parser.add_argument('--oversplit', action='store_true', default=False,
                        help='ablation: train on all data (default: False)')
    parser.add_argument('--aug-model', default='NONE', type=str,
                        help='type of augmentation model NONE/RAND/AUTO/DADA/SHAred/SEParate parameters (default: NONE)')
    parser.add_argument('--los-model', default='NONE', type=str,
                        help='type of model for other loss hyperparams NONE/SOFT/WGHT/BOTH (default: NONE)')
    parser.add_argument('--hyper-opt', default='NONE', type=str,
                        help='type of bilevel optimization NONE/HES (default: NONE)')

    args = parser.parse_args()
    
    return args


def main(args):
    save_folder = '{}/{}'.format(args.data, args.dataset)
    args.hyper_est = True
    overfit = args.overfit
    oversplit = args.oversplit
    hyper_est = args.hyper_est
    imbalance_ratio = args.imbalance_ratio
    subsample_ratio = args.subsample_ratio
    noise_ratio     = args.noise_ratio
    model_postfix = 'ir_{}_sr_{}_nr_{}'.format(imbalance_ratio, subsample_ratio, noise_ratio)
    #
    if args.dataset == 'MNIST':
        model_name = 'resnet18'
        runs = 8
    elif args.dataset == 'CIFAR10':
        model_name = 'wresnet28_10'
        runs = 4
    elif args.dataset == 'CIFAR100':
        model_name = 'wresnet28_10'
        runs = 4
    elif args.dataset == 'SVHN' or args.dataset == 'SVHN_extra':
        extra_svhn = True if 'extra' in args.dataset else False
        model_name = 'wresnet28_10_extra' if extra_svhn else 'wresnet28_10'
        runs = 4
    elif args.dataset == 'ImageNet':
        model_name = 'resnet18'
        runs = 1
    else:
        print('{} is not supported dataset!\n'.format(args.dataset))
        sys.exit(0)
    #
    if overfit:
        model_name = 'overfit_' + model_name
    if oversplit:
        model_name = 'oversplit_' + model_name
    #
    acc = torch.zeros(runs)
    for r in range(runs):
        run_folder = 'run{}'.format(r)
        model_folder = '{}/{}'.format(save_folder, run_folder)
        run_name = '{}_opt_{}_est_{}_aug_model_{}_los_model_{}_{}'.format(
            model_name, args.hyper_opt, args.hyper_est, args.aug_model, args.los_model, model_postfix)
        checkpoint_file = '{}/best_{}.pt'.format(model_folder, run_name)
        checkpoint = torch.load(checkpoint_file)
        acc[r] = checkpoint['acc']
        print('{} : {}'.format(checkpoint_file, acc[r]))
    print('Mean/Std', torch.mean(acc), torch.std(acc))

if __name__ == '__main__':
    args = get_args()
    main(args)
