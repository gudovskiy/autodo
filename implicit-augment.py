from __future__ import print_function
import argparse, os, sys, random, time, datetime
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SubsetRandomSampler, Sampler, Subset, ConcatDataset
#
from custom_models import *
from custom_datasets import *
from custom_transforms import *
from utils import *


def get_args():
    parser = argparse.ArgumentParser(description='AutoDO using Implicit Differentiation')
    parser.add_argument('--data', default='./local_data', type=str, metavar='NAME',
                        help='folder to save all data')
    parser.add_argument('--dataset', default='MNIST', type=str, metavar='NAME',
                        help='dataset MNIST/CIFAR10/CIFAR100/SVHN/SVHN_extra/ImageNet')
    parser.add_argument('--workers', default=4, type=int, metavar='NUM',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=200, metavar='NUM',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, metavar='LR',
                        help='learning rate decay (default: 0.1)')
    parser.add_argument('--lr-decay-epochs', type=str, default='150,175,195', metavar='LR',
                        help='learning rate decay epochs (default: 150,175,195')
    parser.add_argument('--lr-warm-epochs', type=int, default=5, metavar='LR',
                        help='number using cosine annealing (default: False')
    parser.add_argument("--gpu", default='0', type=str, metavar='NUM',
                        help='GPU device number')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=500, metavar='NUM',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--plot-debug', action='store_true', default=False,
                        help='plot train images for debugging purposes')
    parser.add_argument('-ir', '--imbalance-ratio', type=int, default=1, metavar='N',
                        help='ratio of [1:C/2] to [C/2+1:C] labels in the training dataset drawn from uniform distribution')
    parser.add_argument('-sr', '--subsample-ratio', type=float, default=1.0, metavar='N',
                        help='ratio of selected to total labels in the training dataset drawn from uniform distribution')
    parser.add_argument('-nr', '--noise-ratio', type=float, default=0.0, metavar='N',
                        help='ratio of noisy (randomly flipped) labels (default: 0.0)')
    parser.add_argument('-r', '--run-folder', default='run0', type=str,
                        help='dir to save run')
    parser.add_argument('--overfit', action='store_true', default=False,
                        help='ablation: estimate DA from test data (default: False)')
    parser.add_argument('--oversplit', action='store_true', default=False,
                        help='ablation: train on all data (default: False)')
    parser.add_argument('--aug-model', default='NONE', type=str,
                        help='type of augmentation model NONE/RAND/AUTO/DADA/SHAred/SEParate parameters (default: NONE)')
    parser.add_argument('--los-model', default='NONE', type=str,
                        help='type of model for other loss hyperparams NONE/SOFT/WGHT/BOTH (default: NONE)')
    parser.add_argument('--hyper-opt', default='NONE', type=str,
                        help='type of bilevel optimization NONE/HES (default: NONE)')
    parser.add_argument('--hyper-steps', type=int, default=0, metavar='NUM',
                        help='number of gradient calculations to achieve grad(L_train)=0 (default: 0)')
    parser.add_argument('--hyper-iters', type=int, default=5, metavar='NUM',
                        help='number of approxInverseHVP iterations inside hyperparameter estimation loop (default: 5)')
    parser.add_argument('--hyper-alpha', type=float, default=0.01, metavar='HO',
                        help='hyperparameter learning rate (default: 0.01)')
    parser.add_argument('--hyper-beta', type=int, default=0, metavar='HO',
                        help='hyperparameter beta (default: 0)')
    parser.add_argument('--hyper-gamma', type=int, default=0, metavar='HO',
                        help='hyperparameter gamma (default: 0)')

    args = parser.parse_args()
    
    return args

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    init_seeds(seed=int(time.time()))
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}
    args.hyper_est = True
    args.lr_warm = True
    args.lr_cosine = True
    dataset = args.dataset
    overfit = args.overfit
    oversplit = args.oversplit
    hyper_est = args.hyper_est
    hyper_opt = args.hyper_opt
    imbalance_ratio = args.imbalance_ratio
    subsample_ratio = args.subsample_ratio
    noise_ratio = args.noise_ratio
    model_postfix = 'ir_{}_sr_{}_nr_{}'.format(imbalance_ratio, subsample_ratio, noise_ratio)
    run_folder = args.run_folder
    # create folders
    if not os.path.isdir(args.data):
        os.mkdir(args.data)
    save_folder = '{}/{}'.format(args.data, dataset)
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    long_run_folder = '{}/{}'.format(save_folder, run_folder)
    print('long_run_folder:', long_run_folder)
    if not os.path.isdir(long_run_folder):
        os.mkdir(long_run_folder)
    model_folder = '{}/{}'.format(save_folder, run_folder)
    print('model_folder:', model_folder)
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    # shared among datasets
    task_optimizer = 'sgd'
    task_momentum = 0.9
    task_weight_decay = 0.0001
    task_nesterov = True
    aug_mode = 0
    #
    aug_K, aug_M = 2, 5
    if dataset == 'MNIST':
        total_images = 60000
        valid_images = 10000
        train_images = total_images - valid_images
        num_classes = 10
        num_channels = 1
        hyperEpochStart = 50
        # data:        
        test_data = MNIST(save_folder, train=False, transform=transform_test_mnist, download=True)
        if args.aug_model == 'RAND': # full RandAugment
            transform_train_mnist.transforms.insert(0, RandAugment(3,5))
        elif args.aug_model in ['SHA', 'SEP']: # subset of RandAugment, rest of it in Kornia
            transform_train_mnist.transforms.insert(0, RandSubAugment(3,5))
        train_data = MNIST(save_folder, train=True,  transform=transform_train_mnist, download=True)
        print('TRANSFORM:', transform_train_mnist)
        # ResNet18 model:
        task_lr = 0.05
        train_batch_size = 32
        hyper_batch_size = 256
        args.hyper_theta = ['cls']
        model_name = 'resnet18'
        encoder = EncoderResNet(dataset=dataset, depth=18, num_classes=num_classes).to(device)
        decoder = SupCeResNet(dataset=dataset, depth=18, num_classes=num_classes).to(device)
    elif dataset == 'CIFAR10':
        total_images = 50000
        valid_images = 10000
        train_images = total_images - valid_images
        num_classes = 10
        num_channels = 3
        hyperEpochStart = 50
        # data:
        test_data = CIFAR10(save_folder, train=False, transform=transform_test_cifar10, download=True)
        if args.aug_model == 'RAND': # full RandAugment
            transform_train_cifar10.transforms.insert(0, RandAugment(3,3*5))
        elif args.aug_model in ['SHA', 'SEP']: # subset of RandAugment, rest of it in Kornia
            transform_train_cifar10.transforms.insert(0, RandSubAugment(1,3*5))
        elif args.aug_model == 'AUTO': # Fast AutoAugment
            transform_train_cifar10.transforms.insert(0, AutoAugment(dataset))
        elif args.aug_model == 'DADA': # DadaAugment
            transform_train_cifar10.transforms.insert(0, DadaAugment(dataset))
        train_data = CIFAR10(save_folder, train=True,  transform=transform_train_cifar10, download=True)
        print('TRANSFORM:', transform_train_cifar10)
        # WideResNet model:
        task_lr = 0.1
        train_batch_size = 256
        hyper_batch_size = 256
        args.hyper_theta = ['cls']
        model_name = 'wresnet28_10'
        encoder = EncoderWideResNet(depth=28, widen_factor=10, num_classes=num_classes).to(device)
        decoder = SupCeWideResNet(name=model_name, num_classes=num_classes).to(device)
    elif dataset == 'CIFAR100':
        total_images = 50000
        valid_images = 10000
        train_images = total_images - valid_images
        num_classes = 100
        num_channels = 3
        hyperEpochStart = 50
        # data:
        test_data = CIFAR100(save_folder, train=False, transform=transform_test_cifar100, download=True)
        if args.aug_model == 'RAND': # full RandAugment
            transform_train_cifar100.transforms.insert(0, RandAugment(3,3*5))
        elif args.aug_model in ['SHA', 'SEP']: # subset of RandAugment, rest of it in Kornia
            transform_train_cifar100.transforms.insert(0, RandSubAugment(1,3*5))
        elif args.aug_model == 'AUTO': # Fast AutoAugment
            transform_train_cifar100.transforms.insert(0, AutoAugment(dataset))
        elif args.aug_model == 'DADA': # DadaAugment
            transform_train_cifar100.transforms.insert(0, DadaAugment(dataset))
        train_data = CIFAR100(save_folder, train=True,  transform=transform_train_cifar100, download=True)
        print('TRANSFORM:', transform_train_cifar100)
        # WideResNet model:
        task_lr = 0.1
        train_batch_size = 256
        hyper_batch_size = 256
        args.hyper_theta = ['cls']
        model_name = 'wresnet28_10'
        encoder = EncoderWideResNet(depth=28, widen_factor=10, num_classes=num_classes).to(device)
        decoder = SupCeWideResNet(name=model_name, num_classes=num_classes).to(device)
    elif dataset == 'SVHN' or dataset == 'SVHN_extra':
        num_classes = 10
        num_channels = 3
        extra_svhn = True if 'extra' in dataset else False
        hyperEpochStart = 50
        # data:
        test_data = SVHN(save_folder, split='test',  transform=transform_test_svhn, download=True)
        if args.aug_model == 'RAND': # full RandAugment
            transform_train_svhn.transforms.insert(0, RandAugment(3,7))
        elif args.aug_model in ['SHA', 'SEP']: # subset of RandAugment, rest of it in Kornia
            transform_train_svhn.transforms.insert(0, RandSubAugment(3,7))
        elif args.aug_model == 'AUTO': # Fast AutoAugment
            transform_train_svhn.transforms.insert(0, AutoAugment(dataset))
        elif args.aug_model == 'DADA': # DadaAugment
            transform_train_svhn.transforms.insert(0, DadaAugment(dataset))
        train_data = SVHN(save_folder, split='train', transform=transform_train_svhn, download=True)
        print('TRANSFORM:', transform_train_svhn)
        if extra_svhn:
            total_images = 604388
            valid_images = 104388
            train_images = total_images - valid_images
            extra_data = SVHN(save_folder, split='extra', transform=transform_train_svhn, download=True)
            train_data = ConcatDataset([train_data, extra_data])
        else:
            total_images = 73257
            valid_images = 23257
            train_images = total_images - valid_images
        # WideResNet model:
        task_lr = 0.005
        train_batch_size = 256
        hyper_batch_size = 256
        args.hyper_theta = ['cls']
        model_name = 'wresnet28_10_extra' if extra_svhn else 'wresnet28_10'
        encoder = EncoderWideResNet(depth=28, widen_factor=10, num_classes=num_classes).to(device)
        decoder = SupCeWideResNet(name=model_name, num_classes=num_classes).to(device)
    elif dataset == 'ImageNet':
        aug_mode = 2 # no upscale to save memory
        total_images = 1281167
        valid_images = int(0.2 * total_images) # 20% of train
        train_images = total_images - valid_images
        num_classes = 1000
        num_channels = 3
        hyperEpochStart = 100
        # data:
        test_data = ImageNet(save_folder, split='val', transform=transform_test_imagenet, download=False)
        if args.aug_model == 'RAND': # full RandAugment
            transform_train_imagenet.transforms.insert(0, RandAugment(2,9))
        elif args.aug_model in ['SHA', 'SEP']: # subset of RandAugment, rest of it in Kornia
            transform_train_imagenet.transforms.insert(0, RandSubAugment(2,9))
        elif args.aug_model == 'AUTO': # Fast AutoAugment
            transform_train_imagenet.transforms.insert(0, AutoAugment(dataset))
        elif args.aug_model == 'DADA': # DadaAugment
            transform_train_imagenet.transforms.insert(0, DadaAugment(dataset))
        train_data = ImageNet(save_folder, split='train', transform=transform_train_imagenet, download=False)
        print('TRANSFORM:', transform_train_imagenet)
        # ResNet18 model:
        task_lr = 0.1
        train_batch_size = 256
        hyper_batch_size = 128
        args.hyper_theta = ['cls']
        model_name = 'resnet18'
        encoder = EncoderResNet(dataset=dataset, depth=18, num_classes=num_classes).to(device)
        decoder = SupCeResNet(dataset=dataset, depth=18, num_classes=num_classes).to(device)
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(dataset))
    # dataloaders:
    data_file = '{}/data_{}.pt'.format(model_folder, model_postfix)
    if os.path.isfile(data_file):
        valid_sub_indices, train_sub_indices, train_targets = torch.load(data_file) # load saved indices
    else:
        sss = StratifiedShuffleSplit(n_splits=5, test_size=valid_images, random_state=0)
        sss = sss.split(list(range(total_images)), train_data.targets)
        for _ in range(random.randint(1,5)):
            train_indices, valid_indices = next(sss)
        #
        train_indices, valid_indices = list(train_indices), list(valid_indices)
        valid_sub_indices = valid_indices
        # save targets for soft label estimation
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=False, **kwargs)
        MLEN = len(train_loader.dataset) # dataset size
        BLEN = len(train_loader) # number of batches
        train_targets = torch.zeros(MLEN, dtype=torch.long)
        for batch_idx, data in enumerate(train_loader):
            if batch_idx % args.log_interval == 0:
                print('Reading train batch {}/{}'.format(batch_idx, BLEN))
            _, train_target, train_index = data
            train_targets[train_index] = train_target
        # subsampling
        SR = int(1.0 * train_images * subsample_ratio) # number of subsampled examples
        train_sr_indices = random.sample(train_indices, SR)
        #
        train_sub_data = torch.utils.data.Subset(train_data, train_sr_indices)
        train_sub_loader = torch.utils.data.DataLoader(train_sub_data, batch_size=train_batch_size, shuffle=False, **kwargs)
        SUB = len(train_sub_loader.dataset)
        print('Train dataset/subset: {}->{}'.format(MLEN, SUB))
        # imbalance
        if imbalance_ratio == 1:
            train_sub_indices = train_sr_indices # use all train subsampled data
        else: # distort dataset
            for batch_idx, data in enumerate(train_sub_loader):
                image, target, index = data
                if batch_idx == 0:
                    targets = target
                    indices = index
                else:
                    targets = torch.cat([targets, target])
                    indices = torch.cat([indices, index])
            #
            mskL = targets.lt(num_classes//2) # 0...4
            indL = mskL.nonzero(as_tuple=False).squeeze()
            indicesL = torch.index_select(indices, 0, indL)
            L = indicesL.size(0)
            #
            mskU = targets.ge(num_classes//2) # 5...9
            indU = mskU.nonzero(as_tuple=False).squeeze()
            indicesU = torch.index_select(indices, 0, indU)
            U = indicesU.size(0)
            #
            S = int(1.0 * L / imbalance_ratio) # number of U examples
            indS = torch.tensor(random.sample(range(U), S), dtype=torch.long)
            indicesS = torch.index_select(indicesU, 0, indS)
            #
            train_sub_indices = torch.cat([indicesL, indicesS])
            train_sub_indices = train_sub_indices.tolist()
            print('Imbalance =', L, U, ':', S, '->', L+S)
        # label noise
        if noise_ratio > 0.0:
            num_noisy_labels = round(noise_ratio*len(train_sub_indices))
            noisy_sub_indices = random.sample(train_sub_indices, num_noisy_labels)
            train_targets[noisy_sub_indices] = torch.randint(num_classes, (num_noisy_labels,), dtype=torch.long)
            print('Noisy labels: {:.0f}% ({}/{})'.format(100.0*len(noisy_sub_indices)/len(train_sub_indices), len(noisy_sub_indices), len(train_sub_indices)))
        # save indices
        with open(data_file, 'wb') as f:
            torch.save((valid_sub_indices, train_sub_indices, train_targets), f)
    # samplers
    print('Valid/Train Split: {}/{}'.format(len(valid_sub_indices), len(train_sub_indices)))
    # loaders
    train_sub_data = torch.utils.data.Subset(train_data, train_sub_indices)
    valid_sub_data = torch.utils.data.Subset(train_data, valid_sub_indices)
    if overfit:
        test_loader  = torch.utils.data.DataLoader(test_data,      batch_size=train_batch_size, shuffle=False, **kwargs)
        valid_loader = torch.utils.data.DataLoader(test_data,      batch_size=hyper_batch_size, shuffle=True, drop_last=True, **kwargs)
        train_loader = torch.utils.data.DataLoader(train_sub_data, batch_size=train_batch_size, shuffle=True, drop_last=True, **kwargs)
        hyper_loader = torch.utils.data.DataLoader(train_sub_data, batch_size=hyper_batch_size, shuffle=True, drop_last=True, **kwargs)        
    elif oversplit:
        test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=train_batch_size, shuffle=False, **kwargs)
        valid_loader = torch.utils.data.DataLoader(train_data, batch_size=hyper_batch_size, shuffle=True, drop_last=True, **kwargs)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True, drop_last=True, **kwargs)
        hyper_loader = torch.utils.data.DataLoader(train_data, batch_size=hyper_batch_size, shuffle=True, drop_last=True, **kwargs)            
    else:
        test_loader  = torch.utils.data.DataLoader(test_data,      batch_size=train_batch_size, shuffle=False, **kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_sub_data, batch_size=hyper_batch_size, shuffle=True, drop_last=True, **kwargs)
        train_loader = torch.utils.data.DataLoader(train_sub_data, batch_size=train_batch_size, shuffle=True, drop_last=True, **kwargs)
        hyper_loader = torch.utils.data.DataLoader(train_sub_data, batch_size=hyper_batch_size, shuffle=True, drop_last=True, **kwargs)
    # train data augmentation model
    if hyper_opt in ['NONE', 'RAND']:
        hyperGradEnable = False
    elif (hyper_opt in ['HES']) and hyper_est:
        hyperGradEnable = True
    elif (hyper_opt in ['HES']) and not(hyper_est):
        hyperGradEnable = False
    else:
        raise NotImplementedError('{} is not supported hyper optimization model!'.format(hyper_opt))
    # save other hyperparameters to arguments
    args.hyper_lr = 0.05
    if dataset == 'ImageNet':
        args.hyper_lr = 0.01
    else:
        args.hyper_lr = 0.05
    args.hyper_start = hyperEpochStart
    args.lr = task_lr
    args.train_batch_size = train_batch_size
    args.num_classes = num_classes
    # task optimizer
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list()
    args.hyper_lr_decay_epochs = list()
    for i in iterations:
        args.lr_decay_epochs.append(int(i))
        args.hyper_lr_decay_epochs.append(int(i)-args.hyper_start)
    args.hyper_epochs = args.epochs-args.hyper_start
    if args.lr_warm:
        args.lr_warmup_from = args.lr/10.0
        args.hyper_lr_warmup_from = args.hyper_lr/10.0
        if args.lr_cosine:
            eta_min = args.lr * (args.lr_decay_rate ** 3)
            args.lr_warmup_to = eta_min + (args.lr - eta_min) * (1 + math.cos(math.pi * args.lr_warm_epochs / args.epochs)) / 2
            hyper_eta_min = args.hyper_lr * (args.lr_decay_rate ** 3)
            args.hyper_lr_warmup_to = hyper_eta_min + (args.hyper_lr - hyper_eta_min) * (1 + math.cos(math.pi * args.lr_warm_epochs / args.epochs)) / 2
        else:
            args.lr_warmup_to = args.lr
            args.hyper_lr_warmup_to = args.hyper_lr
    #
    if task_optimizer == 'sgd':
        params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = optim.SGD(params, lr=args.lr, momentum=task_momentum, weight_decay=task_weight_decay, nesterov=task_nesterov)
    else:
        raise NotImplementedError('{} is not supported task optimizer!'.format(task_optimizer))
    # list model layers
    #for n, p in encoder.named_parameters():
    #    print (n, p.data.shape)
    for n, p in decoder.named_parameters():
        print (n, p.data.shape)
    # hyper models
    T = total_images
    L = len(test_loader.dataset)
    M = len(valid_loader.dataset)
    N = len(train_loader.dataset)
    print('Test/Valid/Train Split: {}/{}/{} out of total {} train images'.format(L,M,N,T))
    # validation data loss/augmentation model
    if hyper_est:
        validLosModel = LossModel(N=1, C=num_classes, init_targets=list(), apply=False, model='NONE', grad=False, sym=False, device=device).to(device)
        validAugModel = AugmentModel(N=1, magn=aug_M, apply=False, mode=aug_mode, grad=False, device=device).to(device)
    # train data loss/augmentation models
    symmetricKlEnable = False if (imbalance_ratio == 1) and (noise_ratio == 0.0) else True
    trainLosModel = LossModel(N=T, C=num_classes, init_targets=train_targets, apply=True, model=args.los_model, grad=hyperGradEnable, sym=symmetricKlEnable, device=device).to(device)
    # select model
    if   args.aug_model in ['NONE', 'RAND', 'AUTO', 'DADA']:
        trainAugModel = AugmentModel(N=1, magn=aug_M, apply=False, mode=aug_mode, grad=False,           device=device).to(device)
    elif args.aug_model == 'SHA':
        trainAugModel = AugmentModel(N=1, magn=aug_M, apply=True,  mode=aug_mode, grad=hyperGradEnable, device=device).to(device)
    elif args.aug_model == 'SEP':
        trainAugModel = AugmentModel(N=T, magn=aug_M, apply=True,  mode=aug_mode, grad=hyperGradEnable, device=device).to(device)
    else:
        raise NotImplementedError('{} is not supported train augmentation model!'.format(args.aug_model))
    # hyperoptimizer
    hyperParams = list(trainLosModel.parameters()) + list(trainAugModel.parameters())
    hyperOptimizer = optim.RMSprop(hyperParams, lr=args.hyper_lr)
    hyperScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(hyperOptimizer, args.epochs-args.hyper_start)
    # initial step to save pretrained model
    best_acc = 0.0
    run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    if overfit:
        model_name = 'overfit_' + model_name
    if oversplit:
        model_name = 'oversplit_' + model_name
    run_name = '{}_opt_{}_est_{}_aug_model_{}_los_model_{}_{}'.format(
        model_name, hyper_opt, hyper_est, args.aug_model, args.los_model, model_postfix)
    writer = SummaryWriter('./logs/{}/{}_{}_{}'.format(dataset, run_folder, run_name, run_date))
    checkpoint_file = '{}/best_{}.pt'.format(model_folder, run_name)
    # load hypermodel with estimated hyperparameters
    if not(hyper_est):
        load_name = '{}_opt_{}_est_{}_aug_model_{}_los_model_{}_{}'.format(
            model_name, hyper_opt, 'True', args.aug_model, args.los_model, model_postfix)
        load_file = '{}/best_{}.pt'.format(model_folder, load_name)
        checkpoint = torch.load(load_file)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        trainLosModel.load_state_dict(checkpoint['reweight_state_dict'])
        trainAugModel.load_state_dict(checkpoint['augment_state_dict'])
        print('Loading pretrained model...', load_file)
    print('Run: {}/{} - {}\n'.format(model_folder, run_name, run_date))
    dDivs = 4*[0.0]
    for epoch in range(0, args.epochs):
        print('Run {}/{} - {}: {:.0f}% ({}/{})'.format(model_folder, run_name, run_date, 100.0*epoch/args.epochs, epoch, args.epochs))
        adjust_learning_rate(args, optimizer, epoch)
        testEnable  = True #if  (epoch >= hyperEpochStart) else False
        hyperEnable = True if ((epoch >  hyperEpochStart) and hyperGradEnable)  else False
        if not(hyper_est): # train classifier only
            train_loss = classTrain(args, encoder, decoder, optimizer, device, train_loader, epoch, trainLosModel, trainAugModel)
        else:
            # train hyperparameters
            if hyper_opt == 'HES' and hyperEnable:
                hyper_adjust_learning_rate(args, hyperOptimizer, epoch-hyperEpochStart)
                dDivs = hyperHesTrain(args, encoder, decoder, optimizer, device, valid_loader, hyper_loader, epoch, hyperEpochStart,
                            trainLosModel, trainAugModel, validLosModel, validAugModel, hyperOptimizer)
            # train encoder and classifier
            train_loss = innerTrain(args, encoder, decoder, optimizer, device, train_loader, epoch, trainLosModel, trainAugModel)
        # test
        if testEnable:
            acc, test_loss, _ = innerTest(args, encoder, decoder, device, test_loader, epoch)
            # save checkpoint (acc-based)
            if acc >= best_acc:
                print('SAVING trained model at epoch {} with {:.2f}% accuracy'.format(epoch, acc))
                save(encoder, decoder, trainLosModel, trainAugModel, acc, epoch, checkpoint_file)
                best_acc = acc
        else:
            acc, test_loss = 0.0, 0.0
        # save log
        writer.add_scalar('Accuracy', acc, epoch)
        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Test Loss', test_loss, epoch)
    #
    print('BEST trained model has {:.2f}% accuracy'.format(best_acc))
    writer.flush()
    writer.close()

if __name__ == '__main__':
    args = get_args()
    main(args)
