from __future__ import print_function
import argparse, os, sys, random, time, datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
#
import torch
import torch.nn as nn
import torch.nn.functional as F
#
from custom_models import *
from custom_datasets import *
from custom_transforms import *
from utils import *

# sklearn
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
# RAPIDS AI (much faster t-SNE we used in the paper)
#from cuml import TSNE


def get_args():
    parser = argparse.ArgumentParser(description='Visualization of AutoDO')
    parser.add_argument('--data', default='./local_data', type=str, metavar='NAME',
                        help='folder to save all data')
    parser.add_argument('--dataset', default='MNIST', type=str, metavar='NAME',
                        help='dataset MNIST/CIFAR10/CIFAR100/SVHN/SVHN_extra/ImageNet')
    parser.add_argument('--workers', default=4, type=int, metavar='NUM',
                        help='number of data loading workers (default: 4)')
    parser.add_argument("--gpu", default='0', type=str, metavar='NUM',
                        help='GPU device number')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=500, metavar='NUM',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-ir', '--imbalance-ratio', type=int, default=1, metavar='N',
                        help='ratio of [1:C/2] to [C/2+1:C] labels in the training dataset drawn from uniform distribution')
    parser.add_argument('-sr', '--subsample-ratio', type=float, default=1.0, metavar='N',
                        help='ratio of selected to total labels in the training dataset drawn from uniform distribution')
    parser.add_argument('-nr', '--noise-ratio', type=float, default=0.0, metavar='N',
                        help='ratio of noisy (randomly flipped) labels (default: 0.0)')
    parser.add_argument('-r', '--run-folder', default='run0', type=str,
                        help='dir to save run')
    parser.add_argument('--plot-cmx', action='store_true', help='confusion matrix or TSNE')
    
    args = parser.parse_args()

    return args


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    init_seeds(seed=int(time.time()))
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}
    args.hyper_est = True
    dataset = args.dataset
    imbalance_ratio = args.imbalance_ratio
    subsample_ratio = args.subsample_ratio
    noise_ratio = args.noise_ratio
    model_postfix = 'ir_{}_sr_{}_nr_{}'.format(imbalance_ratio, subsample_ratio, noise_ratio)
    run_folder = args.run_folder
    save_folder = '{}/{}'.format(args.data, dataset)
    model_folder = '{}/{}'.format(save_folder, run_folder)
    assert os.path.isdir(model_folder), 'Error: {} model folder is not found!'.format(model_folder)
    print('model_folder:', model_folder)
    plot_cmx = args.plot_cmx
    results_folder = './visualizations'
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
    #
    if   dataset == 'MNIST':
        total_images = 60000
        valid_images = 10000
        train_images = total_images - valid_images
        num_classes = 10
        num_channels = 1
        # model:
        model_name = 'resnet18'
        encoder = EncoderResNet(dataset=dataset, depth=18, num_classes=num_classes).to(device)
        decoder = SupCeResNet(dataset=dataset, depth=18, num_classes=num_classes).to(device)
        # dataloaders
        train_batch_size = 1000
        test_data  = MNIST(save_folder, train=False, transform=transform_test_mnist)
        train_data = MNIST(save_folder, train=True,  transform=transform_train_mnist)
    elif dataset == 'SVHN':
        total_images = 73257
        valid_images = 23257
        train_images = total_images - valid_images
        num_classes = 10
        num_channels = 3
        # model:
        train_batch_size = 256
        model_name = 'wresnet28_10'
        encoder = EncoderWideResNet(depth=28, widen_factor=10, num_classes=num_classes).to(device)
        decoder = SupCeWideResNet(name=model_name, num_classes=num_classes).to(device)
        # data:
        test_data  = SVHN(save_folder, split='test',  transform=transform_test_svhn,  download=True)
        train_data = SVHN(save_folder, split='train', transform=transform_train_svhn, download=True)
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(dataset))
    # dataloaders:
    data_file = '{}/data_{}.pt'.format(model_folder, model_postfix)
    if os.path.isfile(data_file):
        _, train_sub_indices, train_targets = torch.load(data_file) # load saved indices
    else:
        raise NotImplementedError('{} is missing!'.format(data_file))
    #
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size=train_batch_size, shuffle=False, **kwargs)
    #train_sub_data = torch.utils.data.Subset(train_data, train_sub_indices)
    #train_loader = torch.utils.data.DataLoader(train_sub_data, batch_size=train_batch_size, shuffle=False, **kwargs)
    #
    args.train_batch_size = train_batch_size
    args.num_classes = num_classes
    # list model layers
    for n, p in encoder.named_parameters():
        print (n, p.data.shape)
    F = p.data.shape[0] # last layer dimension
    for n, p in decoder.named_parameters():
        print (n, p.data.shape)
    # hyper models
    C = num_classes
    T = total_images
    M = len(valid_loader.dataset)
    N = 0 #len(train_loader.dataset)
    print('Train/Test Split: {}/{} out of total {} images'.format(M, N, T))
    # CFGS:
    cfgs = list()
    # format: [hyper_opt, hyper_est, aug_model, los_model]
    #cfgs.append(['RAND', True, 'RAND', 'NONE'])
    cfgs.append(['NONE', True, 'NONE', 'NONE'])
    cfgs.append(['NONE', True, 'AUTO', 'NONE'])
    cfgs.append([ 'HES', True,  'SEP', 'BOTH'])
    E = len(cfgs)
    # experiment name
    exp_name = 'debug'
    if plot_cmx:
        results_file = '{}_cmat_{}'.format(dataset, exp_name)
    else:
        results_file = '{}_tsne_{}'.format(dataset, exp_name)
    fvs_2d_file = '{}/fvs_2d_tsne_{}.npy'.format(results_folder, exp_name)
    gts_file    = '{}/gts_tsne_{}.npy'.format(results_folder, exp_name)
    prs_file    = '{}/prs_tsne_{}.npy'.format(results_folder, exp_name)
    szs_file    = '{}/szs_tsne_{}.npy'.format(results_folder, exp_name)
    cmx_file    = '{}/cmx_tsne_{}.npy'.format(results_folder, exp_name)
    if os.path.isfile(fvs_2d_file):
        fvs_2d = np.load(fvs_2d_file)
        gts    = np.load(gts_file)
        prs    = np.load(prs_file)
        szs    = np.load(szs_file)
        cmx    = np.load(cmx_file)
    else:
        fvs    = np.empty([E,M+N,F])
        fvs_2d = np.empty([E,M+N,2])
        gts = np.empty([E,M+N], dtype=int)
        prs = np.empty([E,M+N])
        szs = np.empty([E,M+N])
        cmx = np.empty([E,C,C])
        for i, cfg in enumerate(cfgs):
            hyper_opt, hyper_est, aug_model, los_model = cfg
            run_name = '{}_opt_{}_est_{}_aug_model_{}_los_model_{}_{}'.format(
                model_name, hyper_opt, hyper_est, aug_model, los_model, model_postfix)
            checkpoint_file = '{}/best_{}.pt'.format(model_folder, run_name)
            if os.path.isfile(checkpoint_file):
                print('Loading pretrained models...', checkpoint_file)
                checkpoint = torch.load(checkpoint_file)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                decoder.load_state_dict(checkpoint['decoder_state_dict'])
            else:
                raise NotImplementedError('{} checkpoint is missing!'.format(checkpoint_file))
            # test data
            fv, gt, pr, mi = vizStat(args, encoder, decoder, device, valid_loader, M, F)
            sz = torch.zeros(M).to(device)
            sz[mi] = 1.0
            fv, gt, pr, sz = fv.cpu().numpy(), gt.cpu().numpy(), pr.cpu().numpy(), sz.cpu().numpy()
            fvs[i,:M] = fv
            gts[i,:M] = gt
            prs[i,:M] = pr
            szs[i,:M] = sz
            cm = confusion_matrix(gt, pr)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cmx[i] = cm
            ## train data
            #fv, gt, pr, _ = vizStat(args, encoder, decoder, device, train_loader, T, F)
            #train_idx = torch.tensor(train_sub_indices, dtype=torch.long, device=device)
            ##print(train_idx.shape)
            #fv, gt, pr, sz = fv[train_idx].cpu().numpy(), gt[train_idx].cpu().numpy(), pr[train_idx].cpu().numpy(), sz.cpu().numpy()
            ##print(fv.shape, gt.shape, pr.shape)
            #fvs[i,M:M+N] = fv
            #gts[i,M:M+N] = gt
            #prs[i,M:M+N] = pr
            #szs[i,M:M+N] = sz
            fvs_2d[i] = TSNE(n_components=2, verbose=1).fit_transform(fvs[i])
            print('{}-{} : {}/{}/{}/{}'.format(i, run_name, fvs.shape, gts.shape, prs.shape, szs.shape))
        # save results
        #fvs_2d = TSNE(n_components=2, verbose=1).fit_transform(fvs.reshape((E*(M+N),F))).reshape((E,M+N,2))
        np.save(fvs_2d_file, fvs_2d)
        np.save(gts_file, gts)
        np.save(prs_file, prs)
        np.save(szs_file, szs)
        np.save(cmx_file, cmx)
    #
    print('Check output:', fvs_2d.shape)
    pdf = PdfPages('{}/{}'.format(results_folder, results_file+'.pdf'))
    np.set_printoptions(precision=2)
    fontText = 16
    S = 9
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'figure.dpi': 600})
    plt.rcParams.update({'savefig.dpi': 600})
    fig, ax = plt.subplots(1, E, sharex=True, sharey=True)
    fig.set_size_inches(np.array([E*5+1,6]), forward=True)
    if plot_cmx:
        print('Plotting confusion matrix')
        fmt = '.1f'
        for e, cfg in enumerate(cfgs):
            cm = 100.0*cmx[e]
            th = cm.max() / 2.0
            ax[e].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            for i in range(C):
                for j in range(C):
                    ax[e].text(j, i, format(cm[i,j], fmt), ha="center", va="center", color="white" if cm[i,j] > th else "black", fontsize = 10)
            #ax[e].text(0.25, 10.4, t[e], rotation = 0, fontsize = fontText)
            plt.tight_layout()
        ax[0].set(xticks=np.arange(C), yticks=np.arange(C), xlabel='Predicted label, (a)', ylabel='True label')
        ax[1].set(xticks=np.arange(C), yticks=np.arange(C), xlabel='Predicted label, (b)')
        ax[2].set(xticks=np.arange(C), yticks=np.arange(C), xlabel='Predicted label, (c)')
    else:
        print('Plotting t-sne')
        t = ['(a)', '(b)', '(c)']
        for e, cfg in enumerate(cfgs):
            im = ax[e].scatter(fvs_2d[e,:,0], fvs_2d[e,:,1], cmap=plt.get_cmap('tab10'), c=gts[e], s=S*szs[e]+1)
            ax[e].text(-240, -245, t[e], rotation = 0, fontsize = fontText)
            plt.tight_layout()
            plt.xlim(-255, 255)
            plt.ylim(-260, 260)
        #fig.colorbar(im, ax=ax.ravel().tolist(), cmap=plt.get_cmap('tab10'), orientation='horizontal', shrink = 0.25, spacing = 'proportional', drawedges = True)
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.91, 0.1, 0.01, 0.84])
        fig.colorbar(im, cax=cbar_ax, cmap=plt.get_cmap('tab10'), orientation='vertical')
        #5
        ellipse = Ellipse(xy=(-60.0, -25.0), width=250.0, height=250.0, angle=  0.0, edgecolor='k', fc='None', lw=2, ls='--')
        ax[0].add_patch(ellipse)
        ellipse = Ellipse(xy=( 35.0, -15.0), width=210.0, height=210.0, angle=  0.0,  edgecolor='k', fc='None', lw=2, ls='--')
        ax[1].add_patch(ellipse)
        ellipse = Ellipse(xy=(-90.0, -10.0), width=125.0, height=125.0, angle=  0.0,  edgecolor='k', fc='None', lw=2, ls='--')
        ax[2].add_patch(ellipse)
        #
    plt.savefig('{}/{}'.format(results_folder, results_file+'.png'), bbox_inches='tight')
    #plt.savefig('{}/{}'.format(results_folder, results_file+'.svg'), format="svg", bbox_inches='tight')
    pdf.savefig(bbox_inches='tight')
    pdf.close()
    print('Plot done!')

if __name__ == '__main__':
    args = get_args()
    main(args)
