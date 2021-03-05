import math
import numpy as np
import torch


def save(encoder, decoder, reweight_model, augment_model, acc, epoch, checkpoint_file):
    state = {   'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'reweight_state_dict': reweight_model.state_dict(),
                'augment_state_dict': augment_model.state_dict(),
                'acc': acc,
                'epoch': epoch}
    torch.save(state, checkpoint_file)


def exp_lr_scheduler(optimizer, epoch, lr_decay, lr_decay_epoch):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if (epoch % lr_decay_epoch) or (epoch == 0):
        return optimizer
    
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer


def exp_lr_reset(optimizer, lr):
    """Reset LR"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.lr_cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch >= np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.lr_warm and epoch < args.lr_warm_epochs:
        p = (batch_id + epoch * total_batches) / \
            (args.lr_warm_epochs * total_batches)
        lr = args.lr_warmup_from + p * (args.lr_warmup_to - args.lr_warmup_from)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    #
    for param_group in optimizer.param_groups:
        lrate = param_group['lr']
    return lrate


def hyper_adjust_learning_rate(args, optimizer, epoch):
    lr = args.hyper_lr
    if args.lr_cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.hyper_epochs)) / 2
    else:
        steps = np.sum(epoch >= np.asarray(args.hyper_lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def hyper_warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.lr_warm and epoch < args.lr_warm_epochs:
        p = (batch_id + epoch * total_batches) / \
            (args.lr_warm_epochs * total_batches)
        lr = args.hyper_lr_warmup_from + p * (args.hyper_lr_warmup_to - args.hyper_lr_warmup_from)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    #
    for param_group in optimizer.param_groups:
        lrate = param_group['lr']
    return lrate

