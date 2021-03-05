import sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import kornia
import math
from utils import *
from custom_models.utils import *
from custom_transforms import plot_debug_images

_SOFTPLUS_UNITY_ = 1.4427
_SIGMOID_UNITY_ = 2.0
_EPS_ = 1e-8 # small regularization constant

__all__ = ['innerTest', 'innerTrain', 'classTrain', 'vizStat',
            'LossModel', 'AugmentModelNONE', 'AugmentModel', 'hyperHesTrain']


def metricCE(logit, target):
    return F.cross_entropy(logit, target)


class LossModel(nn.Module):
    def __init__(self, N, C, init_targets, apply, model, grad, sym, device):
        super(LossModel, self).__init__()
        self.alpha = 0.1
        self.apply = apply
        self.sym = sym
        self.act = nn.Softplus()
        self.actSoft     = nn.Softmax(dim=1)
        self.actLogSoft  = nn.LogSoftmax(dim=1)
        if self.apply:
            initWeights = torch.zeros(N)
            eps = 0.05 # alpha label-smoothing constant [0.05-0.2]
            diff = math.log(1.0-C+C/eps) # solution for softmax
            initSoftTargets = diff*(F.one_hot(init_targets, num_classes=C)-0.5)
            if model == 'NONE':
                self.hyperW = nn.Parameter(initWeights,     requires_grad=False)
                self.hyperS = nn.Parameter(initSoftTargets, requires_grad=False)
                self.soft = False
                self.cls_loss = nn.CrossEntropyLoss(reduction='none')
            elif model == 'WGHT':
                self.hyperW = nn.Parameter(initWeights,     requires_grad=grad)
                self.hyperS = nn.Parameter(initSoftTargets, requires_grad=False)
                self.soft = False
                self.cls_loss = nn.CrossEntropyLoss(reduction='none')
            elif model == 'SOFT':
                self.hyperW = nn.Parameter(initWeights,     requires_grad=False)
                self.hyperS = nn.Parameter(initSoftTargets, requires_grad=grad)
                self.soft = True
                self.cls_loss = nn.KLDivLoss(reduction='none')
            elif model == 'BOTH':
                self.hyperW = nn.Parameter(initWeights,     requires_grad=grad)
                self.hyperS = nn.Parameter(initSoftTargets, requires_grad=grad)
                self.soft = True
                self.cls_loss = nn.KLDivLoss(reduction='none')
            else:
                raise NotImplementedError('{} is not supported loss model'.format(model))
        else:
            self.cls_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, idx, logit, target):
        if self.apply: # always train
            hyperW = _SOFTPLUS_UNITY_ * self.act(self.hyperW[idx])
            hyperS = self.hyperS[idx]
            hyperH = torch.argmax(hyperS, dim=1)
            if self.soft:
                if self.sym:
                    cls_loss = 0.5*torch.sum(self.cls_loss(self.actLogSoft(logit[0]), self.actSoft(hyperS)) +
                                             self.cls_loss(self.actLogSoft(hyperS), self.actSoft(logit[0])), dim=1)
                else:
                    cls_loss = 1.0*torch.sum(self.cls_loss(self.actLogSoft(logit[0]), self.actSoft(hyperS)), dim=1)
            else:
                cls_loss = self.cls_loss(logit[0], hyperH)
            cls_loss = hyperW*cls_loss
            cls_loss = cls_loss.mean()
            return cls_loss
        else: # always valid
            cls_loss = self.cls_loss(logit[0], target)
            cls_loss = cls_loss.mean()
            return cls_loss


class AugmentModelNONE(nn.Module):
    def __init__(self):
        super(AugmentModelNONE, self).__init__()

    def forward(self, idx, x):
        return x


class AugmentModel(nn.Module):
    def __init__(self, N, magn, apply, mode, grad, device):
        super(AugmentModel, self).__init__()
        # enable/disable manual augmentation
        self.N = N
        self.apply = apply
        self.device = device
        self.mode = mode # mode
        self.K = 7 # number of affine magnitude params
        self.J = 0 # number of color magnitude params
        K = self.K
        J = self.J
        magnNorm = torch.ones(1)*magn/10.0 # normalize to 10 like in RandAugment
        probNorm = torch.ones(1)*1/(K-2) # 1/(K-2) probability
        magnLogit = torch.log(magnNorm/(1-magnNorm)) # convert to logit
        probLogit = torch.log(probNorm/(1-probNorm)) # convert to logit
        # affine transforms (mid, range)
        self.angle = [0.0, 30.0] # [-30.0:30.0] rotation angle
        self.trans = [0.0, 0.45] # [-0.45:0.45] X/Y translate
        self.shear = [0.0, 0.30] # [-0.30:0.30] X/Y shear
        self.scale = [1.0, 0.50] # [ 0.50:1.50] X/Y scale
        # color transforms (mid, range)
        #self.bri = [0.0, 0.9] # [-0.9:0.9] brightness
        #self.con = [1.0, 0.9] # [0.1:1.9] contrast
        #self.sat = [0.1, 1.9] # [-0.30:0.30] saturation
        #self.hue = [1.0, 0.50] # [ 0.70:1.30] hue
        #self.gam = [1.0, 0.50] # [ 0.70:1.30] gamma
        #
        self.actP = nn.Sigmoid()
        self.actM = nn.Sigmoid()
        self.paramP = nn.Parameter(probLogit*torch.ones(K+J,N), requires_grad=grad)
        self.paramM = nn.Parameter(magnLogit*torch.ones(K+J,N), requires_grad=grad)

    def forward(self, idx, x):
        B,C,H,W = x.shape
        device = self.device
        mode = self.mode
        if self.apply:
            K = self.K
            J = self.J
            # learnable hyperparameters
            if self.N == 1:
                paramPos = torch.log(    self.actP(self.paramP)).repeat(1,B) # [-Inf:0]
                paramNeg = torch.log(1.0-self.actP(self.paramP)).repeat(1,B) # [-Inf:0]
                paramM = self.actM(self.paramM).repeat(1,B) # (K+J)xB [0:1], default=magn
            else:
                paramPos = torch.log(    self.actP(self.paramP[:,idx])) # [-Inf:0]
                paramNeg = torch.log(1.0-self.actP(self.paramP[:,idx])) # [-Inf:0]
                paramM = self.actM(self.paramM[:,idx]) # (K+J)xB [0:1], default=magn
            paramP = torch.cat([paramPos.view(-1,1), paramNeg.view(-1,1)], dim=1) # B*(K+J)x2
            # reparametrize probabilities and magnitudes
            sampleP = F.gumbel_softmax(paramP, tau=1.0, hard=True).to(device) # B*(K+J)x2
            sampleP = sampleP[:,0]
            sampleP = sampleP.reshape(K,B)
            # reparametrize magnitudes
            #sampleM = paramM[:K] * torch.rand(K,B).to(device) # KxB, prior: U[0,1]
            sampleM = paramM[:K] * torch.randn(K,B).to(device) # KxB, prior: N(0,1)
            # affine augmentations
            R: torch.tensor = torch.zeros(B,3,3).to(device) + torch.eye(3).to(device)
            # define the rotation angle
            ANG: torch.tensor = sampleP[0] * sampleM[0] * self.angle[1] # B(0/1)*U[0,1]*M/10
            # define the rotation center
            CTR: torch.tensor = torch.cat((W*torch.ones(B).to(device)//2, H*torch.ones(B).to(device)//2)).view(-1,2)
            # define the scale factor
            SCL: torch.tensor = torch.zeros_like(CTR).to(device)
            SCL[:,0] = self.scale[0] + sampleP[5] * sampleM[5] * self.scale[1] # mid + B(0/1)*U[0,1]*M/10
            SCL[:,1] = self.scale[0] + sampleP[6] * sampleM[6] * self.scale[1] # mid + B(0/1)*U[0,1]*M/10
            R[:,0:2] = kornia.get_rotation_matrix2d(CTR, ANG, SCL)
            # translation: border not defined yet
            T: torch.tensor = torch.zeros_like(R) + torch.eye(3).to(device)
            T[:,0,2] = W * sampleP[1] * sampleM[1] * self.trans[1]
            T[:,1,2] = H * sampleP[2] * sampleM[2] * self.trans[1]
            # shear: check this
            S: torch.tensor = torch.zeros_like(R) + torch.eye(3).to(device)
            S[:,0,1] = sampleP[3] * sampleM[3] * self.shear[1]
            S[:,1,0] = sampleP[4] * sampleM[4] * self.shear[1]
            # apply the transformation to original image
            M: torch.tensor = torch.bmm(torch.bmm(S,T),R)
            if mode == 0: #upscale
                x = kornia.geometry.resize(x, (4*H, 4*W))
                x_warped: torch.tensor = kornia.warp_perspective(x, M, dsize=(4*H,4*W), border_mode='border')
                x_warped = kornia.geometry.resize(x_warped, (H,W))
            else:
                x_warped: torch.tensor = kornia.warp_perspective(x, M, dsize=(H,W), border_mode='border')
            ## color augmentations
            #if mode == 1:
            #    BRI: torch.tensor = self.bri[0] + sampleP[6] * sampleC[0] * self.bri[1] # mid + B(0/1)*U[0,1]*M/10
            #    CON: torch.tensor = self.con[0] + sampleP[7] * sampleC[1] * self.con[1]
            #    x_color = kornia.adjust_brightness(kornia.adjust_contrast(x_warped, CON), BRI)
            #else:
            #    x_color = x_warped
            
            return x_warped

        else: # process val to compensate for Kornia artifacts!
            if mode == 0: #upscale
                M: torch.tensor = torch.zeros(B,3,3).to(device) + torch.eye(3).to(device)
                x = kornia.geometry.resize(x, (4*H, 4*W))
                x_warped: torch.tensor = kornia.warp_perspective(x, M, dsize=(4*H,4*W), border_mode='border')
                x_warped = kornia.geometry.resize(x_warped, (H,W))
            else:
                x_warped = x #torch.tensor = kornia.warp_perspective(x, M, dsize=(H,W), border_mode='border')
            ## color augmentations
            #if mode == 1:
            #    BRI: torch.tensor = torch.zeros(B).to(device)
            #    CON: torch.tensor = torch.ones(B).to(device)
            #    x_color = kornia.adjust_brightness(kornia.adjust_contrast(x_warped, CON), BRI)
            #else:
            #    x_color = x_warped
            
            return x_warped


def hyperHesTrain(args, encoder, decoder, optimizer, device, valid_loader, train_loader, epoch, start,
        trainLosModel, trainAugModel, validLosModel, validAugModel, hyperOptimizer):
    encoder.eval()
    decoder.eval()
    validLosModel.eval()
    validAugModel.eval()
    trainLosModel.train()
    trainAugModel.train()
    M = len(valid_loader.dataset)
    N = len(train_loader.dataset)
    B = len(train_loader) # number of batches
    v_loader_iterator = iter(valid_loader)
    t_loader_iterator = iter(train_loader)
    dDivs = 4*[0.0]
    #
    for param_group in optimizer.param_groups:
        task_lr = param_group['lr']
    hyperParams = list()
    for n,p in trainAugModel.named_parameters():
        if p.requires_grad:
            hyperParams.append(p)
    for n,p in trainLosModel.named_parameters():
        if p.requires_grad:
            hyperParams.append(p)
    theta = list()
    for n,p in decoder.named_parameters():
        if (len(args.hyper_theta) == 0) or (any([e in n for e in args.hyper_theta]) and ('.weight' in n)):
            theta.append(p)
    #
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    v0Norms    = AverageMeter()
    v1Norms    = AverageMeter()
    mvpNorms   = AverageMeter()
    end = time.time()
    ######## start train hyper model ########
    for batch_idx in range(0,1*B):
        # sample val batch
        try:
            vData, vTarget, vIndex = next(v_loader_iterator)
        except StopIteration:
            v_loader_iterator = iter(valid_loader)
            vData, vTarget, vIndex = next(v_loader_iterator)
        # sample train batch
        try:
            tData, tTarget, tIndex = next(t_loader_iterator)
        except StopIteration:
            t_loader_iterator = iter(train_loader)
            tData, tTarget, tIndex = next(t_loader_iterator)
        #
        tData  = tData.to(device)
        tTarget= tTarget.to(device)
        tIndex = tIndex.to(device)
        vData  = vData.to(device)
        vTarget= vTarget.to(device)
        vIndex = vIndex.to(device)
        # measure data loading time
        data_time.update(time.time() - end)
        # warm-up learning rate
        hyper_lr = hyper_warmup_learning_rate(args, epoch-start, batch_idx, B, hyperOptimizer)
        # v1 = dL_v / dTheta: Lx1
        optimizer.zero_grad()
        vData = validAugModel(vIndex, vData)
        vEncode = encoder(vData)
        vOutput = decoder(vEncode)
        vLoss = validLosModel(vIndex, vOutput, vTarget)
        g1 = torch.autograd.grad(vLoss, theta)
        v1 = [e.detach().clone() for e in g1]
        # v0 = dL_t / dTheta: Lx1
        optimizer.zero_grad()
        tData = trainAugModel(tIndex, tData)
        tEncode = encoder(tData)
        tOutput = decoder(tEncode)
        tLoss = trainLosModel(tIndex, tOutput, tTarget)
        g0 = torch.autograd.grad(tLoss, theta, create_graph=True)
        v0 = [e.detach().clone() for e in g0]
        # v2 = H^-1 * v0: Lx1
        v2 = [-e.detach().clone() for e in v1]
        if args.hyper_iters > 0: # Neumann series
            for j in range(0, args.hyper_iters):
                ns = torch.autograd.grad(g0, theta, grad_outputs=v1, create_graph=True)
                v1 = [v1[l] - args.hyper_alpha*e for l,e in enumerate(ns)]
                v2 = [v2[l] - e.detach().clone() for l,e in enumerate(v1)]
        # MVP compute
        v0Norm = torch.sum(torch.cat([t.detach().clone().view(-1)*v.detach().clone().view(-1) for t,v in zip(v0,v0)])) # gLt*gLt
        v1Norm = torch.sum(torch.cat([t.detach().clone().view(-1)*v.detach().clone().view(-1) for t,v in zip(v1,v1)])) # gLv*gLv
        v2Norm = torch.sum(torch.cat([t.detach().clone().view(-1)*v.detach().clone().view(-1) for t,v in zip(v0,v1)])) # gLv*gLt
        mmd = v0Norm + v1Norm -2.0*v2Norm
        bDivs = list([v0Norm, v1Norm, -2.0*v2Norm, mmd])
        dDivs = [e1+e2 for e1,e2 in zip(dDivs, bDivs)]
        mvpNorm = mmd
        #
        v0Norms.update(v0Norm.item())
        v1Norms.update(v1Norm.item())
        mvpNorms.update(mvpNorm.item())
        # v3 = (dL_t / dLambda) * v2: Px1
        hyperOptimizer.zero_grad()
        torch.autograd.backward(g0, grad_tensors=v2)
        hyperOptimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # logger
        if batch_idx % 200 == 0: #args.log_interval == 0:
            print('hyperTrain batch {:.0f}% ({}/{}), task_lr={:.6f}, hyper_lr={:.6f}\t'
                #'gLtNorm {v0Norm.val:.4f} ({v0Norm.avg:.4f})\t'
                #'gLvNorm {v1Norm.val:.4f} ({v1Norm.avg:.4f})\t'
                #'mvpNorm {mvpNorm.val:.4f} ({mvpNorm.avg:.4f})\n'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                100.0*batch_idx/B, batch_idx, B, task_lr, hyper_lr,
                #v0Norm=v0Norms, v1Norm=v1Norms, mvpNorm=mvpNorms,
                batch_time=batch_time, data_time=data_time))
    #
    dDivs = [e/B for e in dDivs]
    #print('Epoch: {}\t Divergence: {:.4f}'.format(epoch, dDivs[-1]))
    return dDivs


def innerTrain(args, encoder, decoder, optimizer, device, loader, epoch, losModel, augModel):
    encoder.train() # train encoder model
    decoder.train() # train decoder model
    losModel.eval() # use fixed hyperModel parameters
    augModel.eval() # use fixed hyperModel parameters
    #
    N = len(loader.dataset) # dataset size
    B = len(loader) # number of batches
    train_loss = 0.0
    #
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    #
    for batch_idx, data in enumerate(loader):
        image = data[0].to(device)
        target= data[1].to(device)
        index = data[2].to(device)
        # measure data loading time
        data_time.update(time.time() - end)
        # warm-up learning rate
        lr = warmup_learning_rate(args, epoch, batch_idx, B, optimizer)
        # model+loss
        aug_image = augModel(index, image)
        encode = encoder(aug_image)
        output = decoder(encode)
        loss = losModel(index, output, target)
        losses.update(loss.item())
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # plot images from first batch for debugging
        if args.plot_debug and (epoch == 0) and (batch_idx < 64):
            fname = 'ori_train_batch_{}_{}.png'.format(batch_idx, args.dataset)
            plot_debug_images(args, rows=2, cols=2, imgs=image, fname=fname)
            fname = 'aug_train_batch_{}_{}.png'.format(batch_idx, args.dataset)
            plot_debug_images(args, rows=2, cols=2, imgs=aug_image, fname=fname)
        # logger
        if batch_idx % args.log_interval == 0:
            print('innerTrain batch {:.0f}% ({}/{}), lr={:.6f}\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                100.0*batch_idx/B, batch_idx, B, lr,
                loss=losses, batch_time=batch_time, data_time=data_time))
    #
    train_loss /= B
    print('Epoch: {}\t Inner Train loss: {:.4f}'.format(epoch, train_loss))
    return train_loss


def classTrain(args, encoder, decoder, optimizer, device, loader, epoch, losModel, augModel):
    encoder.eval() # eval encoder
    decoder.train() # train decoder
    losModel.eval() # use fixed hyperModel parameters
    augModel.eval() # use fixed hyperModel parameters
    #
    N = len(loader.dataset) # dataset size
    B = len(loader) # number of batches
    train_loss = 0.0
    #
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    #
    for batch_idx, data in enumerate(loader):
        image = data[0].to(device)
        target= data[1].to(device)
        index = data[2].to(device)
        # measure data loading time
        data_time.update(time.time() - end)
        # warm-up learning rate
        lr = warmup_learning_rate(args, epoch, batch_idx, B, optimizer)
        # augment+encoder
        with torch.no_grad():
            aug_image = augModel(index, image)
            encode = encoder(aug_image)
        # classifier
        output = decoder(encode.detach())
        loss = losModel(index, output, target)
        losses.update(loss.item())
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # plot images from first batch for debugging
        if args.plot_debug and (epoch == 0) and (batch_idx < 64):
            fname = 'ori_train_batch_{}_{}.png'.format(batch_idx, args.dataset)
            plot_debug_images(args, rows=2, cols=2, imgs=image, fname=fname)
            fname = 'aug_train_batch_{}_{}.png'.format(batch_idx, args.dataset)
            plot_debug_images(args, rows=2, cols=2, imgs=aug_image, fname=fname)
        # logger
        if batch_idx % args.log_interval == 0:
            print('classTrain batch {:.0f}% ({}/{}), lr={:.6f}\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                100.0*batch_idx/B, batch_idx, B, lr,
                loss=losses, batch_time=batch_time, data_time=data_time))
    #
    train_loss /= B
    print('Epoch: {}\t Class Train loss: {:.4f}'.format(epoch, train_loss))
    return train_loss


def innerTest(args, encoder, decoder, device, loader, epoch):
    encoder.eval() # eval encoder
    decoder.eval() # eval decoder
    #
    M = len(loader.dataset) # dataset size
    B = len(loader) # number of batches
    test_loss = 0.0
    correct = 0
    miss_indices = list()
    #
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    #
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            image = data[0].to(device)
            target= data[1].to(device)
            index = data[2].to(device)
            # plot images for debugging
            if args.plot_debug and (epoch == 0) and (batch_idx < 64):
                fname = 'test_batch_{}_{}.png'.format(batch_idx, args.dataset)
                plot_debug_images(args, rows=2, cols=2, imgs=image, fname=fname)
            #
            encode = encoder(image)
            output = decoder(encode)
            loss = metricCE(output[0], target)
            losses.update(loss)
            test_loss += loss
            pred = output[0].max(1, keepdim=True)[1] # get the index of the max probability
            match = pred.eq(target.view_as(pred))
            correct += match.sum().item()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # logger
            if batch_idx % args.log_interval == 0:
                print('innerTest batch {:.0f}% ({}/{})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(100.0*batch_idx/B, batch_idx, B,
                    loss=losses, batch_time=batch_time))
    #
    test_loss /= B
    acc = 100.0 * correct / M
    print('Epoch: {}\t Test loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(epoch, test_loss, correct, M, acc))

    return acc, test_loss, miss_indices


def vizStat(args, encoder, decoder, device, loader, T, F):
    encoder.eval() # eval encoder
    decoder.eval() # eval decoder
    total_loss = 0.0
    correct = 0
    #
    M = len(loader.dataset) # dataset size
    B = len(loader) # number of batches
    #
    fv = torch.zeros(T, F).to(device)
    gt = torch.zeros(T, dtype=torch.long).to(device)
    pr = torch.zeros(T, dtype=torch.long).to(device)
    mi = torch.tensor([], dtype=torch.long).to(device)
    #
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    #
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            image  = data[0].to(device)
            target = data[1].to(device)
            index  = data[2].to(device)
            #
            encode = encoder(image)
            output = decoder(encode)
            loss = metricCE(output[0], target)
            losses.update(loss)
            total_loss += loss
            pred = output[0].max(1, keepdim=True)[1] # get the index of the max probability
            match = pred.eq(target.view_as(pred))
            mask = match.le(0)
            fv[index] = encode
            gt[index] = target.view(-1)
            pr[index] = pred.view(-1)
            mi = torch.cat([mi, torch.masked_select(index.view(-1, 1), mask)])
            correct += match.sum().item()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # logger
            if 1: #batch_idx % args.log_interval == 0:
                print('vizTest batch {:.0f}% ({}/{})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(100.0*batch_idx/B, batch_idx, B,
                    loss=losses, batch_time=batch_time))
    #
    total_loss /= B
    acc = 100.0 * correct / M
    print('Loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(total_loss, correct, M, acc))

    return fv, gt, pr, mi
