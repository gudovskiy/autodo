from __future__ import division
import numpy as np
import random, math
import torch
from torchvision import transforms
from PIL import Image
from custom_transforms.rand_augment import Lighting
import matplotlib.pyplot as plt

_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}

__all__ = ["transform_test_mnist", "transform_train_mnist",
           "transform_test_e2mnist", "transform_train_e2mnist",
           "transform_test_svhn", "transform_train_svhn",
           "transform_test_cifar10", "transform_train_cifar10",
           "transform_test_cifar100", "transform_train_cifar100",
           "transform_test_imagenet", "transform_train_imagenet",
           "plot_debug_images", "TwoCropTransform"]

################## From Fast AA code ##################

class EfficientNetRandomCrop:
    def __init__(self, imgsize, min_covered=0.1, aspect_ratio_range=(3./4, 4./3), area_range=(0.08, 1.0), max_attempts=10):
        assert 0.0 < min_covered
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]
        assert 0 < area_range[0] <= area_range[1]
        assert 1 <= max_attempts

        self.min_covered = min_covered
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range
        self.max_attempts = max_attempts
        self._fallback = EfficientNetCenterCrop(imgsize)

    def __call__(self, img):
        # https://github.com/tensorflow/tensorflow/blob/9274bcebb31322370139467039034f8ff852b004/tensorflow/core/kernels/sample_distorted_bounding_box_op.cc#L111
        original_width, original_height = img.size
        min_area = self.area_range[0] * (original_width * original_height)
        max_area = self.area_range[1] * (original_width * original_height)

        for _ in range(self.max_attempts):
            aspect_ratio = random.uniform(*self.aspect_ratio_range)
            height = int(round(math.sqrt(min_area / aspect_ratio)))
            max_height = int(round(math.sqrt(max_area / aspect_ratio)))

            if max_height * aspect_ratio > original_width:
                max_height = (original_width + 0.5 - 1e-7) / aspect_ratio
                max_height = int(max_height)
                if max_height * aspect_ratio > original_width:
                    max_height -= 1

            if max_height > original_height:
                max_height = original_height

            if height >= max_height:
                height = max_height

            height = int(round(random.uniform(height, max_height)))
            width = int(round(height * aspect_ratio))
            area = width * height

            if area < min_area or area > max_area:
                continue
            if width > original_width or height > original_height:
                continue
            if area < self.min_covered * (original_width * original_height):
                continue
            if width == original_width and height == original_height:
                return self._fallback(img) # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/preprocessing.py#L102

            x = random.randint(0, original_width - width)
            y = random.randint(0, original_height - height)
            return img.crop((x, y, x + width, y + height))

        return self._fallback(img)


class EfficientNetCenterCrop:
    def __init__(self, imgsize):
        self.imgsize = imgsize

    def __call__(self, img):
        """Crop the given PIL Image and resize it to desired size.

        Args:
            img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
            output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions
        Returns:
            PIL Image: Cropped image.
        """
        image_width, image_height = img.size
        image_short = min(image_width, image_height)

        crop_size = float(self.imgsize) / (self.imgsize + 32) * image_short

        crop_height, crop_width = crop_size, crop_size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))


_MNIST_MEAN, _MNIST_STD = (0.1307,), (0.3081,)
_SVHN_MEAN, _SVHN_STD = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
_CIFAR10_MEAN, _CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
_CIFAR100_MEAN, _CIFAR100_STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
_IMAGENET_MEAN, _IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def plot_debug_images(args, imgs, rows, cols, fname, path='./debug'):
    ims = imgs.clone().detach().cpu().numpy()
    fig = plt.figure(figsize=(10, 10))
    bs, c, h, w = ims.shape # batch size, channel, height, width
    ims = ims.transpose(0,2,3,1) # B,H,W,C
    # denormalization
    if args.dataset == 'MNIST':
        ims = ims*np.asarray(_MNIST_STD) + np.asarray(_MNIST_MEAN)
    elif args.dataset == 'CIFAR10':
        ims = ims*np.asarray(_CIFAR10_STD) + np.asarray(_CIFAR10_MEAN)
    elif args.dataset == 'CIFAR100':
        ims = ims*np.asarray(_CIFAR100_STD) + np.asarray(_CIFAR100_MEAN)
    elif args.dataset == 'SVHN' or args.dataset == 'SVHN_extra':
        ims = ims*np.asarray(_SVHN_STD) + np.asarray(_SVHN_MEAN)
    elif args.dataset == 'ImageNet':
        ims = ims*np.asarray(_IMAGENET_STD) + np.asarray(_IMAGENET_MEAN)
    # clipping
    #ims = np.clip(ims, 0.0, 1.0)
    ims = 1.0 - np.clip(ims, 0.0, 1.0)
    # greyscale
    if c == 1:
        ims = np.squeeze(ims, axis=3)
    bs = rows*cols #min(bs, 16) # limit plot to 16 images
    #ns = np.ceil(bs ** 0.5) # number of subplots
    #
    for i in range(bs):
        #print(np.amin(ims[i]), np.amax(ims[i]))
        plt.subplot(rows, cols, i+1).imshow(ims[i], cmap='gray')
        plt.axis('off')
    fig.tight_layout()
    fig.savefig('{}/{}'.format(path, fname), dpi=300)
    plt.close()

################## SLC ##################
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


################## MNIST ##################
transform_train_mnist = transforms.Compose([
    transforms.ToTensor(),
    #transforms.RandomErasing(p=0.5, scale=(0.02, 0.06), ratio=(0.3, 3.3)),
    transforms.Normalize(_MNIST_MEAN, _MNIST_STD),
])
transform_test_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_MNIST_MEAN, _MNIST_STD),
])
################## E2MNIST ##################
transform_train_e2mnist = transforms.Compose([
    transforms.Pad((0, 0, 1, 1), fill=0),
    transforms.ToTensor(),
    #transforms.RandomErasing(p=0.5, scale=(0.02, 0.06), ratio=(0.3, 3.3)),
    transforms.Normalize(_MNIST_MEAN, _MNIST_STD),
])
transform_test_e2mnist = transforms.Compose([
    transforms.Pad((0, 0, 1, 1), fill=0),
    transforms.ToTensor(),
    transforms.Normalize(_MNIST_MEAN, _MNIST_STD),
])
################## SVHN ##################
transform_train_svhn = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(_SVHN_MEAN, _SVHN_STD),
    transforms.RandomErasing(),
])
transform_test_svhn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_SVHN_MEAN, _SVHN_STD),
])
################## CIFAR10 ##################
transform_train_cifar10 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    transforms.RandomErasing(),
])
transform_test_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
])
################## CIFAR100 ##################
transform_train_cifar100 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
    transforms.RandomErasing(),
])
transform_test_cifar100 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
])
################## ImageNet ##################
transform_train_imagenet = transforms.Compose([
    #EfficientNetRandomCrop(224),
    #transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    #Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    #transforms.RandomErasing(),
])
transform_test_imagenet = transforms.Compose([
    #EfficientNetCenterCrop(224),
    #transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])
