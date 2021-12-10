import random
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import torch
from augment import rand_augment_both
import math

class Compose(object):
    """
    Composes a sequence of transforms.
    Arguments:
        transforms: A list of transforms.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

class RandAugment:
    def __init__(self,N,M,ops,prob,fill,ignore_value):
        #prob=1.0,fill=(128,128,128),ignore_value=255
        self.N=N
        self.M=M
        self.prob=prob
        self.fill=fill
        self.ignore_value=ignore_value
        self.ops=ops
    def __call__(self, image, target):
        return rand_augment_both(image,target,ops=self.ops,n_ops=self.N,magnitude=self.M,prob=self.prob,fill=self.fill,ignore_value=self.ignore_value)


class Normalize(object):
    """
    Normalizes image by mean and std.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, label

class ValResize(object):
    def __init__(self,val_input_size,val_label_size):
        self.input_size=val_input_size
        self.label_size=val_label_size
    def __call__(self, image, target):
        if isinstance(target, tuple) or isinstance(target,list):
            target,_=target # we don't need the centroid in the second position
        image = F.resize(image, self.input_size)
        target = F.resize(target, self.label_size, interpolation=F.InterpolationMode.NEAREST)
        return image,target

class RandomResize(object):
    def __init__(self, min_size, max_size, sampling_mode):
        self.min_size = min_size
        self.max_size = max_size
        if sampling_mode not in ["uniform","log_uniform"]:
            raise NotImplementedError()
        self.sampling_mode=sampling_mode

    def __call__(self, image, target):
        if self.sampling_mode=="uniform":
            size = random.randint(self.min_size, self.max_size)
        elif self.sampling_mode=="log_uniform":
            size=int(2**random.uniform(math.log2(self.min_size),math.log2(self.max_size)))
        else:
            raise NotImplementedError()
        if isinstance(target, tuple) or isinstance(target,list):
            target,centroid=target
        else:
            centroid=None
        if centroid is not None:
            w,h=target.size
            scale=size/min(w,h)
            centroid = [int(c * scale) for c in centroid]
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=F.InterpolationMode.NEAREST)
        if centroid is not None:
            target=(target,centroid)
        return image, target

class ColorJitter:
    def __init__(self,brightness=0.2, contrast=0.2, saturation=(0.5,4), hue=0.2, prob=0.5):
        self.jitter=T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.prob=prob
    def __call__(self, image, target):
        if random.random()<self.prob:
            image=self.jitter(image)
        return image,target

class AddNoise:
    # gaussian
    # factor = random.uniform(0, self.factor)
    def __init__(self,factor, prob=0.5):
        self.factor=factor
        self.prob=prob
    def __call__(self, image, target):
        if random.random()<self.prob:
            factor = random.uniform(0, self.factor)
            image = np.array(image)
            assert(image.dtype==np.uint8)
            gauss=np.array(torch.normal(0,factor,image.shape))
            #gauss = np.array(torch.randn(*image.shape)) * factor
            noisy = (image + gauss).clip(0, 255).astype("uint8")
            image = Image.fromarray(noisy)
        return image, target
class AddNoise2:
    # gaussian
    # factor = self.factor
    def __init__(self,factor, prob=0.5):
        self.factor=factor
        self.prob=prob
    def __call__(self, image, target):
        if random.random()<self.prob:
            factor = self.factor
            image = np.array(image)
            assert(image.dtype==np.uint8)
            gauss=np.array(torch.normal(0,factor,image.shape))
            noisy = (image + gauss).clip(0, 255).astype("uint8")
            image = Image.fromarray(noisy)
        return image, target
class AddNoise3:
    # shot
    def __init__(self,factor, prob=0.5):
        self.factor=factor
        self.prob=prob
    def __call__(self, image, target):
        if random.random()<self.prob:
            factor = self.factor
            image = np.array(image)
            assert(image.dtype==np.uint8)
            shot=np.random.randint(0,2,size=image.shape)
            shot=(shot*2-1)*factor
            noisy = (image + shot).clip(0, 255).astype("uint8")
            image = Image.fromarray(noisy)
        return image, target

class RandomRotation:
    def __init__(self,degrees,mean,ignore_value, prob,expand=False):
        self.degrees=degrees
        self.mean=mean
        self.ignore_value=ignore_value
        self.prob=prob
        self.expand=expand
    def __call__(self, image, target):
        if random.random()<self.prob:
            angle = random.uniform(*self.degrees)
            image=F.rotate(image, angle,fill=self.mean,expand=self.expand)
            target=F.rotate(target,angle,fill=self.ignore_value,expand=self.expand)
        return image,target

def get_edge_aware_crop_param(img,output_size):
    w, h = F._get_image_size(img)
    th, tw = output_size

    if h + 1 < th or w + 1 < tw:
        raise ValueError(
            "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
        )

    if w == tw and h == th:
        return 0, 0, h, w

    i = torch.randint(-th+1, h, size=(1, ))
    i=torch.clip(i,0,h-th).item()
    j = torch.randint(-tw+1, w, size=(1, ))
    j=torch.clip(j,0,w-tw).item()
    return i, j, th, tw

def get_centroid_crop_params(img,output_size,centroid):
    c_x, c_y = centroid
    w, h = F._get_image_size(img)
    th, tw = output_size
    max_x = w - tw
    max_y = h - th
    x1 = random.randint(c_x - tw, c_x)
    x1 = min(max_x, max(0, x1))
    y1 = random.randint(c_y - th, c_y)
    y1 = min(max_y, max(0, y1))
    return y1,x1,th, tw
class RandomPad(object):
    def __init__(self,crop_h, crop_w, pad_value, ignore_label, random_pad):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.pad_value = pad_value
        self.ignore_label = ignore_label
        self.random_pad = random_pad
    def __call__(self, image, label):
        img_w,img_h=image.size
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            if self.random_pad:
                pad_top = random.randint(0, pad_h)
                pad_bottom = pad_h - pad_top
                pad_left = random.randint(0, pad_w)
                pad_right = pad_w - pad_left
            else:
                pad_top, pad_bottom, pad_left, pad_right = 0, pad_h, 0, pad_w
            image = F.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=self.pad_value)
            label= F.pad(label, (pad_left, pad_top, pad_right, pad_bottom), fill=self.ignore_label)
        return image,label
class RandomCrop(object):
    def __init__(self, crop_h, crop_w, pad_value, ignore_label, random_pad, edge_aware):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.pad_value = pad_value
        self.ignore_label = ignore_label
        self.random_pad = random_pad
        self.edge_aware=edge_aware

    def __call__(self, image, label):
        if isinstance(label, tuple) or isinstance(label,list):
            label,centroid=label
        else:
            centroid=None
        img_w,img_h=image.size
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            if self.random_pad:
                pad_top = random.randint(0, pad_h)
                pad_bottom = pad_h - pad_top
                pad_left = random.randint(0, pad_w)
                pad_right = pad_w - pad_left
            else:
                pad_top, pad_bottom, pad_left, pad_right = 0, pad_h, 0, pad_w
            image = F.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=self.pad_value)
            label= F.pad(label, (pad_left, pad_top, pad_right, pad_bottom), fill=self.ignore_label)
        if centroid is not None:
            crop_params=get_centroid_crop_params(image,(self.crop_h, self.crop_w),centroid)
        elif self.edge_aware:
            crop_params=get_edge_aware_crop_param(image,(self.crop_h, self.crop_w))
        else:
            crop_params = T.RandomCrop.get_params(image, (self.crop_h, self.crop_w))
        image = F.crop(image, *crop_params)
        label = F.crop(label, *crop_params)
        return image,label
class RandomCrop2(object):
    def __init__(self, crop_h, crop_w, edge_aware):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.edge_aware=edge_aware

    def __call__(self, image, label):
        if isinstance(label, tuple) or isinstance(label,list):
            label,centroid=label
        else:
            centroid=None
        img_w,img_h=image.size
        crop_h=min(self.crop_h,img_h)
        crop_w=min(self.crop_w,img_w)
        if centroid is not None:
            crop_params=get_centroid_crop_params(image,(crop_h, crop_w),centroid)
        elif self.edge_aware:
            crop_params=get_edge_aware_crop_param(image,(crop_h, crop_w))
        else:
            crop_params = T.RandomCrop.get_params(image, (crop_h, crop_w))
        image = F.crop(image, *crop_params)
        label = F.crop(label, *crop_params)
        return image,label

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target
