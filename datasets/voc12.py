import os
import tarfile
import torch.utils.data as data

from PIL import Image
from torchvision.datasets.utils import download_url

class Voc12Segmentation(data.Dataset):
    def __init__(self,root,image_set,transforms,download=False):
        self.root = os.path.expanduser(root)
        self.url='http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
        self.filename='VOCtrainval_11-May-2012.tar'
        self.md5='6cd6e144f989b92b3379bac3b3de84fd'
        self.base_dir='VOCdevkit/VOC2012'
        self.transforms=transforms
        voc_root = os.path.join(self.root, self.base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        if download:
            download_extract(self.url, self.root, self.filename, self.md5)
        if not os.path.isdir(voc_root):
            raise RuntimeError(f'{voc_root} not found')
        if image_set == 'train_aug':
            mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
            split_f = os.path.join(voc_root, f'ImageSets/Segmentation/{image_set}.txt')
        else:
            mask_dir = os.path.join(voc_root, 'SegmentationClass')
            split_f = os.path.join(voc_root, f'ImageSets/Segmentation/{image_set}.txt')
        if not os.path.exists(split_f):
            raise RuntimeError(f'{split_f} not found')
        with open(split_f, "r") as f:# os.path.join(split_f)
            file_names = [x.strip() for x in f.readlines()]
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=root)
