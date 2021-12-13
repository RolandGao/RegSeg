import os
from collections import namedtuple
from typing import Any, Callable, List, Optional, Union, Tuple
import torch.utils.data as data
from PIL import Image
import datasets.class_uniform_sampling as uniform

def find_cityscapes_filenames(images_dir,targets_dir,split,target_suffix):
    all_imgs=[]
    images_dir=os.path.join(images_dir,split)
    targets_dir=os.path.join(targets_dir,split)
    for city in sorted(os.listdir(images_dir)):
        if city[0]==".":
            continue
        img_dir = os.path.join(images_dir,city)
        target_dir = os.path.join(targets_dir, city)
        for file_name in sorted(os.listdir(img_dir)):
            target_types = []
            target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],target_suffix)
            target_types.append(os.path.join(target_dir, target_name))
            image_filename=os.path.join(img_dir, file_name)
            target_filename=target_types[0]
            all_imgs.append((image_filename,target_filename))
    return all_imgs

class Cityscapes(data.Dataset):
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    def __init__(
            self,
            root: str,
            split: str = "train",
            mode: str = "fine",
            target_type: Union[List[str], str] = "semantic",
            transforms: Optional[Callable] = None,
            class_uniform_pct=0.5
    ) -> None:
        assert target_type=="semantic","currently we only support semantic labels"
        self.root=root
        self.transforms=transforms
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit')
        self.targets_dir = os.path.join(self.root, self.mode)
        target_suffix=self._get_target_suffix(self.mode, "semantic")
        self.split = split
        self.class_uniform_pct=class_uniform_pct
        self.all_imgs=[]
        self.imgs=None
        self.num_classes=19

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete')
        if "train" in self.split:
            self.all_imgs.extend(find_cityscapes_filenames(self.images_dir,self.targets_dir,"train",target_suffix))
        if "val" in self.split:
            self.all_imgs.extend(find_cityscapes_filenames(self.images_dir,self.targets_dir,"val",target_suffix))
        if "test" in self.split:
            self.all_imgs.extend(find_cityscapes_filenames(self.images_dir,self.targets_dir,"test",target_suffix))
        if self.class_uniform_pct>0 and self.split in ["train","trainval"]:
            json_fn=self.root+"_train_centroids.json"
            if self.split=="trainval":
                json_fn=self.root+"_trainval_centroids.json"
            self.centroids = uniform.build_centroids(
                self.all_imgs,
                self.num_classes,
                id2trainid=False,
                json_fn=json_fn
            )
        else:
            self.centroids=[]
        self.build_epoch()

    def build_epoch(self):
        """
        For class uniform sampling ... every epoch, we want to recompute
        which tiles from which images we want to sample from, so that the
        sampling is uniformly random.
        """
        if len(self.centroids)!= 0:
            self.imgs = uniform.build_epoch(
                self.all_imgs,
                self.centroids,
                self.num_classes,
                self.class_uniform_pct
            )
        else:
            self.imgs=self.all_imgs


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        if len(self.imgs[index])==2:
            image_filename,target_filename=self.imgs[index]
            centroid=None
            class_id=None
        else:
            image_filename,target_filename, centroid, class_id = self.imgs[index]
        image = Image.open(image_filename).convert('RGB')
        target=Image.open(target_filename)

        if self.transforms is not None:
            image, target = self.transforms(image, (target,centroid))

        return image, target

    def __len__(self) -> int:
        return len(self.imgs)

    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelTrainIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            return '{}_polygons.json'.format(mode)

if __name__=='__main__':
    import transforms as T
    ts=T.Compose([T.RandomResize(1024,1024,"uniform"),T.ToTensor()])
    dataset=Cityscapes("../cityscapes_dataset", transforms=ts)
    image,target=dataset[1]
    print(image.shape)
    print(target.shape)
