import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from typing import Any, Callable, List, Optional, Union, Tuple
import torch.utils.data as data
from PIL import Image, ImageDraw

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import v2

# sys.path.append("/home/j20morli/Documents/Projects/01_Schneider/")
# sys.path.append("/home/j20morli/Documents/Projects/01_Schneider/third_party/RegSeg/")
import transforms as T
import datasets.class_uniform_sampling as uniform


def get_boxes_ids(dicos, image_id) :
    boxes = []
    ids = []
    for dico in dicos : 
        if dico["image_id"] == image_id :
            temp_dico = dico["bbox"]
            xmin = int(temp_dico[0])
            xmax = int(temp_dico[0] + temp_dico[2])
            ymin = int(temp_dico[1])
            ymax = int(temp_dico[1] + temp_dico[3])
            ids.append(str(dico["category_id"]))
            boxes.append([xmin, ymin, xmax, ymax])
    return torch.tensor(boxes), ids

def draw_bounding_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    colors = None,
    fill = False,
    width: int = 1,
    rotations = torch.Tensor
) -> torch.Tensor:

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size(0) not in {1, 3}:
        raise ValueError("Only grayscale and RGB images are supported")
    elif boxes.dim() != 2 :
        return image
    elif (boxes[:, 0] > boxes[:, 2]).any() or (boxes[:, 1] > boxes[:, 3]).any():
        raise ValueError(
            "Boxes need to be in (xmin, ymin, xmax, ymax) format. Use torchvision.ops.box_convert to convert them"
        )
    

    num_boxes = boxes.shape[0]

    if num_boxes == 0:
        print("no boxes")
        return image

    # Handle Grayscale images
    if image.size(0) == 1:
        image = torch.tile(image, (3, 1, 1))

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    img_boxes = boxes.to(torch.int64).tolist()

    if fill:
        draw = ImageDraw.Draw(img_to_draw, "RGBA")
    else:
        draw = ImageDraw.Draw(img_to_draw)

    for bbox, color in zip(img_boxes, colors): 
    #for bbox, color, rotation in zip(img_boxes, colors, rotations):  # type: ignore[arg-type]

        draw.rectangle(bbox, width=width, fill=color)

    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)


def find_hit_uav_filenames(images_dir,targets_dir,split,target_suffix):
    all_imgs=[]
    images_dir=os.path.join(images_dir,split)
    targets_dir=os.path.join(targets_dir,split)

    for file_name in sorted(os.listdir(images_dir)) :
        target_types = []
        target_name = file_name
        target_types.append(os.path.join(targets_dir, target_name))
        image_filename=os.path.join(images_dir, file_name)
        target_filename=target_types[0]
        print(image_filename, target_filename)
        all_imgs.append((image_filename,target_filename))
    return all_imgs

HIT_UAVClass = namedtuple('HIT_UAVClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
class HIT_UAV(data.Dataset):


# [{'supercategory': 'Person', 'id': 0, 'name': 'Person'}, {'supercategory': 'Vehicle', 'id': 1, 'name': 'Car'}, 
#    {'supercategory': 'Vehicle', 'id': 2, 'name': 'Bicycle'}, {'supercategory': 'Vehicle', 'id': 3, 'name': 'OtherVehicle'},
#     {'supercategory': 'DontCare', 'id': 4, 'name': 'DontCare'}]

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
        self.mode = 'normal'
        self.images_dir = self.root
        self.targets_dir = os.path.join(self.root, "semantic")
        self.annotations_dir = os.path.join(self.root, "annotations")
        target_suffix=self._get_target_suffix(self.mode, "semantic")
        self.split = split
        self.class_uniform_pct=class_uniform_pct
        self.all_imgs=[]
        self.imgs=None
        self.num_classes=19
        self.centroids = []

        self.classes = [
        HIT_UAVClass('Person', 0, 255, 'Person', 0, False, True, (255, 0, 0)),
        HIT_UAVClass('ego Car', 1, 255, 'Vehicle', 1, False, True, (0, 255, 10)),
        HIT_UAVClass('Bicycle', 2, 255, 'Vehicle', 1, False, True, (0, 255, 70)),
        HIT_UAVClass('OtherVehicle', 3, 255, 'Vehicle', 1, False, True, (0, 255, 150)),
        HIT_UAVClass('DontCare', 4, 255, 'DontCare', 2, False, True, (255, 255, 255)),
        HIT_UAVClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    ]

        if not os.path.isdir(self.targets_dir) :
            self.generate_semantic()
        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete')
        if "train" in self.split:
            self.all_imgs.extend(find_hit_uav_filenames(self.images_dir,self.targets_dir,"train",target_suffix))
        if "val" in self.split:
            self.all_imgs.extend(find_hit_uav_filenames(self.images_dir,self.targets_dir,"val",target_suffix))
        if "test" in self.split:
            self.all_imgs.extend(find_hit_uav_filenames(self.images_dir,self.targets_dir,"test",target_suffix))

        # if self.class_uniform_pct>0 and self.split in ["train","trainval"]:
        #     json_fn=self.root+"_train_centroids.json"
        #     if self.split=="trainval":
        #         json_fn=self.root+"_trainval_centroids.json"
        #     self.centroids = uniform.build_centroids(
        #         self.all_imgs,
        #         self.num_classes,
        #         id2trainid=False,
        #         json_fn=json_fn
        #     )
        # else:
        #     self.centroids=[]
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
        target=Image.open(target_filename).convert('RGB')
        print(image.size)
        print(target.size)
        print(centroid)
        if self.transforms is not None:
            image, target = self.transforms(image, (target,centroid))

        return image, target

    def __len__(self) -> int:
        return len(self.imgs)

    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return ".png"
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            return '{}_polygons.json'.format(mode)
    

    def get_colors(self, ids) : 
        return [self.classes[int(i)][-1] for i in ids]
        

    def generate_semantic(self) :
        # get oriented bounding boxes and generate semantic segmentation images
        if not os.path.isdir(self.targets_dir) :
            os.mkdir(self.targets_dir)

        splits = ["/train", "/val", "/test"]
        for split in splits :
            path = self.targets_dir + split

            if not os.path.isdir(path) :
                os.mkdir(path)
            
            json_name = split + ".json"
            with open(self.annotations_dir + json_name, "r+") as file :
                contents = file.read()

            json_content = json.loads(contents)
            json_annotations = json_content["annotation"]
            image_id = json_annotations[0]["image_id"]
            new_image = True
            min_id = json_content["images"][0]["id"]
            max_id = json_content["images"][-1]["id"]
            print(len(json_content["images"]), max_id)

            for image_id in range(0, max_id + 1 - min_id) :
                t1 = time.time()
                #print("Time 0: ", time.strftime("%H:%M:%S", time.gmtime()))
                image = json_content["images"][image_id]
                image_size =  (1, image["height"], image["width"])
                image_filename = image["filename"]
                print(image_size)
                mask = torch.zeros(image_size, dtype=torch.uint8)
                fig = plt.figure(frameon=False)
                fig.set_size_inches(float(image["width"]/100),float(image["height"]/100))
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                boxes, ids = get_boxes_ids(json_annotations, image_id + min_id)
                #print(boxes.size()) 
                t2 = time.time()
                #print("Time 1: ", time.strftime("%H:%M:%S", time.gmtime()))
                colors = self.get_colors(ids)
                mask = draw_bounding_boxes(mask, boxes, fill=True, colors=colors)
                mask = F.to_pil_image(mask)
                ax.imshow(mask, aspect='auto')
               
                fig.savefig(self.targets_dir + split + "/" + image_filename)
                #fig.close()
                #plt.show()
                t3 = time.time()
                print("Time : ", time.strftime("%H:%M:%S", time.gmtime()), " T1 :", t2 - t1, "T2 :", t3 - t2, "image_id : ", image_id)

if __name__=='__main__':
    #import third_party.RegSeg.transforms as T
    ts=T.Compose([T.RandomResize(1024,1024,"uniform"),T.ToTensor()])
    #transforms = v2.Compose([v2.RandomResizedCrop(size=(480, 480)), v2.ToTensor()])
    dataset=HIT_UAV("/home/j20morli/Documents/Projects/01_Schneider/data/HIT-UAV/HIT-UAV/normal_json/", transforms=ts)
    image,target=dataset[1]
    print(image.shape)
    print(target.shape)
