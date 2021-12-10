from PIL import Image
import torch.utils.data as data
import os
import json
import numpy as np
from pycocotools import mask as coco_mask

def load_coco_json(json_file, image_dir):
    from pycocotools.coco import COCO
    coco_api = COCO(json_file)
    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    records = []

    for img_dict, anno_dict_list in zip(imgs, anns):
        record = {}
        record["file_name"] = os.path.join(image_dir, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["image_id"] = img_dict["id"]

        annotations = []
        for anno in anno_dict_list:
            obj={}
            obj["segmentation"] = anno["segmentation"]
            obj["category_id"]=anno["category_id"]
            obj["area"]=anno["area"]
            annotations.append(obj)
        record["annotations"] = annotations
        records.append(record)
    return records
def filter_and_remap_records(records, categories,filter_records=True):
    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if more than 1k pixels occupied in the image
        return sum(obj["area"] for obj in anno) > 1000
    new_records=[]
    for record in records:
        annotations=record["annotations"]
        annotations=[obj for obj in annotations if obj["category_id"] in categories]
        for obj in annotations:
            obj["category_id"]=categories.index(obj["category_id"])
        record["annotations"]=annotations
        if filter_records:
            if _has_valid_annotation(annotations):
                new_records.append(record)
        else:
            new_records.append(record)
    return new_records

def convert_polygons_to_mask(polygons, height, width):
    rles = coco_mask.frPyObjects(polygons, height, width)
    mask = coco_mask.decode(rles)
    if len(mask.shape) < 3:
        mask = mask[..., None]
    mask = mask.any(axis=2)
    return mask
def convert_annotations_to_mask(annotations, h, w):
    target = np.zeros((h, w), dtype=np.uint8)
    global_mask=np.zeros((h, w), dtype=np.uint8)
    for obj in annotations:
        polygons=obj["segmentation"]
        cat=obj["category_id"]
        mask=convert_polygons_to_mask(polygons, h, w)
        target[mask]=cat
        global_mask=global_mask+mask
    target[global_mask>1]=255
    target = Image.fromarray(target)
    return target
class Coco(data.Dataset):
    def __init__(self, root, image_set, transforms, categories=None):
        # images, masks, json splits
        if categories is None:
            categories=[0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        root= os.path.expanduser(root)
        img_dir=os.path.join(root,"images",f"{image_set}2017")
        json_file=os.path.join(root,"annotations", f"instances_{image_set}2017.json")
        filter_records=(image_set=="train")
        records=load_coco_json(json_file,img_dir)
        records=filter_and_remap_records(records, categories,filter_records)
        self.records=records
        self.transforms=transforms

    def __getitem__(self, index):
        record=self.records[index]
        h,w=record["height"],record["width"]
        image_filename=record["file_name"]
        annotations=record["annotations"]
        image = Image.open(image_filename).convert('RGB')
        target=convert_annotations_to_mask(annotations, h, w)
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.records)
