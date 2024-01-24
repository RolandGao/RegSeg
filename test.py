import datetime
import time
import torch
import yaml
import torch.cuda.amp as amp
import torchvision
import os
import copy
import random
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
import numpy as np
from train_utils import get_lr_function, get_loss_fun,get_optimizer,get_dataset_loaders,get_model,get_val_dataset
from benchmark import compute_time_full, compute_time_no_loader,compute_loader_time
from precise_bn import compute_precise_bn_stats
from model import RegSeg
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

path = "/home/j20morli/Documents/Projects/01_Schneider/data/models/model2"

dictio = torch.load(path, map_location=torch.device('cpu'))
model_dic = dictio["model"]

model = RegSeg(name="exp48_decoder26", num_classes=6)
model.load_state_dict(model_dic)

red = (255, 0, 0)
blue = (0, 0, 255)
green = (0, 255, 0)
violet = (255, 0, 255)
yellow = (255, 255, 0)
other = (122, 122, 255)
colors = [red, blue, green, violet, yellow, other]

test_name = "1_110_30_0_02555.jpg"
image = torchvision.io.read_image("/home/j20morli/Documents/Projects/01_Schneider/data/HIT-UAV/HIT-UAV/normal_json/test/" + test_name)
save_path = "/home/j20morli/Documents/Projects/01_Schneider/data/examples/"
number = "4"

print(image.size())
image = image.expand(3, -1, -1)
small = image[:, 0:80, 210:290]
plt.imshow(small.permute(1, 2, 0))
plt.savefig(save_path + "image_" + number + "_10")
plt.close()
small = image[:, 0:32, 240:272]
plt.imshow(small.permute(1, 2, 0))
plt.savefig(save_path + "image_" + number + "_11")
plt.close()
image2 = image.unsqueeze(0)
image2 = image2.float()
print(image.size(), image.dtype)
output = model(image2)
print(output.size())
output_image = torch.zeros_like(image)
plt.imshow(image.permute(1, 2, 0))
plt.savefig(save_path + "image_" + number + "_0")
plt.close()

image3 = torchvision.utils.draw_segmentation_masks(image, output.squeeze(0) > 0.5)
image3 = image3.permute(1, 2, 0)
plt.imshow(image3)
plt.savefig(save_path + "image_" + number + "_1")
plt.close()


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


# show original bounding boxes
json_path = "/home/j20morli/Documents/Projects/01_Schneider/data/HIT-UAV/HIT-UAV/normal_json/annotations/test.json"
with open(json_path, "r+") as file :
    contents = file.read()

json_content = json.loads(contents)
json_annotations = json_content["annotation"]
image_id = json_annotations[0]["image_id"]
min_id = json_content["images"][0]["id"]
max_id = json_content["images"][-1]["id"]
print(len(json_content["images"]), max_id)

for image_id in range(0, max_id + 1 - min_id) :
    t1 = time.time()
    #print("Time 0: ", time.strftime("%H:%M:%S", time.gmtime()))
    imagee = json_content["images"][image_id]
    image_size =  (1, imagee["height"], imagee["width"])
    image_filename = imagee["filename"]

    if image_filename == test_name :
        boxes, ids = get_boxes_ids(json_annotations, image_id + min_id)
        colorsss = [colors[int(i)] for i in ids]
        img = torchvision.utils.draw_bounding_boxes(image, boxes, fill=True, colors=colorsss)
        img = F.to_pil_image(img)
        plt.imshow(img)
        plt.savefig(save_path + "image_" + number + "_5")
        plt.close()
            
    # mask = torch.zeros(image_size, dtype=torch.uint8)
    # fig = plt.figure(frameon=False)
    # fig.set_size_inches(float(image["width"]/100),float(image["height"]/100))
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # boxes, ids = get_boxes_ids(json_annotations, image_id + min_id)
    # #print(boxes.size()) 
    # t2 = time.time()
    # #print("Time 1: ", time.strftime("%H:%M:%S", time.gmtime()))
    # colors = self.get_colors(ids)
    # mask = draw_bounding_boxes(mask, boxes, fill=True, colors=colors)
    # mask = F.to_pil_image(mask)
    # ax.imshow(mask, aspect='auto')
    
    # fig.savefig(self.targets_dir + split + "/" + image_filename)
    # plt.close(fig)
    # #plt.show()
    # t3 = time.time()
    # print(split[1:], "  Time : ", time.strftime("%H:%M:%S", time.gmtime()), " T1 :", t2 - t1, "T2 :", t3 - t2, "image_id : ", image_id)