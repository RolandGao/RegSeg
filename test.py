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
from model import RegSeg, Enet_Regseg
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchinfo import summary

path = "/users/local/j20morli/Schneider2/third_party/RegSeg/checkpoints/0.01_hit_uav_enet_500epochs_latest"
test_name = "1_130_60_0_07248.jpg"
test_name = "0_80_30_0_06926.jpg"
test_name = "1_60_80_0_00669.jpg"
save_path = "/users/local/j20morli/Schneider2/examples/"
image_path = "/users/local/j20morli/Schneider2/third_party//HIT-UAV/normal_json/train/" + test_name
json_path = "/users/local/j20morli/Schneider2/third_party//HIT-UAV/normal_json/annotations/train.json"

number = "12"
dictio = torch.load(path, map_location=torch.device('cpu'))
model_dic = dictio["model"]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#model = RegSeg(name="exp48_decoder26", num_classes=2)
model = Enet_Regseg(name="lowres", num_classes=2)
model.load_state_dict(model_dic)
model.to(device)

model_test = Enet_Regseg(name="highres", num_classes=2)
summary(model_test, input_size=(1, 3, 80, 80), depth=2)

red = (255, 0, 0)
blue = (0, 0, 255)
green = (0, 255, 0)
violet = (255, 0, 255)
yellow = (255, 255, 0)
other = (122, 122, 255)
colors = [red, blue, green, violet, yellow, other]

# utilities
def print_and_save(image, save_name) :
	plt.imshow(image)
	plt.savefig(save_name)
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

def add_bounding_boxes(image, json_content, save_name) :
	json_annotations = json_content["annotations"]
	min_id = json_content["images"][0]["id"]
	max_id = json_content["images"][-1]["id"]
	
	for image_id in range(0, max_id + 1 - min_id) :
		json_image = json_content["images"][image_id]
		image_size =  (1, json_image["height"], json_image["width"])
		image_filename = json_image["filename"]

		if image_filename == test_name :
			boxes, ids = get_boxes_ids(json_annotations, image_id + min_id)
			colorsss = [colors[int(i)] for i in ids]
			print(ids)
			img = torchvision.utils.draw_bounding_boxes(image, boxes, fill=True, colors=colorsss)
			img = F.to_pil_image(img)
			print_and_save(img, save_name)
			
def pipeline(image, model, threshold, name) :
	# show image before transformation
	print_and_save(image.permute(1, 2, 0), save_path + "image_" + name + "_original" + ".png")
	
	output = model(image.unsqueeze(0).float().to(device)).cpu()
	draw_mask = torchvision.utils.draw_segmentation_masks(image, output[0] > threshold)
	print_and_save(draw_mask.permute(1, 2, 0), save_path + "image_" + name + "_mask" + ".png")

	argmax = torch.argmax(output, dim=1)
	print_and_save(argmax.permute(1, 2, 0), save_path + "image_" + name + "_argmax" + ".png")
	
	return output, draw_mask, argmax
	
	
	
# original image
image = torchvision.io.read_image(image_path)
image = image.expand(3, -1, -1) # to RGB standard
print_and_save(image.permute(1, 2, 0), save_path + "image_" + number + "_original" + ".png")

# smaller images
small80 = image[:, 200:280, 300:380]
small32 = image[:, 50:82, 210:242]
print_and_save(small80.permute(1, 2, 0), save_path + "image_" + number + "_80" + ".png")
print_and_save(small32.permute(1, 2, 0), save_path + "image_" + number + "_32" + ".png")

threshold = 0
voutput_original, draw_mask_original, argmax_original = pipeline(image, model, threshold, "original_" + number)
pipeline(small80, model, threshold, "80_" + number)
pipeline(small32, model, threshold, "32_"+ number)

# show original bounding boxes
with open(json_path, "r+") as file :
    contents = file.read()

json_content = json.loads(contents)
add_bounding_boxes(draw_mask_original, json_content,  save_path + "image_" + number + "mask_bounding_original" + ".png")
            