import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import time
from data import get_cityscapes,get_pascal_voc, build_val_transform
from datasets.cityscapes import Cityscapes
from model import RegSeg
from train import get_dataset_loaders
import yaml
from data_utils import get_dataloader_val
import torchvision.transforms as T
from datasets.camvid import Camvid

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def get_colors():
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.arange(255).view(-1, 1) * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors
def get_colors_cityscapes():
    colors=np.zeros((256,3))
    colors[255]=[255,255,255]
    for c in Cityscapes.classes:
        if 0<=c.train_id<=18:
            colors[c.train_id]=c.color
    return colors.astype("uint8")
def get_colors_cityscapes_labelid():
    colors=np.zeros((256,3))
    colors[255]=[255,255,255]
    for c in Cityscapes.classes:
        colors[c.id]=c.color
    return colors.astype("uint8")
def get_colors_mapillary():
    #colors=[[165, 42, 42], [0, 192, 0], [250, 170, 31], [250, 170, 32], [196, 196, 196], [190, 153, 153], [180, 165, 180], [90, 120, 150], [250, 170, 33], [250, 170, 34], [128, 128, 128], [250, 170, 35], [102, 102, 156], [128, 64, 255], [140, 140, 200], [170, 170, 170], [250, 170, 36], [250, 170, 160], [250, 170, 37], [96, 96, 96], [230, 150, 140], [128, 64, 128], [110, 110, 110], [110, 110, 110], [244, 35, 232], [128, 196, 128], [150, 100, 100], [70, 70, 70], [150, 150, 150], [150, 120, 90], [220, 20, 60], [220, 20, 60], [255, 0, 0], [255, 0, 100], [255, 0, 200], [255, 255, 255], [255, 255, 255], [250, 170, 29], [250, 170, 28], [250, 170, 26], [250, 170, 25], [250, 170, 24], [250, 170, 22], [250, 170, 21], [250, 170, 20], [255, 255, 255], [250, 170, 19], [250, 170, 18], [250, 170, 12], [250, 170, 11], [255, 255, 255], [255, 255, 255], [250, 170, 16], [250, 170, 15], [250, 170, 15], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [64, 170, 64], [230, 160, 50], [70, 130, 180], [190, 255, 255], [152, 251, 152], [107, 142, 35], [0, 170, 30], [255, 255, 128], [250, 0, 30], [100, 140, 180], [220, 128, 128], [222, 40, 40], [100, 170, 30], [40, 40, 40], [33, 33, 33], [100, 128, 160], [20, 20, 255], [142, 0, 0], [70, 100, 150], [250, 171, 30], [250, 172, 30], [250, 173, 30], [250, 174, 30], [250, 175, 30], [250, 176, 30], [210, 170, 100], [153, 153, 153], [153, 153, 153], [128, 128, 128], [0, 0, 80], [210, 60, 60], [250, 170, 30], [250, 170, 30], [250, 170, 30], [250, 170, 30], [250, 170, 30], [250, 170, 30], [192, 192, 192], [192, 192, 192], [192, 192, 192], [220, 220, 0], [220, 220, 0], [0, 0, 196], [192, 192, 192], [220, 220, 0], [140, 140, 20], [119, 11, 32], [150, 0, 255], [0, 60, 100], [0, 0, 142], [0, 0, 90], [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110], [0, 0, 70], [0, 0, 142], [0, 0, 192], [170, 170, 170], [32, 32, 32], [111, 74, 0], [120, 10, 10], [81, 0, 81], [111, 111, 0], [0, 0, 0]]
    colors=[[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153], [180, 165, 180], [90, 120, 150], [102, 102, 156], [128, 64, 255], [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96], [230, 150, 140], [128, 64, 128], [110, 110, 110], [244, 35, 232], [150, 100, 100], [70, 70, 70], [150, 120, 90], [220, 20, 60], [255, 0, 0], [255, 0, 100], [255, 0, 200], [200, 128, 128], [255, 255, 255], [64, 170, 64], [230, 160, 50], [70, 130, 180], [190, 255, 255], [152, 251, 152], [107, 142, 35], [0, 170, 30], [255, 255, 128], [250, 0, 30], [100, 140, 180], [220, 220, 220], [220, 128, 128], [222, 40, 40], [100, 170, 30], [40, 40, 40], [33, 33, 33], [100, 128, 160], [142, 0, 0], [70, 100, 150], [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 80], [250, 170, 30], [192, 192, 192], [220, 220, 0], [140, 140, 20], [119, 11, 32], [150, 0, 255], [0, 60, 100], [0, 0, 142], [0, 0, 90], [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110], [0, 0, 70], [0, 0, 192], [32, 32, 32], [120, 10, 10], [0, 0, 0]]
    colors=np.array(colors).astype("uint8")
    return colors
def get_colors_mapillary_reduced():
    colors=[[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153], [180, 165, 180], [90, 120, 150], [102, 102, 156], [128, 64, 255], [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96], [230, 150, 140], [128, 64, 128], [110, 110, 110], [244, 35, 232], [150, 100, 100], [70, 70, 70], [150, 120, 90], [220, 20, 60], [255, 0, 0], [255, 0, 100], [255, 0, 200], [200, 128, 128], [255, 255, 255], [64, 170, 64], [230, 160, 50], [70, 130, 180], [190, 255, 255], [152, 251, 152], [107, 142, 35], [0, 170, 30], [255, 255, 128], [250, 0, 30], [100, 140, 180], [220, 220, 220], [220, 128, 128], [222, 40, 40], [100, 170, 30], [40, 40, 40], [33, 33, 33], [100, 128, 160], [142, 0, 0], [70, 100, 150], [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 80], [250, 170, 30], [192, 192, 192], [220, 220, 0], [140, 140, 20], [119, 11, 32], [150, 0, 255], [0, 60, 100], [0, 0, 142], [0, 0, 90], [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110], [0, 0, 70], [0, 0, 192], [32, 32, 32], [120, 10, 10], [0, 0, 0]]
    colors=np.array(colors).astype("uint8")
    ious=[0.0, 0.0, 57.68, 58.66, 63.16, 56.59, 50.9, 45.04, 39.82, 18.31, 22.22, 45.77, 49.91, 87.97, 43.31, 70.61, 76.67, 86.43, 41.84, 66.81, 46.77, 50.41, 0.0, 69.53, 57.07, 48.28, 4.99, 97.77, 76.83, 68.69, 88.77, 72.93, 17.02, 22.26, 5.05, 45.31, 29.76, 0.0, 20.38, 36.26, 2.43, 43.12, 4.4, 0.0, 37.03, 40.48, 52.69, 44.16, 60.96, 36.36, 66.61, 43.94, 47.44, 16.69, 73.89, 89.68, 0.0, 55.74, 46.28, 22.28, 6.71, 67.39, 8.41, 68.79, 91.75,0]
    ious=np.array(ious)
    colors=colors[ious>30]
    all_colors=np.zeros((256,3)).astype("uint8")
    all_colors[:len(colors)]=colors
    return all_colors
def get_colors_camvid(color_to_class):
    colors=np.zeros((256,3))
    colors[255]=[255,255,255]
    for color,cls in color_to_class.items():
        colors[cls]=color
    return colors.astype("uint8")

def show_image(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
def _open(filename):
    image = Image.open(filename).convert("RGB")
    preprocess = T.Compose([
        T.Resize(1024),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    input_tensor = preprocess(image)
    images = input_tensor.unsqueeze(0)
    return images
def show_mask(images):
    colors=get_colors()
    r = Image.fromarray(images.byte().cpu().numpy())
    r.putpalette(colors)
    plt.imshow(r)
def show_cityscapes_mask(images):
    colors=get_colors_cityscapes()
    r = Image.fromarray(images.byte().cpu().numpy())
    r.putpalette(colors)
    plt.imshow(r)
def show_camvid_mask(images,colors):
    r = Image.fromarray(images.byte().cpu().numpy())
    r.putpalette(colors)
    plt.imshow(r)
def show_mapillary_mask(images,reduced=False):
    colors=get_colors_mapillary()
    if reduced:
        colors=get_colors_mapillary_reduced()
    r = Image.fromarray(images.byte().cpu().numpy())
    r.putpalette(colors)
    plt.imshow(r)

def display(data_loader,show_mask,num_images=5,skip=4,images_per_line=6):
    images_so_far = 0
    fig = plt.figure(figsize=(6, 4))
    num_rows=int(np.ceil(num_images/images_per_line))
    data_loader = iter(data_loader)
    for _ in range(skip):
        next(data_loader)
    for images, targets in data_loader:
        for image, target in zip(images, targets):
            print(image.size(), target.size())
            plt.subplot(num_rows, 2*images_per_line, images_so_far + 1)
            plt.axis('off')
            show_image(image)

            plt.subplot(num_rows, 2*images_per_line, images_so_far + 2)
            plt.axis('off')
            show_mask(target)

            images_so_far += 2
            if images_so_far == 2 * num_images:
                plt.tight_layout()
                plt.show()
                return
    plt.tight_layout()
    plt.show()
def show(model,data_loader,device,show_mask,num_images=5,skip=4,images_per_line=2):
    images_so_far=0
    model.eval()
    num_rows = int(np.ceil(num_images / images_per_line))
    fig=plt.figure(figsize=(8,4))
    data_loader=iter(data_loader)
    for _ in range(skip):
        next(data_loader)
    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images.to(device), targets.to(device)
            start=time.time()
            outputs = model(images)
            end=time.time()
            print(end-start)
            for image,target,output in zip(images,targets,outputs):
                output = output.argmax(0)
                print(image.size(),target.size(),output.size())
                plt.subplot(num_rows, 3*images_per_line, images_so_far+1)
                plt.axis('off')
                show_image(image)

                plt.subplot(num_rows, 3*images_per_line, images_so_far+2)
                plt.axis('off')
                show_mask(target)

                plt.subplot(num_rows,3*images_per_line,images_so_far+3)
                plt.axis('off')
                show_mask(output)

                images_so_far+=3
                if images_so_far==3*num_images:
                    plt.tight_layout()
                    plt.show()
                    return
    plt.tight_layout()
    plt.show()
def show_files(model,files,device,show_mask,num_images=5,images_per_line=2):
    images_so_far=0
    model.eval()
    num_rows = int(np.ceil(num_images / images_per_line))
    fig=plt.figure(figsize=(8,4))
    with torch.no_grad():
        for filename in files:
            images=_open(filename)
            images= images.to(device)
            start=time.time()
            outputs = model(images)
            end=time.time()
            print(end-start)
            for image,output in zip(images,outputs):
                output = output.argmax(0)
                print(image.size(),output.size())
                plt.subplot(num_rows, 2*images_per_line, images_so_far+1)
                plt.axis('off')
                show_image(image)

                plt.subplot(num_rows, 2*images_per_line, images_so_far+2)
                plt.axis('off')
                show_mask(output)

                images_so_far+=2
                if images_so_far==2*num_images:
                    plt.tight_layout()
                    plt.show()
                    return
    plt.tight_layout()
    plt.show()

def display_cityscapes():
    import random
    num_images=16
    images_per_line=4
    skip=0
    seed=0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    config_filename="configs/cityscapes_500epochs.yaml"
    with open(config_filename) as file:
        config=yaml.full_load(file)
    config["class_uniform_pct"]=0.5
    config["dataset_dir"]="cityscapes_dataset"
    config["val_split"]="val"
    config["train_split"]="train"
    config["train_min_size"]=400
    config["train_max_size"]=1600
    config["train_crop_size"]=[1024,1024]
    config["aug_mode"]="randaug_reduced"
    config["num_workers"]=0
    train_loader,val_loader,train_set=get_dataset_loaders(config)
    display(val_loader,show_cityscapes_mask,num_images=num_images,skip=skip,images_per_line=images_per_line)
def display_camvid():
    import random
    num_images=9
    images_per_line=3
    skip=0
    seed=0
    torch.manual_seed(seed)
    random.seed(seed)
    config_filename= "configs/camvid_200epochs.yaml"
    with open(config_filename) as file:
        config=yaml.full_load(file)
    config["train_split"]="trainval"
    config["val_split"]="test"
    config["dataset_dir"]="CamVid3"
    train_loader,val_loader,train_set=get_dataset_loaders(config)
    colours=get_colors_camvid(train_set.color_to_class)
    _show_mask = lambda image : show_camvid_mask(image,colours)
    display(train_loader,_show_mask,num_images=num_images,skip=skip,images_per_line=images_per_line)
def display_coco():
    import random
    num_images=9
    images_per_line=3
    skip=0
    seed=0
    torch.manual_seed(seed)
    random.seed(seed)
    config_filename="configs/coco_exp30_decoder12_5epochs.yaml"
    with open(config_filename) as file:
        config=yaml.full_load(file)
    config["aug_mode"]="baseline"
    config["dataset_name"]="coco2"
    config["dataset_dir"]="coco"
    train_loader,val_loader,train_set=get_dataset_loaders(config)
    display(val_loader,show_mask,num_images=num_images,skip=skip,images_per_line=images_per_line)
def dispay_mapillary():
    import random
    num_images=16
    images_per_line=4
    skip=16
    seed=0
    torch.manual_seed(seed)
    random.seed(seed)
    config_filename="configs/mapillary_240epochs.yaml"
    with open(config_filename) as file:
        config=yaml.full_load(file)
    train_loader,val_loader,train_set=get_dataset_loaders(config)
    display(train_loader,lambda images:show_mapillary_mask(images,False),num_images=num_images,skip=skip,images_per_line=images_per_line)

def show_mapillary_model():
    import random
    model=RegSeg(
        name="exp49_decoder14",
        num_classes=65,
        pretrained="checkpoints/mapillary_exp49_decoder14_300_epochs_run1"
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_images=10
    images_per_line=2
    skip=0
    config_filename="configs/mapillary_240epochs.yaml"
    with open(config_filename) as file:
        config=yaml.full_load(file)
    config["num_workers"]=0
    config["batch_size"]=1
    seed=0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    train_loader,val_loader,train_set=get_dataset_loaders(config)
    show(model,train_loader,device,show_mapillary_mask,num_images=num_images,skip=skip,images_per_line=images_per_line)
def show_cityscapes_test():
    import os
    model=RegSeg(
        name="exp48_decoder26",
        num_classes=19,
        pretrained="checkpoints/cityscapes_exp48_decoder26_train_1000_epochs_run2"
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    files=["munich/munich_000203_000019_leftImg8bit.png",
           "munich/munich_000235_000019_leftImg8bit.png",
           "munich/munich_000292_000019_leftImg8bit.png",
           "munich/munich_000298_000019_leftImg8bit.png"
           ]
    # files=["berlin/berlin_000129_000019_leftImg8bit.png",
    #        "berlin/berlin_000019_000019_leftImg8bit.png",
    #        "berlin/berlin_000035_000019_leftImg8bit.png",
    #        "berlin/berlin_000043_000019_leftImg8bit.png",
    #        "berlin/berlin_000087_000019_leftImg8bit.png",
    #        "berlin/berlin_000117_000019_leftImg8bit.png"
    #        ]
    new_files=[]
    for file in files:
        file=os.path.join("cityscapes_dataset/leftImg8bit/test",file)
        new_files.append(file)
    num_images=len(files)
    images_per_line=2
    show_files(model,new_files,device,show_cityscapes_mask,num_images=num_images,images_per_line=images_per_line)
def show_cityscapes_model():
    import random
    model=RegSeg(
        name="exp48_decoder26",
        num_classes=19,
        pretrained="checkpoints/cityscapes_exp48_decoder26_train_1000_epochs_run2"
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_images=4
    images_per_line=1
    skip=8
    config_filename="configs/cityscapes_1000epochs.yaml"
    with open(config_filename) as file:
        config=yaml.full_load(file)
    config["num_workers"]=0
    config["batch_size"]=1
    config["class_uniform_pct"]=0
    config["train_crop_size"]=[1024,1024*2]
    config["train_max_size"]=1024
    config["train_min_size"]=1024
    config["dataset_dir"]="cityscapes_dataset"
    seed=0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    train_loader,val_loader,train_set=get_dataset_loaders(config)
    show(model,val_loader,device,show_cityscapes_mask,num_images=num_images,skip=skip,images_per_line=images_per_line)
def show_cityscapes_failure_modes():
    num_images=4
    images_per_line=1
    skip=0
    model=RegSeg(
        name="exp48_decoder26",
        num_classes=19,
        pretrained="checkpoints/cityscapes_exp48_decoder26_train_1000_epochs_run2"
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    val_transform=build_val_transform(1024,1024)
    indices=[312, 84, 54, 161, 230, 319, 303, 297, 262, 310, 263, 318, 360, 11, 321, 290, 257, 283, 121, 101, 276, 100, 317, 124, 175, 267, 278, 178, 231, 228]
    # indices=[312, 84, 161, 228, 54, 319, 313, 310, 230, 317, 262, 276, 318, 66, 301, 296, 321, 360, 140, 257, 290, 101, 83, 178, 263, 278, 121, 147, 26, 125]
    val = Cityscapes("cityscapes_dataset", split="val", target_type="semantic",
                     transforms=val_transform, class_uniform_pct=0)
    val=torch.utils.data.Subset(val,indices)
    val_loader = get_dataloader_val(val, 0)
    show(model,val_loader,device,show_cityscapes_mask,num_images=num_images,skip=skip,images_per_line=images_per_line)
if __name__=="__main__":
    show_cityscapes_model()
