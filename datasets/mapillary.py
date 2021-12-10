from PIL import Image
import torch.utils.data as data
import os
import numpy as np

def reduce_mask(mask):
    ious=[0.0, 0.0, 57.68, 58.66, 63.16, 56.59, 50.9, 45.04, 39.82, 18.31, 22.22, 45.77, 49.91, 87.97, 43.31, 70.61, 76.67, 86.43, 41.84, 66.81, 46.77, 50.41, 0.0, 69.53, 57.07, 48.28, 4.99, 97.77, 76.83, 68.69, 88.77, 72.93, 17.02, 22.26, 5.05, 45.31, 29.76, 0.0, 20.38, 36.26, 2.43, 43.12, 4.4, 0.0, 37.03, 40.48, 52.69, 44.16, 60.96, 36.36, 66.61, 43.94, 47.44, 16.69, 73.89, 89.68, 0.0, 55.74, 46.28, 22.28, 6.71, 67.39, 8.41, 68.79, 91.75]
    ious=np.array(ious)
    relabelling=[]
    i=0
    for iou in ious:
        if iou>30:
            relabelling.append(i)
            i+=1
        else:
            relabelling.append(65) #ignore class
    relabelling.append(65) #ignore class
    mask=np.array(mask)
    relabelling=np.array(relabelling).astype(mask.dtype)
    new_mask=relabelling[mask]
    new_mask=Image.fromarray(new_mask)
    return new_mask

class Mapillary(data.Dataset):
    def __init__(self,root,image_set,transforms,reduced,version="v1.2"):
        # images, masks, json splits
        assert version in ["v1.2","v2.0"]
        root= os.path.expanduser(root)
        dic={"train":"training","val":"validation","test":"testing"}
        image_set=dic[image_set]
        # image_path = "training/images/{}.jpg".format(image_id)
        # label_path = "training/{}/labels/{}.png".format(version, image_id)
        image_dir=os.path.join(root,image_set,"images")
        mask_dir=os.path.join(root,image_set,version,"labels")
        file_names=[]
        for filename in sorted(os.listdir(image_dir)):
            assert filename[0] != "."
            file_names.append(filename[:-4])
        self.transforms=transforms
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        self.reduced=reduced
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])# .convert('L')
        if self.reduced:
            target=reduce_mask(target)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

if __name__=="__main__":
    dataset=Mapillary("../mapillary_dataset","train",None,True)
    x=dataset[0]
    print(x)
