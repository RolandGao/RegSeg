# This code is adapted from
# https://github.com/facebookresearch/pycls/blob/f8cd962737e33ce9e19b3083a33551da95c2d9c0/pycls/datasets/augment.py
# and https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/auto_augment.py
# RandAugment - https://arxiv.org/abs/1909.13719

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


# Minimum value for posterize (0 in EfficientNet implementation)
POSTERIZE_MIN = 1

# Parameters for affine warping and rotation
WARP_PARAMS = {"fillcolor": (128, 128, 128), "resample": Image.BILINEAR}


def affine_warp(im, data):
    """Applies affine transform to image."""
    return im.transform(im.size, Image.AFFINE, data, **WARP_PARAMS)


OP_FUNCTIONS = {
    # Each op takes an image x and a level v and returns an augmented image.
    "auto_contrast": lambda x, _: ImageOps.autocontrast(x),
    "equalize": lambda x, _: ImageOps.equalize(x),
    "invert": lambda x, _: ImageOps.invert(x),
    "rotate": lambda x, v: x.rotate(v, **WARP_PARAMS),
    "posterize": lambda x, v: ImageOps.posterize(x, max(POSTERIZE_MIN, int(v))),
    "posterize_inc": lambda x, v: ImageOps.posterize(x, max(POSTERIZE_MIN, 4 - int(v))),
    "solarize": lambda x, v: x.point(lambda i: i if i < int(v) else 255 - i),
    "solarize_inc": lambda x, v: x.point(lambda i: i if i < 256 - v else 255 - i),
    "solarize_add": lambda x, v: x.point(lambda i: min(255, v + i) if i < 128 else i),
    "color": lambda x, v: ImageEnhance.Color(x).enhance(v),
    "contrast": lambda x, v: ImageEnhance.Contrast(x).enhance(v),
    "brightness": lambda x, v: ImageEnhance.Brightness(x).enhance(v),
    "sharpness": lambda x, v: ImageEnhance.Sharpness(x).enhance(v),
    "color_inc": lambda x, v: ImageEnhance.Color(x).enhance(1 + v),
    "contrast_inc": lambda x, v: ImageEnhance.Contrast(x).enhance(1 + v),
    "brightness_inc": lambda x, v: ImageEnhance.Brightness(x).enhance(1 + v),
    "sharpness_inc": lambda x, v: ImageEnhance.Sharpness(x).enhance(1 + v),
    "shear_x": lambda x, v: affine_warp(x, (1, v, 0, 0, 1, 0)),
    "shear_y": lambda x, v: affine_warp(x, (1, 0, 0, v, 1, 0)),
    "trans_x": lambda x, v: affine_warp(x, (1, 0, v * x.size[0], 0, 1, 0)),
    "trans_y": lambda x, v: affine_warp(x, (1, 0, 0, 0, 1, v * x.size[1])),
}
affine_ops=[
    "rotate","shear_x","shear_y","trans_x","trans_y"
]


OP_RANGES = {
    # Ranges for each op in the form of a (min, max, negate).
    "auto_contrast": (0, 1, False),
    "equalize": (0, 1, False),
    "invert": (0, 1, False),
    "rotate": (0.0, 30.0, True),
    "posterize": (0, 4, False),
    "posterize_inc": (0, 4, False),
    "solarize": (0, 256, False),
    "solarize_inc": (0, 256, False),
    "solarize_add": (0, 110, False),
    "color": (0.1, 1.9, False),
    "contrast": (0.1, 1.9, False),
    "brightness": (0.1, 1.9, False),
    "sharpness": (0.1, 1.9, False),
    "color_inc": (0, 0.9, True),
    "contrast_inc": (0, 0.9, True),
    "brightness_inc": (0, 0.9, True),
    "sharpness_inc": (0, 0.9, True),
    "shear_x": (0.0, 0.3, True),
    "shear_y": (0.0, 0.3, True),
    "trans_x": (0.0, 0.45, True),
    "trans_y": (0.0, 0.45, True),
}

# Scaling Wide Residual Networks for Panoptic Segmentation
# https://arxiv.org/abs/2011.11675
PANOPTIC_POLICY=[
    [("sharpness", 0.4, 0.7), ("brightness", 0.2, 1.0)],
    [("equalize", 0, 0.9), ("contrast", 0.2, 1.0)],
    [("sharpness", 0.2, 0.9), ("color", 0.2, 0.9)],
    [("solarize", 0.2, 0.7), ("equalize", 0.6, 0.9)],
    [("sharpness", 0.2, 0.2), ("solarize", 0.2, 1.4)],
]


RANDAUG_OPS = [
    # RandAugment list of operations using "increasing" transforms.
    "auto_contrast",
    "equalize",
    #"invert",
    "rotate",
    "posterize_inc",
    "solarize_inc",
    "solarize_add",
    "color_inc",
    "contrast_inc",
    "brightness_inc",
    "sharpness_inc",
    "shear_x",
    "shear_y",
    "trans_x",
    "trans_y",
]
RANDAUG_OPS_REDUCED = [
    "auto_contrast",
    "equalize",
    "rotate",
    "color_inc",
    "contrast_inc",
    "brightness_inc",
    "sharpness_inc",
]

def check_support():
    mask=np.zeros((100,100)).astype("uint8")
    mask=Image.fromarray(mask)
    magnitude=1.0
    for op in RANDAUG_OPS:
        min_v, max_v, negate = OP_RANGES[op]
        v = magnitude * (max_v - min_v) + min_v
        v = -v if negate and random.random() > 0.5 else v
        OP_FUNCTIONS[op](mask, v)


def apply_op(im, op, prob, magnitude):
    """Apply the selected op to image with given probability and magnitude."""
    # The magnitude is converted to an absolute value v for an op (some ops use -v or v)
    assert 0 <= magnitude <= 1
    assert op in OP_RANGES and op in OP_FUNCTIONS, "unknown op " + op
    if prob < 1 and random.random() > prob:
        return im
    min_v, max_v, negate = OP_RANGES[op]
    v = magnitude * (max_v - min_v) + min_v
    v = -v if negate and random.random() > 0.5 else v
    return OP_FUNCTIONS[op](im, v)

def apply_op_both(im,mask, op, prob, magnitude,fill,ignore_value=255):
    """Apply the selected op to image with given probability and magnitude."""
    # The magnitude is converted to an absolute value v for an op (some ops use -v or v)
    assert 0 <= magnitude <= 1
    assert op in OP_RANGES and op in OP_FUNCTIONS, "unknown op " + op
    if prob < 1 and random.random() > prob:
        return im,mask
    min_v, max_v, negate = OP_RANGES[op]
    v = magnitude * (max_v - min_v) + min_v
    v = -v if negate and random.random() > 0.5 else v
    WARP_PARAMS["fillcolor"]=fill
    WARP_PARAMS["resample"]=Image.BILINEAR
    im=OP_FUNCTIONS[op](im, v)
    if op in affine_ops:
        WARP_PARAMS["fillcolor"]=ignore_value
        WARP_PARAMS["resample"]=Image.NEAREST
        mask=OP_FUNCTIONS[op](mask, v)
    return im,mask

def rand_augment_both(im, mask,magnitude, ops, n_ops=2, prob=1.0,fill=(128,128,128),ignore_value=255):
    """Applies random augmentation to an image."""
    ops = ops if ops else RANDAUG_OPS
    if ops=="full":
        ops=RANDAUG_OPS
    elif ops=="reduced":
        ops=RANDAUG_OPS_REDUCED
    else:
        raise NotImplementedError()
    for op in random.sample(ops,int(n_ops)):
        im,mask = apply_op_both(im,mask, op, prob, magnitude,fill,ignore_value)
    return im,mask

def rand_augment(im, magnitude, ops=None, n_ops=2, prob=1.0):
    """Applies random augmentation to an image."""
    ops = ops if ops else RANDAUG_OPS
    for op in random.sample(ops,int(n_ops)):
        im = apply_op(im, op, prob, magnitude)
    return im

def panoptic_auto_aug(im, mask,fill=(128,128,128),ignore_value=255):
    policy=PANOPTIC_POLICY
    for op, prob, magnitude in random.choice(policy):
        im,mask = apply_op_both(im,mask, op, prob, magnitude,fill,ignore_value)
    return im,mask

def visualize_ops(im, ops=None, num_steps=10):
    """Visualize ops by applying each op by varying amounts."""
    ops = ops if ops else RANDAUG_OPS
    w, h, magnitudes = im.size[0], im.size[1], np.linspace(0, 1, num_steps)
    output = Image.new("RGB", (w * num_steps, h * len(ops)))
    for i, op in enumerate(ops):
        for j, m in enumerate(magnitudes):
            out = apply_op(im, op, prob=1.0, magnitude=m)
            output.paste(out, (j * w, i * h))
    return output

def visualize_opsXops(im, ops, magnitude):
    ops = ops if ops else RANDAUG_OPS
    w, h = im.size[0], im.size[1]
    output = Image.new("RGB", (w * len(ops), h * len(ops)))
    for i, op in enumerate(ops):
        for j, op2 in enumerate(ops):
            out = apply_op(im, op, prob=1.0, magnitude=magnitude)
            out = apply_op(out, op2, prob=1.0, magnitude=magnitude)
            output.paste(out, (j * w, i * h))
    return output


def visualize_aug(im, augment=rand_augment, num_trials=10, **kwargs):
    """Visualize augmentation by applying random augmentations."""
    w, h = im.size[0], im.size[1]
    output = Image.new("RGB", (w * num_trials, h * num_trials))
    for i in range(num_trials):
        for j in range(num_trials):
            output.paste(augment(im, **kwargs), (j * w, i * h))
    return output

if __name__=="__main__":
    im=Image.open("cityscapes_dataset_half2/leftImg8bit/train/cologne/cologne_000114_000019_leftImg8bit.png")
    w, h = im.size[0], im.size[1]
    im=im.resize((w//4,h//4))
    output=visualize_aug(im,magnitude=0.2,ops=RANDAUG_OPS_REDUCED)
    output.show()
    output=visualize_opsXops(im, RANDAUG_OPS_REDUCED, 0.2)
    output.show()
    output=visualize_ops(im,RANDAUG_OPS_REDUCED)
    output.show()
