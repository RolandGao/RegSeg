# RegSeg

#### The official implementation of "Rethink Dilated Convolution for Real-time Semantic Segmentation"

Paper: [arxiv](https://arxiv.org/abs/2111.09957)

<table border="0">
<tr>
    <td>
    <img src="./figs/miou_vs_params.png" width="90%" />
    </td>
    <td>
    <img src="./figs/miou_vs_flops.png" width="90%" />
    </td>
</tr>
</table>

D block

<img src="./figs/DBlock1.png" width="50%" />

Decoder

<img src="./figs/Decoder.png" width="50%" />


### Setup
Install the dependencies in requirements.txt by using pip and [virtualenv](
https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

### Download Cityscapes
go to https://www.cityscapes-dataset.com, create an account, and download
gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip.
Unzip both of them and put them in a directory called cityscapes_dataset.
The cityscapes_dataset directory should be inside the RegSeg directory.
If you put the dataset somewhere else, you can set the config field
```
config["dataset_dir"]="the location of your dataset"
```
You can delete the test images to save some space if you don't want to submit to the competition.
Make sure that you have downloaded the required python packages and run
```
CITYSCAPES_DATASET=cityscapes_dataset csCreateTrainIdLabelImgs
```
There are 19 classes.

### Results from paper
To see the ablation studies results from the paper, go [here](AblationStudies.md).

### Usage
To visualize your model, go to show.py.

To see the model definitions and do some speed tests, go to model.py.

To train, validate, benchmark, and save the results of your model, go to train.py.

### Cityscapes test results
RegSeg (exp48_decoder26, 30FPS)

[test mIOU: 78.3](https://www.cityscapes-dataset.com/anonymous-results/?id=f88876222a7be564973065f111746e9838d4da9268734457cb57b2409cdb9818)

[model weights](https://github.com/RolandGao/RegSeg/releases/download/v1.0-alpha/cityscapes_exp48_decoder26_trainval_1000_epochs_1024_crop_bootstrapped_run1)

Larger RegSeg (exp53_decoder29, 20 FPS)

[test mIOU: 79.5](https://www.cityscapes-dataset.com/anonymous-results/?id=d15ac10b39ba00bcb344620a423dfe970ece04562a763d8f3f8d0d44376727ae)

[model weights](https://github.com/RolandGao/RegSeg/releases/download/v1.0-alpha/cityscapes_exp53_decoder29_trainval_1000_epochs_1024_crop_bootstrapped_run1)

### Comparison against DDRNet-23
Run | RegSeg model weights | DDRNet-23 model weights
--- | --- | --- 
run1 | [77.76](https://github.com/RolandGao/RegSeg/releases/download/v1.0-alpha/cityscapes_exp48_decoder26_train_1000_epochs_run1) | [77.84](https://github.com/RolandGao/RegSeg/releases/download/v1.0-alpha/cityscapes_ddrnet23_1000_epochs_run1)
run2 | [78.85](https://github.com/RolandGao/RegSeg/releases/download/v1.0-alpha/cityscapes_exp48_decoder26_train_1000_epochs_run2) | [78.07](https://github.com/RolandGao/RegSeg/releases/download/v1.0-alpha/cityscapes_ddrnet23_1000_epochs_run2)
run3 | [78.07](https://github.com/RolandGao/RegSeg/releases/download/v1.0-alpha/cityscapes_exp48_decoder26_train_1000_epochs_run3) | [77.53](https://github.com/RolandGao/RegSeg/releases/download/v1.0-alpha/cityscapes_ddrnet23_1000_epochs_run3)

### Citation
If you find our work helpful, please consider citing our paper.


```
@article{gao2021rethink,
  title={Rethink Dilated Convolution for Real-time Semantic Segmentation},
  author={Gao, Roland},
  journal={arXiv preprint arXiv:2111.09957},
  year={2021}
}
```
