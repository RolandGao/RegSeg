import torch
import torch.cuda.amp as amp
from train import ConfusionMatrix
import yaml
from model import RegSeg
from train_utils import get_dataset_loaders


@torch.no_grad()
def find_failure_modes():
    config_filename = "configs/cityscapes_1000epochs.yaml"
    with open(config_filename) as file:
        config = yaml.full_load(file)
    config["dataset_dir"]="cityscapes_dataset"
    config["class_uniform_pct"]=0
    train_loader, val_loader, train_set = get_dataset_loaders(config)
    model=RegSeg(
        name="exp48_decoder26",
        num_classes=19,
        pretrained="checkpoints/cityscapes_exp48_decoder26_train_1000_epochs_run2"
    ).cuda()
    model.eval()
    accuracy_list = []
    for i, (image, target) in enumerate(val_loader):
        image, target = image.cuda(), target.cuda()
        with amp.autocast(enabled=True):
            output = model(image)
        confmat = ConfusionMatrix(19, [])
        confmat.update(target.flatten(), output.argmax(1).flatten())
        acc_global, acc, iu = confmat.compute()
        accuracy_list.append(acc_global)
        if (i + 1) % 50 == 0:
            print(i + 1)
    print(accuracy_list)
    accuracy_list = torch.Tensor(accuracy_list)
    indices = torch.argsort(accuracy_list).tolist()
    indices = indices[:30]
    print(indices)
    print(accuracy_list[indices])

if __name__=="__main__":
    find_failure_modes()
