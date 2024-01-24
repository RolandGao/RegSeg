import datetime
import time
import torch
import yaml
import torch.cuda.amp as amp
import os
import copy
import random
import numpy as np
from train_utils import get_lr_function, get_loss_fun,get_optimizer,get_dataset_loaders,get_model,get_val_dataset
from benchmark import compute_time_full, compute_time_no_loader,compute_loader_time
from precise_bn import compute_precise_bn_stats

class ConfusionMatrix(object):
    def __init__(self, num_classes, exclude_classes):
        self.num_classes = num_classes
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        self.exclude_classes=exclude_classes

    def update(self, a, b):
        a=a.cpu()
        b=b.cpu()
        n = self.num_classes
        k = (a >= 0) & (a < n)
        inds = n * a + b
        inds=inds[k]
        self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))

        acc_global=acc_global.item() * 100
        acc=(acc * 100).tolist()
        iu=(iu * 100).tolist()
        return acc_global, acc, iu
    def __str__(self):
        acc_global, acc, iu = self.compute()
        acc_global=round(acc_global,2)
        IOU=[round(i,2) for i in iu]
        mIOU=sum(iu)/len(iu)
        mIOU=round(mIOU,2)
        reduced_iu=[iu[i] for i in range(self.num_classes) if i not in self.exclude_classes]
        mIOU_reduced=sum(reduced_iu)/len(reduced_iu)
        mIOU_reduced=round(mIOU_reduced,2)
        return f"IOU: {IOU}\nmIOU: {mIOU}, mIOU_reduced: {mIOU_reduced}, accuracy: {acc_global}"

def evaluate(model, data_loader, device, confmat,mixed_precision,print_every,max_eval):
    model.eval()
    assert(isinstance(confmat,ConfusionMatrix))
    with torch.no_grad():
        for i,(image, target) in enumerate(data_loader):
            if (i+1)%print_every==0:
                print(i+1)
            image, target = image.to(device), target.to(device)
            with amp.autocast(enabled=mixed_precision):
                output = model(image)
            output = torch.nn.functional.interpolate(output, size=target.shape[-2:], mode='bilinear', align_corners=False)
            confmat.update(target.flatten(), output.argmax(1).flatten())
            if i+1==max_eval:
                break
    return confmat

def train_one_epoch(model, loss_fun, optimizer, loader, lr_scheduler, print_every, mixed_precision, scaler):
    model.train()
    losses=0
    for t, x in enumerate(loader):
        image, target=x
        image, target = image.cuda(), target.cuda()
        with amp.autocast(enabled=mixed_precision):
            output = model(image)
            #print(output.size())
            loss = loss_fun(output, target)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        losses+=loss.item()
        if (t+1) % print_every==0:
            print(t+1,loss.item())
    num_iter=len(loader)
    print(losses/num_iter)
    return losses/num_iter

def save(model,optimizer,scheduler,epoch,path,best_mIU,scaler,run):
    dic={
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': scheduler.state_dict(),
        'scaler':scaler.state_dict(),
        'epoch': epoch,
        'best_mIU':best_mIU,
        "run":run
    }
    print(os.path.realpath(path))
    torch.save(dic,path)

def get_config_and_check_files(config_filename):
    with open(config_filename) as file:
        config=yaml.full_load(file)
    return check_config_files(config)
def check_config_files(config):
    save_dir=config["save_dir"]
    log_dir=config["log_dir"]
    config["save_best_path"]=os.path.join(save_dir,config["save_name"]+f"_run{config['run']}")
    config["save_latest_path"]=os.path.join(save_dir,config["save_name"]+"_latest")
    config["resume_path"]=config["save_latest_path"]
    config["log_path"]=os.path.join(config["log_dir"],config["save_name"]+"_log.txt")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    if not os.path.isdir(config["dataset_dir"]):
        raise FileNotFoundError(f"{config['dataset_dir']} is not a directory")
    if config["resume"]:
        if not os.path.isfile(config["resume_path"]):
            config["resume_path"]=config["save_best_path"]
        if not os.path.isfile(config["resume_path"]):
            raise FileNotFoundError(f"{config['resume_path']} is not a file")
    elif not config["pretrained_backbone"]:
        if config["pretrained_path"] != "" and not os.path.isfile(config["pretrained_path"]):
            raise FileNotFoundError(f"{config['pretrained_path']} is not a file")
    return config

def get_epochs_to_save(config):
    if not config["eval_while_train"]:
        print("warning: no checkpoint/eval during training")
        return []
    epochs=config["epochs"]
    save_every_k_epochs=config["save_every_k_epochs"]
    save_best_on_epochs=[i*save_every_k_epochs-1 for i in range(1,epochs//save_every_k_epochs+1)]
    if epochs-1 not in save_best_on_epochs:
        save_best_on_epochs.append(epochs-1)
    if 0 not in save_best_on_epochs:
        save_best_on_epochs.append(0)
    if "save_last_k_epochs" in config:
        for i in range(max(epochs-config["save_last_k_epochs"],0),epochs):
            if i not in save_best_on_epochs:
                save_best_on_epochs.append(i)
    save_best_on_epochs=sorted(save_best_on_epochs)
    return save_best_on_epochs
def setup_env(config):
    torch.backends.cudnn.benchmark=True
    seed=0
    if "RNG_seed" in config:
        seed=config["RNG_seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed) # might remove dependency on np later
def train_multiple(configs):
    global_accuracies=[]
    mIOUs=[]
    new_configs=[]
    for config in configs:
        new_configs.append(config)
    for config in new_configs:
        best_mIU,best_global_accuracy=train_one(config)
        mIOUs.append(best_mIU)
        global_accuracies.append(best_global_accuracy)
    log_path=configs[0]["log_path"]
    with open(log_path,"a") as f:
        f.write(f"mIOUs: {mIOUs}\n")
        f.write(f"global_accuracies: {global_accuracies}\n")
    return mIOUs,global_accuracies
def train_one(config):
    config=check_config_files(config)
    setup_env(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    save_best_path=config["save_best_path"]
    print("saving to: "+save_best_path)
    save_latest_path=config["save_latest_path"]
    epochs=config["epochs"]
    max_epochs=config["max_epochs"]
    num_classes=config["num_classes"]
    exclude_classes=config["exclude_classes"]
    mixed_precision=config["mixed_precision"]
    log_path=config["log_path"]
    run=config["run"]
    max_eval=config["max_eval"]
    eval_print_every=config["eval_print_every"]
    train_print_every=config["train_print_every"]
    bn_precise_stats=config["bn_precise_stats"]
    bn_precise_num_samples=config["bn_precise_num_samples"]

    checkpoints = config["save_dir"]

    model=get_model(config).to(device)
    train_loader, val_loader,train_set=get_dataset_loaders(config)
    total_iterations=len(train_loader) * max_epochs
    optimizer = get_optimizer(model,config)
    scaler = amp.GradScaler(enabled=mixed_precision)
    loss_fun=get_loss_fun(config)
    lr_function=get_lr_function(config,total_iterations)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,lr_function
    )
    epoch_start=0
    best_mIU=0
    save_best_on_epochs=get_epochs_to_save(config)
    print("save on epochs: ",save_best_on_epochs)

    if config["resume"]:
        dic=torch.load(config["resume_path"],map_location='cpu')
        model.load_state_dict(dic['model'])
        optimizer.load_state_dict(dic['optimizer'])
        lr_scheduler.load_state_dict(dic['lr_scheduler'])
        epoch_start = dic['epoch'] + 1
        if "best_mIU" in dic:
            best_mIU=dic["best_mIU"]
        if "scaler" in dic:
            scaler.load_state_dict(dic["scaler"])

    start_time = time.time()
    best_global_accuracy=0
    if not config["resume"]:
        with open(log_path,"a") as f:
            f.write(f"{config}\n")
            f.write(f"run: {run}\n")
    for epoch in range(epoch_start,epochs):
        # Setting the seed to the curent epoch allows models with config["resume"]=True to be consistent.
        torch.manual_seed(epoch)
        random.seed(epoch)
        np.random.seed(epoch)
        with open(log_path,"a") as f:
            f.write(f"epoch: {epoch}\n")
        print(f"epoch: {epoch}")
        if hasattr(train_set, 'build_epoch'):
            print("build epoch")
            train_set.build_epoch()
        average_loss=train_one_epoch(model, loss_fun, optimizer, train_loader, lr_scheduler, print_every=train_print_every, mixed_precision=mixed_precision, scaler=scaler)
        with open(log_path,"a") as f:
            f.write(f"loss: {average_loss}\n")
        if epoch in save_best_on_epochs:
            if bn_precise_stats:
                print("calculating precise bn stats")
                compute_precise_bn_stats(model,train_loader,bn_precise_num_samples)
            confmat=ConfusionMatrix(num_classes,exclude_classes)
            confmat = evaluate(model, val_loader, device,confmat,
                               mixed_precision, eval_print_every,max_eval)
            with open(log_path,"a") as f:
                f.write(f"{confmat}\n")
            print(confmat)
            acc_global, acc, iu = confmat.compute()
            mIU=sum(iu)/len(iu)
            if acc_global>best_global_accuracy:
                best_global_accuracy=acc_global
            if mIU > best_mIU:
                best_mIU=mIU
                save(model, optimizer, lr_scheduler, epoch, checkpoints + "/" + save_best_path,best_mIU,scaler,run)
        if save_latest_path != "":
            save(model, optimizer, lr_scheduler, epoch, checkpoints + "/" + save_latest_path,best_mIU,scaler,run)
        # if config["model_name"]=="exp26":
        #     decode_dilations_exp26(model.body)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Best mIOU: {best_mIU}\n")
    print(f"Best global accuracy: {best_global_accuracy}\n")
    print(f"Training time {total_time_str}\n")
    with open(log_path,"a") as f:
        f.write(f"Best mIOU: {best_mIU}\n")
        f.write(f"Best global accuracy: {best_global_accuracy}\n")
        f.write(f"Training time {total_time_str}\n")
    print(f"Training time {total_time_str}")
    return best_mIU,best_global_accuracy

def validate_multiple(configs):
    confmats=[]
    for config in configs:
        confmat=validate_one(config)
        confmats.append(confmat)
    return confmats
def validate_one(config):
    setup_env(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_loader, val_loader,train_set=get_dataset_loaders(config)
    model=get_model(config).to(device)
    mixed_precision=config["mixed_precision"]
    print_every=config["eval_print_every"]
    num_classes=config["num_classes"]
    exclude_classes=config["exclude_classes"]
    confmat = ConfusionMatrix(num_classes,exclude_classes)
    max_eval=100000
    if "max_eval" in config:
        max_eval=config["max_eval"]
    loader=val_loader
    if "validate_train_loader" in config and config["validate_train_loader"]==True:
        loader=train_loader
    if config["bn_precise_stats"]:
        print("calculating precise bn stats")
        compute_precise_bn_stats(model,train_loader,config["bn_precise_num_samples"])
    print("evaluating")
    confmat = evaluate(model, loader, device,confmat,mixed_precision,
                       print_every,max_eval)
    print(confmat)
    return confmat

def save_cityscapes_results(config,pred_dir):
    from evaluator import save_results
    from PIL import Image
    setup_env(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    val=get_val_dataset(config)
    model=get_model(config).to(device)
    mixed_precision=config["mixed_precision"]
    print("evaluating")
    all_imgs=val.all_imgs
    transforms=val.transforms
    os.makedirs(pred_dir,exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i,(image_filename,target_filename) in enumerate(all_imgs):
            if (i+1)%100==0:
                print(i+1)
            image=Image.open(image_filename).convert('RGB')
            dummy_target=Image.fromarray(np.zeros((1024,2048)))
            # transforms has to take 2 inputs, currently
            image,dummy_target=transforms(image,dummy_target)
            image = image.to(device)
            image=image.unsqueeze(0)
            with amp.autocast(enabled=mixed_precision):
                output = model(image)
            #print(output.shape)
            output=torch.nn.functional.interpolate(output, size=[1024,2048], mode='bilinear', align_corners=False)
            output=output.argmax(1) # 3D output instead of 4D
            output=output.squeeze(0)
            save_results(pred_dir,image_filename,output)

def benchmark_multiple(configs):
    for config in configs:
        benchmark_one(config)

def benchmark_one(config):
    setup_env(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    mixed_precision=config["mixed_precision"]
    warmup_iter=config["warmup_iter"]
    num_iter=config["num_iter"]
    crop_size=config["train_crop_size"]
    val_input_size=config["val_input_size"]
    batch_size=config["batch_size"]
    num_classes=config["num_classes"]
    benchmark_loader=config["benchmark_loader"]
    benchmark_model=config["benchmark_model"]
    model=get_model(config).to(device)
    loss_fun=get_loss_fun(config)
    print(config["save_name"])
    if benchmark_model:
        dic=compute_time_no_loader(model,warmup_iter,num_iter,device,crop_size,val_input_size,batch_size,num_classes,mixed_precision,loss_fun)
        for k,v in dic.items():
            print(f"{k}: {v}")
    if benchmark_loader:
        train_loader, val_loader,train_set=get_dataset_loaders(config)
        train_loader_time=compute_loader_time(train_loader,warmup_iter,num_iter)
        val_loader_time=compute_loader_time(val_loader,warmup_iter,num_iter)
        print("train loader time:",train_loader_time)
        print("val loader time:",val_loader_time)

def benchmark_main():
    config_filename="configs/hit_uav_500epochs.yaml"
    with open(config_filename) as file:
        config=yaml.full_load(file)
    #config["dataset_dir"]="cityscapes_dataset"
    config["class_uniform_pct"]=0
    config["benchmark_model"]=True
    config["benchmark_loader"]=True
    benchmark_one(config)

def validate_main():
    config_filename="configs/hit_uav_500epochs.yaml"
    with open(config_filename) as file:
        config=yaml.full_load(file)
    #config["dataset_dir"]="cityscapes_dataset"
    config["class_uniform_pct"]=0 # since we're only evalutaing, not training
    config["pretrained_path"]="checkpoints/hit_uav_exp48_decoder26_trainval_500_epochs_run2"
    confmat=validate_one(config)
    return confmat

def train_main():
    config_filename= "configs/hit_uav_500epochs.yaml"
    with open(config_filename) as file:
        config=yaml.full_load(file)
    #config["dataset_dir"]="cityscapes_dataset"
    train_one(config)

def train_3runs():
    # train the same model 3 times to get error bounds
    config_filename="configs/cityscapes_1000epochs.yaml"
    with open(config_filename) as file:
        config=yaml.full_load(file)
    configs=[]
    for run in range(1,4):
        new_config = copy.deepcopy(config)
        new_config["run"] = run
        new_config["RNG_seed"] = run
        configs.append(new_config)
    train_multiple(configs)

def save_results_main():
    config_filename= "configs/cityscapes_trainval_1000epochs.yaml"
    with open(config_filename) as file:
        config=yaml.full_load(file)
    config["model_name"]="exp53_decoder29"
    config["val_split"]="test"
    config["pretrained_path"]="checkpoints/cityscapes_exp53_decoder29_trainval_1000_epochs_1024_crop_bootstrapped_run1"
    pred_dir="test_submission_dir"
    save_cityscapes_results(config,pred_dir)

if __name__=='__main__':
    benchmark_main()
    train_main()
    validate_main()
