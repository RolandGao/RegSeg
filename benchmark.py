# adapted from https://github.com/facebookresearch/pycls/blob/f8cd962737e33ce9e19b3083a33551da95c2d9c0/pycls/core/benchmark.py

import torch
import time
import torch.cuda.amp as amp
import torch.nn.functional
import torch.nn as nn

@torch.no_grad()
def compute_eval_time(model,device,warmup_iter,num_iter,val_input_size,mixed_precision):
    model.eval()
    if isinstance(val_input_size,int):
        h,w=val_input_size,val_input_size*2
    else:
        h,w=val_input_size
    x=torch.randn(1,3,h,w).to(device)
    times=[]
    for cur_iter in range(warmup_iter+num_iter):
        if cur_iter == warmup_iter:
            times.clear()
        t1=time.time()
        with amp.autocast(enabled=mixed_precision):
            output = model(x)
        torch.cuda.synchronize()
        t2=time.time()
        times.append(t2-t1)
    return average(times)

@torch.no_grad()
def compute_eval_time2(model,x,warmup_iter,num_iter,mixed_precision):
    model.eval()
    times=[]
    for cur_iter in range(warmup_iter+num_iter):
        if cur_iter == warmup_iter:
            times.clear()
        t1=time.time()
        with amp.autocast(enabled=mixed_precision):
            output = model(x)
        torch.cuda.synchronize()
        t2=time.time()
        times.append(t2-t1)
    return average(times)


def compute_train_time2(model,x,target,warmup_iter,num_iter,mixed_precision):
    model.train()
    times=[]
    for cur_iter in range(warmup_iter+num_iter):
        if cur_iter == warmup_iter:
            times.clear()
        t1=time.time()
        with amp.autocast(enabled=mixed_precision):
            output = model(x,target)
        torch.cuda.synchronize()
        t2=time.time()
        times.append(t2-t1)
    return average(times)

def average(v):
    return sum(v)/len(v)
def compute_train_time(model,warmup_iter,num_iter,train_crop_size,batch_size,num_classes,mixed_precision, loss_fun):
    model.train()
    if isinstance(train_crop_size,int):
        crop_h,crop_w=train_crop_size,train_crop_size
    else:
        crop_h,crop_w=train_crop_size
    x=torch.randn(batch_size, 3, crop_h,crop_w).cuda(non_blocking=False)
    target=torch.randint(0,num_classes,(batch_size, crop_h,crop_w)).cuda(non_blocking=False)
    fw_times=[]
    bw_times=[]
    scaler = amp.GradScaler(enabled=mixed_precision)
    for cur_iter in range(warmup_iter+num_iter):
        if cur_iter == warmup_iter:
            fw_times.clear()
            bw_times.clear()
        t1=time.time()
        with amp.autocast(enabled=mixed_precision):
            output = model(x)
            loss = loss_fun(output,target)
        torch.cuda.synchronize()
        t2=time.time()
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        t3=time.time()
        fw_times.append(t2-t1)
        bw_times.append(t3-t2)
    return average(fw_times),average(bw_times)

def compute_loader_time(data_loader,warmup_iter,num_iter):
    times=[]
    data_loader_iter=iter(data_loader)

    for cur_iter in range(min(warmup_iter+num_iter,len(data_loader))):
        if cur_iter == warmup_iter:
            times.clear()
        t1=time.time()
        next(data_loader_iter)
        t2=time.time()
        times.append(t2-t1)
    return average(times)


def memory_used(device):
    x=torch.cuda.memory_allocated(device)
    return round(x/1024/1024)
def max_memory_used(device):
    x=torch.cuda.max_memory_allocated(device)
    return round(x/1024/1024)
def memory_test_helper(model,device,train_crop_size,batch_size,num_classes,mixed_precision,loss_fun):
    if isinstance(train_crop_size,int):
        crop_h,crop_w=train_crop_size,train_crop_size
    else:
        crop_h,crop_w=train_crop_size
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    scaler = amp.GradScaler(enabled=mixed_precision)
    x=torch.randn(batch_size, 3, crop_h,crop_w).to(device)
    target=torch.randint(0,num_classes,(batch_size,crop_h,crop_w)).to(device)
    t1=memory_used(device)
    with amp.autocast(enabled=mixed_precision):
        output = model(x)
        loss = loss_fun(output,target)
    scaler.scale(loss).backward()
    torch.cuda.synchronize()
    t2=max_memory_used(device)
    return t2-t1

def compute_memory_usage(model,device,crop_size,batch_size,num_classes,mixed_precision, loss_fun):
    for p in model.parameters():
        p.grad=None
    try:
        t=memory_test_helper(model,device,crop_size,batch_size,num_classes,mixed_precision,loss_fun)
        t=memory_test_helper(model,device,crop_size,batch_size,num_classes,mixed_precision,loss_fun)
    except:
        t=-1
        print("out of memory")
    for p in model.parameters():
        p.grad=None
    return t

def compute_time_no_loader(model,warmup_iter,num_iter,device,crop_size,val_input_size,batch_size,num_classes,mixed_precision,loss_fun):
    model=model.to(device)
    print("benchmarking eval time")
    eval_time=compute_eval_time(model,device,warmup_iter,num_iter,val_input_size,mixed_precision)
    print("benchmarking train time")
    train_fw_time,train_bw_time=compute_train_time(model,warmup_iter,num_iter,crop_size,batch_size,num_classes,mixed_precision,loss_fun)
    train_time=train_fw_time+train_bw_time
    print("benchmarking memory usage")
    memory_usage=compute_memory_usage(model,device,crop_size,batch_size,num_classes,mixed_precision,loss_fun)
    dic1={
        "eval_time":eval_time,
        "train_time":train_time,
        "memory_usage":memory_usage
    }
    return dic1

def compute_time_full(model,data_loader,warmup_iter,num_iter,device,crop_size,val_input_size,batch_size,num_classes,mixed_precision,loss_fun):
    model=model.to(device)
    print("benchmarking eval time")
    eval_time=compute_eval_time(model,device,warmup_iter,num_iter,val_input_size,mixed_precision)
    print("benchmarking train time")
    train_fw_time,train_bw_time=compute_train_time(model,warmup_iter,num_iter,crop_size,batch_size,num_classes,mixed_precision,loss_fun)
    train_time=train_fw_time+train_bw_time
    print("benchmarking memory usage")
    memory_usage=compute_memory_usage(model,device,crop_size,batch_size,num_classes,mixed_precision,loss_fun)
    print("benchmarking loader time")
    loader_time=compute_loader_time(data_loader,warmup_iter,num_iter)
    loader_overhead=max(0,loader_time-train_time)/train_time
    dic1={
        "eval_time":eval_time,
        "train_time":train_time,
        "memory_usage":memory_usage,
        "loader_time":loader_time,
        "loader_overhead":loader_overhead
    }
    dic2={
        "eval_time":eval_time*len(data_loader),
        "train_time":train_time*len(data_loader),
        "memory_usage":memory_usage,
        "loader_time":loader_time,
        "loader_overhead":loader_overhead
    }
    return dic1


def benchmark_eval(models,x,mixed_precision):
    torch.backends.cudnn.benchmark=True
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    x=x.to(device)
    ts=[]
    for model in models:
        model=model.to(device)
        t=compute_eval_time2(model,x,10,100,mixed_precision)
        model.cpu()
        print(t)
        ts.append(t)
    return ts
def benchmark_train(models,batch_size,crop_size,mixed_precision,num_classes=19):
    torch.backends.cudnn.benchmark=True
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loss_fun=nn.CrossEntropyLoss(weight=None,ignore_index=255)
    ts=[]
    for model in models:
        model=model.to(device)
        fw,bw=compute_train_time(model,1,3,crop_size,batch_size,num_classes,mixed_precision,loss_fun)
        model.cpu()
        print(fw+bw)
        ts.append(fw+bw)
    return ts
def benchmark_memory(models,batch_size,crop_size,mixed_precision,num_classes=19):
    torch.backends.cudnn.benchmark=True
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loss_fun=nn.CrossEntropyLoss(ignore_index=255)
    for model in models:
        model=model.to(device)
        memory_usage=compute_memory_usage(model,device,crop_size,batch_size,num_classes,mixed_precision,loss_fun)
        print(memory_usage)
