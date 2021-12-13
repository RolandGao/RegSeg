import json
import os
import numpy as np

def calc_miou(ious):
    best_miou=0
    for iou in ious:
        miou=sum(iou)/len(iou)
        if miou > best_miou:
            best_miou=miou
    return best_miou
def calc_miou_last(ious):
    iou=ious[-1]
    miou=sum(iou)/len(iou)
    return miou
def calc_miou_reduced(ious):
    best_miou=0
    exclude_indices=[14,15,16]
    for iou in ious:
        iou=[iou[i] for i in range(len(iou)) if i not in exclude_indices]
        miou=sum(iou)/len(iou)
        if miou > best_miou:
            best_miou=miou
    return best_miou
def calc_miou_reduced_last(ious):
    exclude_indices=[14,15,16]
    iou=ious[-1]
    iou=[iou[i] for i in range(len(iou)) if i not in exclude_indices]
    miou=sum(iou)/len(iou)
    return miou

def print_dict(dic):
    for k,v in dic.items():
        if "trainval" in k:
            continue
        print(k)
        print(v)

def f(log_dir):
    files=[os.path.join(log_dir,filename) for filename in os.listdir(log_dir) if ".txt" in filename]
    files=sorted(files)
    dic={}
    for filename in files:
        ious=extract_ious(filename)
        best_miou=calc_miou(ious)
        best_miou_reduced=calc_miou_reduced(ious)
        name=filename.split("/")[-1]
        dic[name]=(best_miou, best_miou_reduced)
    miou_dic={k: v[0] for k, v in sorted(dic.items(), key=lambda item: item[1][0], reverse=True)}
    miou_reduced_dic={k: v[1] for k, v in sorted(dic.items(), key=lambda item: item[1][1], reverse=True)}
    print("miou reduced")
    print_dict(miou_reduced_dic)
    # print("miou original")
    # print_dict(miou_dic)

def g(log_dir):
    files=[os.path.join(log_dir,filename) for filename in os.listdir(log_dir) if ".txt" in filename]
    files=sorted(files)
    dic={}
    for filename in files:
        ious=extract_ious(filename)
        best_miou=calc_miou_last(ious)
        best_miou_reduced=calc_miou_reduced_last(ious)
        name=filename.split("/")[-1]
        dic[name]=(best_miou, best_miou_reduced)
    miou_dic={k: v[0] for k, v in sorted(dic.items(), key=lambda item: item[1][0], reverse=True)}
    miou_reduced_dic={k: v[1] for k, v in sorted(dic.items(), key=lambda item: item[1][1], reverse=True)}
    print("miou reduced last")
    print_dict(miou_reduced_dic)
    print("miou original last")
    print_dict(miou_dic)
def z(log_dir):
    files=[os.path.join(log_dir,filename) for filename in os.listdir(log_dir) if ".txt" in filename]
    files=sorted(files)
    dic={}
    for filename in files:
        all_ious=extract_ious_runs(filename)
        for i,ious in enumerate(all_ious):
            best_miou=calc_miou(ious)
            name=filename.split("/")[-1]+f"_run{i+1}"
            dic[name]=best_miou
    print("miou original")
    print_dict(dic)
def extract_ious(filename):
    ious=[]
    with open(filename) as f:
        lines=f.readlines()
        lines=[line.strip().split(": ") for line in lines]
        for line in lines:
            if line[0]=="IoU" or line[0]=="IOU":
                line[1]=line[1].replace("'","")
                #print(line[1])
                iou=json.loads(line[1])
                ious.append(iou)
    return ious
def extract_ious_runs(filename):
    all_ious=[]
    with open(filename) as f:
        lines=f.readlines()
        lines=[line.strip().split(": ") for line in lines]
        ious=[]
        for line in lines:
            if line[0]=="IoU" or line[0]=="IOU":
                line[1]=line[1].replace("'","")
                iou=json.loads(line[1])
                ious.append(iou)
            if line[0]=="run":
                if len(ious)!=0:
                    all_ious.append(ious)
                    ious=[]
        if len(ious)!=0:
            all_ious.append(ious)
            ious=[]
    return all_ious

def mean_and_std(v):
    mean=np.mean(v)
    std=np.std(v)
    print(f"{mean} +- {std}")
    return mean,std

def comparison_against_ddrnet23():
    print("comparison against ddrnet23, 3 runs")
    our1=[98.32, 86.0, 92.81, 60.95, 63.19, 66.33, 70.32, 78.31, 92.71, 64.6, 94.98, 81.64, 62.3, 95.06, 73.72, 82.42, 73.79, 62.04, 76.49]
    our2=[98.29, 85.88, 92.86, 60.39, 64.6, 66.41, 71.01, 79.36, 92.75, 66.12, 95.14, 81.42, 62.33, 94.92, 77.1, 85.8, 83.35, 63.17, 76.28]
    our3=[98.25, 85.58, 92.68, 61.48, 61.88, 66.28, 70.21, 79.04, 92.63, 64.28, 95.29, 81.55, 61.94, 94.95, 74.66, 83.61, 76.97, 62.58, 76.3]
    ddrnet23_run1=[97.98, 84.08, 92.66, 54.68, 60.18, 65.26, 71.52, 78.77, 92.6, 62.69, 95.09, 81.7, 61.33, 95.27, 78.18, 84.87, 80.46, 62.44, 76.99]
    ddrnet23_run2=[98.16, 85.01, 92.54, 52.4, 60.31, 65.44, 71.71, 79.33, 92.6, 61.89, 94.99, 81.85, 62.06, 95.21, 76.68, 86.09, 81.4, 58.34, 76.87]
    ddrnet23_run3=[98.14, 85.09, 92.65, 56.79, 61.43, 65.89, 72.26, 78.39, 92.68, 63.95, 95.07, 82.09, 62.71, 95.02, 74.16, 84.36, 76.06, 59.44, 76.93]
    our_mious=[]
    our_mious_reduced=[]
    ddrnet_mious=[]
    ddrnet_mious_reduced=[]
    for ious in [our1,our2,our3]:
        our_mious.append(calc_miou([ious]))
        our_mious_reduced.append(calc_miou_reduced([ious]))

    for ious in [ddrnet23_run1,ddrnet23_run2,ddrnet23_run3]:
        ddrnet_mious.append(calc_miou([ious]))
        ddrnet_mious_reduced.append(calc_miou_reduced([ious]))
    print("our miou")
    mean_and_std(our_mious)
    # print("our miou reduced")
    # mean_and_std(our_mious_reduced)
    print("ddrnet miou")
    mean_and_std(ddrnet_mious)
    # print("ddrnet miou reduced")
    # mean_and_std(ddrnet_mious_reduced)
    print()
def camvid_5runs():
    w=[80.86231578480114,80.9288031838157,81.02897435968572,80.83710479736328,80.86533043601297]
    print("camvid 5 runs")
    mean_and_std(w)
    print()
def camvid_log():
    print("camvid")
    z("training_log/camvid")
    print()
def backbone_ablation_studies():
    print("backbone ablation studies")
    f("training_log/backbone_ablation_studies")
    print()
def decoder_ablation_studies():
    print("decoder ablation studies")
    f("training_log/decoder_ablation_studies")
    print()
def comparison_against_ddrnet23_log():
    print("comparison against ddrnet23")
    z("training_log/comparison_against_ddrnet23")
    print()
def reproducibility():
    print("reproducibility")
    g("training_log/reproducibility")
    print()
def random_resizes_random_crops():
    print("random_resizes_random_crops")
    f("training_log/random_resizes_random_crops")
    print()
def training_techniques():
    print("training_techniques")
    f("training_log/training_techniques")
    print()
if __name__=="__main__":
    reproducibility()
    backbone_ablation_studies()
    decoder_ablation_studies()
    random_resizes_random_crops()
    training_techniques()
    comparison_against_ddrnet23()
    comparison_against_ddrnet23_log()
    camvid_5runs()
    camvid_log()
