from blocks import *
from competitor_blocks import BiseNetDecoder,SFNetDecoder,FaPNDecoder
from benchmark import benchmark_eval,benchmark_train,benchmark_memory
from ENet_models.ENet import ENet

class Enet_Regseg(nn.Module) :
    def __init__(self, name, num_classes, pretrained="", ablate_decoder=False, change_num_classes=False) :
        super().__init__()

        self.model = ENet(num_classes)

    def forward(self,x):
        input_shape=x.shape[-2:]
        x = self.model(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x
    
class RegSeg(nn.Module):
    # exp48_decoder26 is what we call RegSeg in our paper
    # exp53_decoder29 is a larger version of exp48_decoder26
    # all the other models are for ablation studies
    def __init__(self, name, num_classes, pretrained="", ablate_decoder=False,change_num_classes=False):
        super().__init__()
        self.stem=ConvBnAct(3,32,3,2,1)
        body_name, decoder_name=name.split("_")
        if "exp30" == body_name:
            self.body=RegSegBody(5*[[1,4]]+8*[[1,10]])
        elif "exp43"==body_name:
            self.body=RegSegBody([[1],[1,2],[1,4],[1,6],[1,8],[1,10]]+7*[[1,12]])
        elif "exp46"==body_name:
            self.body=RegSegBody([[1],[1,2],[1,4],[1,6],[1,8]]+8*[[1,10]])
        elif "exp47"==body_name:
            self.body=RegSegBody([[1],[1,2],[1,4],[1,6],[1,8],[1,10],[1,12]]+6*[[1,14]])
        elif "exp48"==body_name:
            self.body=RegSegBody([[1],[1,2]]+4*[[1,4]]+7*[[1,14]])
        elif "exp49"==body_name:
            self.body=RegSegBody([[1],[1,2]]+6*[[1,4]]+5*[[1,6,12,18]])
        elif "exp50"==body_name:
            self.body=RegSegBody([[1],[1,2],[1,4],[1,6],[1,8],[1,10]]+7*[[1,3,6,12]])
        elif "exp51"==body_name:
            self.body=RegSegBody([[1],[1,2],[1,4],[1,6],[1,8],[1,10]]+7*[[1,4,8,12]])
        elif "exp52"==body_name:
            self.body=RegSegBody([[1],[1,2],[1,4]]+10*[[1,6]])
        elif "exp53"==body_name:
            self.body=RegSegBody2([[1],[1,2]]+4*[[1,4]]+7*[[1,14]])
        elif "regnety600mf"==body_name:
            self.body=RegNetY600MF()
        else:
            raise NotImplementedError()
        if "decoder4" ==decoder_name:
            self.decoder=Exp2_Decoder4(num_classes,self.body.channels())
        elif "decoder10" ==decoder_name:
            self.decoder=Exp2_Decoder10(num_classes,self.body.channels())
        elif "decoder12" ==decoder_name:
            self.decoder=Exp2_Decoder12(num_classes,self.body.channels())
        elif "decoder14"==decoder_name:
            self.decoder=Exp2_Decoder14(num_classes,self.body.channels())
        elif "decoder26"==decoder_name:
            self.decoder=Exp2_Decoder26(num_classes,self.body.channels())
        elif "decoder29"==decoder_name:
            self.decoder=Exp2_Decoder29(num_classes,self.body.channels())
        elif "BisenetDecoder"==decoder_name:
            self.decoder=BiseNetDecoder(num_classes,self.body.channels())
        elif "SFNetDecoder"==decoder_name:
            self.decoder=SFNetDecoder(num_classes,self.body.channels())
        elif "FaPNDecoder"==decoder_name:
            self.decoder=FaPNDecoder(num_classes,self.body.channels())
        else:
            raise NotImplementedError()
        print(pretrained, ablate_decoder)
        if pretrained != "" and not ablate_decoder:
            dic = torch.load(pretrained, map_location='cpu')
            if type(dic)==dict and "model" in dic:
                dic=dic['model']
            if change_num_classes:
                current_model=self.state_dict()
                new_state_dict={}
                print("change_num_classes: True")
                for k in current_model:
                    if dic[k].size()==current_model[k].size():
                        new_state_dict[k]=dic[k]
                    else:
                        print(k)
                        new_state_dict[k]=current_model[k]
                self.load_state_dict(new_state_dict,strict=True)
            else:
                self.load_state_dict(dic,strict=True)
    def forward(self,x):
        input_shape=x.shape[-2:]
        x=self.stem(x)
        x=self.body(x)
        x=self.decoder(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

def num_classes_speed_test():
    # We find that the training speed highly correlates with the number of classes
    # while the eval speed does not depend much on the number of classes
    v=[10,20,30,40,50,60,70,80]
    models=[]
    for num_classes in v:
        model=RegSeg("exp48_decoder26",num_classes=num_classes)
        models.append(model)
        benchmark_train([model],8,512,True,num_classes)
    x=torch.randn(1,3,1024,2048)
    benchmark_eval(models,x,True)

def dilation_speed_test():
    group_width=16
    w=256
    x=torch.randn(1,256,64,128)
    ts=[]
    for d in range(1,19):
        model=nn.Conv2d(w,w,3,1,padding=d,dilation=d,groups=w//group_width,bias=False)
        ts.extend(benchmark_eval([model],x,True))
    print(ts)

def block_speed_test():
    print("block speed test")
    model1=DBlock(256,256,[1],16,1,"se")
    model2=DBlock(256,256,[1,1],16,1,"se")
    model3=DBlock(256,256,[1,4],16,1,"se")
    model4=DBlock(256,256,[1,10],16,1,"se")
    x=torch.randn(1,256,64,128) # 1/16 original resolution
    ts=benchmark_eval([model1,model2,model3,model4],x,True)
    print(ts)

def calculate_flops():
    from fvcore.nn import FlopCountAnalysis, flop_count_table,ActivationCountAnalysis
    model1=RegSeg("exp48_decoder26",19).eval()
    from competitors_models.DDRNet_Reimplementation import get_ddrnet_23,get_ddrnet_23slim
    x=torch.randn(1,3,1024,2048)
    model2=get_ddrnet_23().eval()
    for model in [model1,model2]:
        flops = FlopCountAnalysis(model, x)
        print(flop_count_table(flops))

def calculate_params(model):
    #https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/6
    import numpy as np
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    model_parameters = model.parameters()
    params2 = sum([np.prod(p.size()) for p in model_parameters])
    return params,params2

def cityscapes_speed_test():
    print("cityscapes speed test")
    from competitors_models.DDRNet_Reimplementation import get_ddrnet_23
    regseg=RegSeg("exp48_decoder26",19)
    ddrnet23=get_ddrnet_23()
    x=torch.randn(1,3,1024,2048)
    ts=[]
    ts.extend(benchmark_eval([regseg,ddrnet23],x,True))
    print(ts)

def camvid_speed_test():
    print("camvid speed test")
    from competitors_models.DDRNet_Reimplementation import get_ddrnet_23
    regseg=RegSeg("exp48_decoder26",19)
    ddrnet23=get_ddrnet_23()
    x=torch.randn(1,3,720,960)
    ts=[]
    ts.extend(benchmark_eval([regseg,ddrnet23],x,True))
    print(ts)

if __name__ == "__main__":
    cityscapes_speed_test()
