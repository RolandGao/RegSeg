from torch import nn
import torch
from torch.nn import functional as F

def activation():
    return nn.ReLU(inplace=True)

def norm2d(out_channels):
    return nn.BatchNorm2d(out_channels)

class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, apply_act=True):
        super(ConvBnAct, self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias)
        self.bn=norm2d(out_channels)
        if apply_act:
            self.act=activation()
        else:
            self.act=None
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x=self.act(x)
        return x

class SEModule(nn.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""
    def __init__(self, w_in, w_se):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1=nn.Conv2d(w_in, w_se, 1, bias=True)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(w_se, w_in, 1, bias=True)
        self.act2=nn.Sigmoid()

    def forward(self, x):
        y=self.avg_pool(x)
        y=self.act1(self.conv1(y))
        y=self.act2(self.conv2(y))
        return x * y

class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, avg_downsample=False):
        super(Shortcut, self).__init__()
        if avg_downsample and stride != 1:
            self.avg=nn.AvgPool2d(2,2,ceil_mode=True)
            self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn=nn.BatchNorm2d(out_channels)
        else:
            self.avg=None
            self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.bn=nn.BatchNorm2d(out_channels)
    def forward(self, x):
        if self.avg is not None:
            x=self.avg(x)
        x = self.conv(x)
        x = self.bn(x)
        return x

class DilatedConv(nn.Module):
    def __init__(self,w,dilations,group_width,stride,bias):
        super().__init__()
        num_splits=len(dilations)
        assert(w%num_splits==0)
        temp=w//num_splits
        assert(temp%group_width==0)
        groups=temp//group_width
        convs=[]
        for d in dilations:
            convs.append(nn.Conv2d(temp,temp,3,padding=d,dilation=d,stride=stride,bias=bias,groups=groups))
        self.convs=nn.ModuleList(convs)
        self.num_splits=num_splits
    def forward(self,x):
        x=torch.tensor_split(x,self.num_splits,dim=1)
        res=[]
        for i in range(self.num_splits):
            res.append(self.convs[i](x[i]))
        return torch.cat(res,dim=1)

class ConvBnActConv(nn.Module):
    def __init__(self,w,stride,dilation,groups,bias):
        super().__init__()
        self.conv=ConvBnAct(w,w,3,stride,dilation,dilation,groups)
        self.project=nn.Conv2d(w,w,1,bias=bias)
    def forward(self,x):
        x=self.conv(x)
        x=self.project(x)
        return x


class YBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation,group_width, stride):
        super(YBlock, self).__init__()
        groups = out_channels // group_width
        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)
        self.bn1=norm2d(out_channels)
        self.act1=activation()
        self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        self.bn2=norm2d(out_channels)
        self.act2=activation()
        self.conv3=nn.Conv2d(out_channels, out_channels,kernel_size=1,bias=False)
        self.bn3=norm2d(out_channels)
        self.act3=activation()
        self.se=SEModule(out_channels,in_channels//4)
        if stride != 1 or in_channels != out_channels:
            self.shortcut=Shortcut(in_channels,out_channels,stride)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut=self.shortcut(x) if self.shortcut else x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.act1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        x=self.se(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x = self.act3(x + shortcut)
        return x

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations,group_width, stride,attention="se"):
        super().__init__()
        avg_downsample=True
        groups=out_channels//group_width
        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)
        self.bn1=norm2d(out_channels)
        self.act1=activation()
        if len(dilations)==1:
            dilation=dilations[0]
            self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        else:
            self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=False)
        self.bn2=norm2d(out_channels)
        self.act2=activation()
        self.conv3=nn.Conv2d(out_channels, out_channels,kernel_size=1,bias=False)
        self.bn3=norm2d(out_channels)
        self.act3=activation()
        if attention=="se":
            self.se=SEModule(out_channels,in_channels//4)
        elif attention=="se2":
            self.se=SEModule(out_channels,out_channels//4)
        else:
            self.se=None
        if stride != 1 or in_channels != out_channels:
            self.shortcut=Shortcut(in_channels,out_channels,stride,avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut=self.shortcut(x) if self.shortcut else x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.act1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        if self.se is not None:
            x=self.se(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x = self.act3(x + shortcut)
        return x

class Exp2_LRASPP(nn.Module):
    # LRASPP
    def __init__(self, num_classes,channels,inter_channels=128):
        super().__init__()
        channels8,channels16=channels["8"],channels["16"]
        self.cbr=ConvBnAct(channels16,inter_channels,1)
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels16, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(channels8, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, x):
        #intput_shape=x.shape[-2:]
        x8, x16=x["8"],x["16"]
        x = self.cbr(x16)
        s = self.scale(x16)
        x = x * s
        x = F.interpolate(x, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x= self.low_classifier(x8) + self.high_classifier(x)
        return x
class Exp2_Decoder4(nn.Module):
    def __init__(self, num_classes,channels):
        super().__init__()
        channels8,channels16=channels["8"],channels["16"]
        self.head8=ConvBnAct(channels8,32,1)
        self.head16=ConvBnAct(channels16,128,1)
        self.conv=ConvBnAct(128+32,128,3,1,1)
        self.classifier=nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        #intput_shape=x.shape[-2:]
        x8, x16=x["8"],x["16"]
        x16=self.head16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8=self.head8(x8)
        x=torch.cat((x8, x16), dim=1)
        x=self.conv(x)
        x=self.classifier(x)
        return x

class Exp2_Decoder10(nn.Module):
    def __init__(self, num_classes,channels):
        super().__init__()
        channels8,channels16=channels["8"],channels["16"]
        self.head8=ConvBnAct(channels8,32,1)
        self.head16=ConvBnAct(channels16,128,1)
        self.conv=DBlock(128+32,128,[1],16,1,"se")
        self.classifier=nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        #intput_shape=x.shape[-2:]
        x8, x16=x["8"],x["16"]
        x16=self.head16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8=self.head8(x8)
        x=torch.cat((x8, x16), dim=1)
        x=self.conv(x)
        x=self.classifier(x)
        return x

class Exp2_Decoder12(nn.Module):
    def __init__(self, num_classes, channels):
        super().__init__()
        channels8,channels16=channels["8"],channels["16" ]
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.conv=ConvBnAct(128,128,1)
        self.classifier=nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        #intput_shape=x.shape[-2:]
        x8, x16=x["8"],x["16"]
        x16=self.head16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8=self.head8(x8)
        x= x8 + x16
        x=self.conv(x)
        x=self.classifier(x)
        return x

class Exp2_Decoder14(nn.Module):
    def __init__(self, num_classes, channels):
        super().__init__()
        channels8,channels16=channels["8"],channels["16"]
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.conv=ConvBnAct(128,128,3,1,1)
        self.classifier=nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        #intput_shape=x.shape[-2:]
        x8, x16=x["8"],x["16"]
        x16=self.head16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8=self.head8(x8)
        x= x8 + x16
        x=self.conv(x)
        x=self.classifier(x)
        return x

class Exp2_Decoder26(nn.Module):
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16=channels["4"],channels["8"],channels["16"]
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=ConvBnAct(64+8,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x4, x8, x16=x["4"], x["8"],x["16"]
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4

class Exp2_Decoder29(nn.Module):
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16=channels["4"],channels["8"],channels["16"]
        self.head16=ConvBnAct(channels16, 256, 1)
        self.head8=ConvBnAct(channels8, 256, 1)
        self.head4=ConvBnAct(channels4, 16, 1)
        self.conv8=ConvBnAct(256,128,3,1,1)
        self.conv4=ConvBnAct(128+16,128,3,1,1)
        self.classifier=nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        x4, x8, x16=x["4"], x["8"],x["16"]
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4

def generate_stage(num,block_fun):
    blocks=[]
    for _ in range(num):
        blocks.append(block_fun())
    return blocks
class RegNetY600MF(nn.Module):
    def __init__(self):
        super().__init__()
        group_width=16
        self.stage4=YBlock(32,48,1,group_width,2)
        self.stage8=nn.Sequential(
            YBlock(48, 112, 1, group_width, 2),
            YBlock(112, 112, 1, group_width, 1),
            YBlock(112, 112, 1, group_width, 1)
        )
        self.stage16=nn.Sequential(
            YBlock(112, 256, 1, group_width, 2),
            *generate_stage(6, lambda : YBlock(256,256, 1, group_width, 1))
        )
        self.stage32=nn.Sequential(
            YBlock(256, 608, 1, group_width, 1),
            *generate_stage(3, lambda : YBlock(608,608, 2, group_width, 1))
        )
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x16=self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":112,"16":608}

def generate_stage2(ds,block_fun):
    blocks=[]
    for d in ds:
        blocks.append(block_fun(d))
    return blocks
class RegSegBody(nn.Module):
    def __init__(self,ds):
        super().__init__()
        gw=16
        attention="se"
        self.stage4=DBlock(32, 48, [1], gw, 2, attention)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2, attention),
            DBlock(128, 128, [1], gw, 1, attention),
            DBlock(128, 128, [1], gw, 1, attention)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2, attention),
            *generate_stage2(ds[:-1], lambda d: DBlock(256, 256, d, gw, 1, attention)),
            DBlock(256, 320, ds[-1], gw, 1, attention)
        )
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":128,"16":320}

class RegSegBody2(nn.Module):
    def __init__(self,ds):
        super().__init__()
        gw=24
        attention="se"
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2, attention),
            DBlock(48, 48, [1], gw, 1, attention),
        )
        self.stage8=nn.Sequential(
            DBlock(48, 120, [1], gw, 2, attention),
            *generate_stage(5,lambda: DBlock(120, 120, [1], gw, 1, attention)),
        )
        self.stage16=nn.Sequential(
            DBlock(120, 336, [1], gw, 2, attention),
            *generate_stage2(ds[:-1], lambda d: DBlock(336, 336, d, gw, 1, attention)),
            DBlock(336, 384, ds[-1], gw, 1, attention)
        )
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":120,"16":384}
