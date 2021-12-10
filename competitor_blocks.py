from torch import nn
import torch
from torch.nn import functional as F
from torchvision.ops import DeformConv2d
from blocks import ConvBnAct,SEModule

class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, out_channels=128, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        convs = []
        for size in sizes:
            convs.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(size, size)),
                    ConvBnAct(in_channels,out_channels,apply_act=False)
                )
            )
        self.stages=nn.ModuleList(convs)
        self.bottleneck=ConvBnAct(in_channels+len(sizes)*out_channels,out_channels)
        self.dropout=nn.Dropout2d(0.1)

    def forward(self, x):
        y=[x]
        for stage in self.stages:
            z=stage(x)
            z=F.interpolate(z,size=x.shape[-2:],align_corners=False,mode="bilinear")
            y.append(z)
        x=torch.cat(y,1)
        x = self.bottleneck(x)
        return x
class AlignedModule(nn.Module):
    #SFNet-DFNet
    def __init__(self, inplane, outplane):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.interpolate(h_feature,size=size,mode="bilinear",align_corners=False)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid,align_corners=False)
        return output
class FeatureFusionModule(nn.Module):
    # BiseNet
    def __init__(self, in_chan, out_chan):
        super(FeatureFusionModule, self).__init__()
        self.conv=ConvBnAct(in_chan, out_chan)
        self.se=SEModule(out_chan,out_chan//4)

    def forward(self, x, y):
        x = torch.cat([x,y], dim=1)
        x = self.conv(x)
        x= self.se(x)+x
        return x
class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        self.conv=ConvBnAct(in_chan,out_chan,3,1,1)
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_chan, out_chan, 1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        atten=self.scale(x)
        x=x*atten
        return x
class FeatureSelectionModule(nn.Module):
    # FaPN paper
    def __init__(self, in_chan, out_chan):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_chan, in_chan, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False)

    def forward(self, x):
        x = x*self.conv_atten(x) + x
        x = self.conv(x)
        return x

class DeformConv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super().__init__()
        if isinstance(kernel_size,int):
            kernel_size=(kernel_size,kernel_size)
        offset_out_channels=2*deformable_groups*kernel_size[0]*kernel_size[1]
        mask_out_channels=deformable_groups*kernel_size[0]*kernel_size[1]
        self.deform_conv=DeformConv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,bias=True)
        self.offset_conv=nn.Conv2d(in_channels,offset_out_channels,kernel_size,stride,padding,bias=True)
        self.mask_conv=nn.Conv2d(in_channels,mask_out_channels,kernel_size,stride,padding,bias=True)
        self.offset_conv.weight.data.zero_()
        self.offset_conv.bias.data.zero_()
        self.mask_conv.weight.data.zero_()
        self.mask_conv.bias.data.zero_()
    def forward(self,x):
        x,x2=x
        offset=self.offset_conv(x2)
        mask=self.mask_conv(x2)
        mask = torch.sigmoid(mask)
        x=self.deform_conv(x,offset,mask)
        return x

class FeatureAlign_V2(nn.Module):  # FaPN full version
    def __init__(self, in_nc=128, out_nc=128):
        super(FeatureAlign_V2, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc)
        self.project=ConvBnAct(out_nc * 2, out_nc,apply_act=False)
        self.dcpack_L2 = DeformConv(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat_l, feat_s):
        feat_up = F.interpolate(feat_s, feat_l.shape[-2:], mode='bilinear', align_corners=False)
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        offset = self.project(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up, offset]))
        return feat_align, feat_arm
class BiseNetDecoder(nn.Module):
    def __init__(self, num_classes, channels):
        super(BiseNetDecoder, self).__init__()
        channels8,channels16=channels["8"],channels["16"]
        self.arm16 = AttentionRefinementModule(channels16, 128)
        self.conv_head16 = ConvBnAct(128,128,3,1,1)
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.conv_avg = ConvBnAct(channels16,128)
        self.ffm=FeatureFusionModule(128+channels8,128)
        self.conv=ConvBnAct(128,128,3,1,1)
        self.classifier=nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        x8,x16= x["8"], x["16"]

        avg=self.avg_pool(x16)
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, size=x16.shape[-2:], mode='nearest')

        x16 = self.arm16(x16)
        x16 = x16 + avg_up
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='nearest')
        x16 = self.conv_head16(x16)

        x=self.ffm(x8,x16)
        x=self.conv(x)
        x=self.classifier(x)
        return x
class SFNetDecoder(nn.Module):
    def __init__(self, num_classes, channels, fpn_dim=64, fpn_dsn=False):
        super().__init__()
        channels8,channels16=channels["8"],channels["16"]
        self.head16 = PSPModule(channels16,fpn_dim)
        self.head8=ConvBnAct(channels8,fpn_dim)
        self.fpn_dsn = fpn_dsn
        self.fpn_align=AlignedModule(inplane=fpn_dim, outplane=fpn_dim//2)
        self.conv=ConvBnAct(fpn_dim, fpn_dim, 3, 1, 1)
        self.conv_last = nn.Sequential(
            ConvBnAct(2*fpn_dim,fpn_dim,3,1,1),
            nn.Conv2d(fpn_dim, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x8,x16= x["8"], x["16"]
        x16=self.head16(x16)
        x8=self.head8(x8)
        x16_up = self.fpn_align([x8, x16])
        x8 = x8 + x16_up
        x8=self.conv(x8)
        x16_up=F.interpolate(x16, x8.shape[-2:], mode="bilinear", align_corners=True)
        x8=torch.cat([x8,x16_up],dim=1)
        x = self.conv_last(x8)
        return x

class FaPNDecoder(nn.Module):
    # FaPN paper
    def __init__(self, num_classes, channels):
        super().__init__()
        channels8,channels16=channels["8"],channels["16"]
        self.head16=ConvBnAct(channels16, 128, 1)
        self.align=FeatureAlign_V2(channels8,128)
        self.conv8=ConvBnAct(256,128,1)
        self.classifier=nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        #intput_shape=x.shape[-2:]
        x8, x16= x["8"],x["16"]
        x16=self.head16(x16)
        x8, x16_up=self.align(x8,x16)
        x8=torch.cat([x8,x16_up],dim=1)
        x8=self.conv8(x8)
        return self.classifier(x8)
