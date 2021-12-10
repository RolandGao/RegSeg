import torch
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, apply_act=True):
        super(ConvBnAct, self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias)
        self.bn=nn.BatchNorm2d(out_channels)
        if apply_act:
            self.act=nn.ReLU(inplace=True)
        else:
            self.act=None
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x=self.act(x)
        return x

class BnActConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=False):
        super(BnActConv, self).__init__()
        self.bn=nn.BatchNorm2d(in_channels)
        self.act=nn.ReLU(inplace=True)
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias)
    def forward(self, x):
        x = self.bn(x)
        x=self.act(x)
        x = self.conv(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        # self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
        #                             BatchNorm2d(inplanes, momentum=bn_mom),
        #                             nn.ReLU(inplace=True),
        #                             nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        #                             )
        self.scale1=nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            BnActConv(inplanes,branch_planes)
        )
        # self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
        #                             BatchNorm2d(inplanes, momentum=bn_mom),
        #                             nn.ReLU(inplace=True),
        #                             nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        #                             )
        self.scale2=nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            BnActConv(inplanes,branch_planes)
        )
        # self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
        #                             BatchNorm2d(inplanes, momentum=bn_mom),
        #                             nn.ReLU(inplace=True),
        #                             nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        #                             )
        self.scale3=nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            BnActConv(inplanes,branch_planes)
        )
        # self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
        #                             BatchNorm2d(inplanes, momentum=bn_mom),
        #                             nn.ReLU(inplace=True),
        #                             nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        #                             )
        self.scale4=nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BnActConv(inplanes,branch_planes)
        )
        # self.scale0 = nn.Sequential(
        #     BatchNorm2d(inplanes, momentum=bn_mom),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        # )
        self.scale0=BnActConv(inplanes,branch_planes)
        # self.process1 = nn.Sequential(
        #     BatchNorm2d(branch_planes, momentum=bn_mom),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        # )
        self.process1=BnActConv(branch_planes,branch_planes,3,padding=1)
        # self.process2 = nn.Sequential(
        #     BatchNorm2d(branch_planes, momentum=bn_mom),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        # )
        self.process2=BnActConv(branch_planes,branch_planes,3,padding=1)
        # self.process3 = nn.Sequential(
        #     BatchNorm2d(branch_planes, momentum=bn_mom),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        # )
        self.process3=BnActConv(branch_planes,branch_planes,3,padding=1)
        # self.process4 = nn.Sequential(
        #     BatchNorm2d(branch_planes, momentum=bn_mom),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        # )
        self.process4=BnActConv(branch_planes,branch_planes,3,padding=1)
        # self.compression = nn.Sequential(
        #     BatchNorm2d(branch_planes * 5, momentum=bn_mom),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        # )
        self.compression=BnActConv(branch_planes * 5,outplanes,1)
        # self.shortcut = nn.Sequential(
        #     BatchNorm2d(inplanes, momentum=bn_mom),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        # )
        self.shortcut=BnActConv(inplanes,outplanes,1)

    def forward(self, x):

        #x = self.downsample(x)
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                                                   size=[height, width],
                                                   mode='bilinear',
                                                   align_corners=False)+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                                                    size=[height, width],
                                                    mode='bilinear',
                                                    align_corners=False)+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                                                   size=[height, width],
                                                   mode='bilinear',
                                                   align_corners=False)+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                                                   size=[height, width],
                                                   mode='bilinear',
                                                   align_corners=False)+x_list[3])))

        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        # self.scale_factor = scale_factor

    def forward(self, x):

        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        #
        # if self.scale_factor is not None:
        #     height = x.shape[-2] * self.scale_factor
        #     width = x.shape[-1] * self.scale_factor
        #     out = F.interpolate(out,
        #                         size=[height, width],
        #                         mode='bilinear')

        return out

class DualResNet(nn.Module):

    def __init__(self, block, layers, num_classes=19, planes=64, spp_planes=128, head_planes=128, augment=False):
        super(DualResNet, self).__init__()

        highres_planes = planes * 2
        self.augment = augment

        # self.conv1 =  nn.Sequential(
        #     nn.Conv2d(3,planes,kernel_size=3, stride=2, padding=1),
        #     BatchNorm2d(planes, momentum=bn_mom),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(planes,planes,kernel_size=3, stride=2, padding=1),
        #     BatchNorm2d(planes, momentum=bn_mom),
        #     nn.ReLU(inplace=True),
        # )
        self.conv1=nn.Sequential(
            ConvBnAct(3,planes,3,2,1),
            ConvBnAct(planes,planes,3,2,1),
        )

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, planes, planes, layers[0])
        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes * 2, planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2)

        # self.compression3 = nn.Sequential(
        #     nn.Conv2d(planes * 4, highres_planes, kernel_size=1, bias=False),
        #     BatchNorm2d(highres_planes, momentum=bn_mom),
        # )
        self.compression3=ConvBnAct(planes*4,highres_planes,apply_act=False)

        # self.compression4 = nn.Sequential(
        #     nn.Conv2d(planes * 8, highres_planes, kernel_size=1, bias=False),
        #     BatchNorm2d(highres_planes, momentum=bn_mom),
        # )
        self.compression4=ConvBnAct(planes * 8, highres_planes,apply_act=False)

        # self.down3 = nn.Sequential(
        #     nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
        #     BatchNorm2d(planes * 4, momentum=bn_mom),
        # )
        self.down3=ConvBnAct(highres_planes, planes * 4,3,2,1,apply_act=False)

        # self.down4 = nn.Sequential(
        #     nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
        #     BatchNorm2d(planes * 4, momentum=bn_mom),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
        #     BatchNorm2d(planes * 8, momentum=bn_mom),
        # )
        self.down4=nn.Sequential(
            ConvBnAct(highres_planes, planes * 4,3,2,1),
            ConvBnAct(planes * 4, planes * 8,3,2,1,apply_act=False)
        )

        self.layer3_ = self._make_layer(block, planes * 2, highres_planes, 2)

        self.layer4_ = self._make_layer(block, highres_planes, highres_planes, 2)

        self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)

        self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)

        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

        if self.augment:
            self.seghead_extra = segmenthead(highres_planes, head_planes, num_classes)

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     nn.Conv2d(inplanes, planes * block.expansion,
            #               kernel_size=1, stride=stride, bias=False),
            #     nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            # )
            downsample=ConvBnAct(inplanes, planes * block.expansion,stride=stride,apply_act=False)

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)


    def forward(self, x):
        original_size=x.shape[-2:]
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        layers = []

        x = self.conv1(x)

        x = self.layer1(x)
        layers.append(x)

        x = self.layer2(self.relu(x))
        layers.append(x)

        x = self.layer3(self.relu(x))
        layers.append(x)
        x_ = self.layer3_(self.relu(layers[1]))

        x = x + self.down3(self.relu(x_))
        x_ = x_ + F.interpolate(
            self.compression3(self.relu(layers[2])),
            size=[height_output, width_output],
            mode='bilinear',align_corners=False)
        if self.augment:
            temp = x_

        x = self.layer4(self.relu(x))
        layers.append(x)
        x_ = self.layer4_(self.relu(x_))

        x = x + self.down4(self.relu(x_))
        x_ = x_ + F.interpolate(
            self.compression4(self.relu(layers[3])),
            size=[height_output, width_output],
            mode='bilinear',align_corners=False)

        x_ = self.layer5_(self.relu(x_))
        x = F.interpolate(
            self.spp(self.layer5(self.relu(x))),
            size=[height_output, width_output],
            mode='bilinear',align_corners=False)

        x_ = self.final_layer(x + x_)

        x_=F.interpolate(x_, size=original_size, mode='bilinear', align_corners=False)
        return x_

def get_ddrnet_23(num_classes=19):
    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, planes=64, spp_planes=128, head_planes=128, augment=False)
    return model

def get_ddrnet_23slim(num_classes=19):
    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, planes=32, spp_planes=128, head_planes=64, augment=False)
    return model

if __name__=="__main__":
    model=get_ddrnet_23slim().eval()
    print(model)
    x=torch.randn(1,3,1024,2048)
    y=model(x)
    print(y.shape)
