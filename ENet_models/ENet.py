##################################################################
# Reproducing the paper                                          #
# ENet - Real Time Semantic Segmentation                         #
# Paper: https://arxiv.org/pdf/1606.02147.pdf                    #
#                                                                #
# Copyright (c) 2019                                             #
# Authors: @iArunava <iarunavaofficial@gmail.com>                #
#          @AvivSham <mista2311@gmail.com>                       #
#                                                                #
# License: BSD License 3.0                                       #
#                                                                #
# The Code in this file is distributed for free                  #
# usage and modification with proper credits                     #
# directing back to this repository.                             #
##################################################################

import torch
import torch.nn as nn
from InitialBlock import InitialBlock, InitialBlock_LowRes
from RDDNeck import RDDNeck, RDDNeck_LowRes
from UBNeck import UBNeck, UBNeck_LowRes
from ASNeck import ASNeck, ASNeck_LowRes

class ENet(nn.Module):
    def __init__(self, C):
        super().__init__()
        
        # Define class variables
        self.C = C
        
        # The initial block
        self.init = InitialBlock()
        
        
        # The first bottleneck
        self.b10 = RDDNeck(dilation=1, 
                           in_channels=16, 
                           out_channels=64, 
                           down_flag=True, 
                           p=0.01)
        
        self.b11 = RDDNeck(dilation=1, 
                           in_channels=64, 
                           out_channels=64, 
                           down_flag=False, 
                           p=0.01)
        
        self.b12 = RDDNeck(dilation=1, 
                           in_channels=64, 
                           out_channels=64, 
                           down_flag=False, 
                           p=0.01)
        
        self.b13 = RDDNeck(dilation=1, 
                           in_channels=64, 
                           out_channels=64, 
                           down_flag=False, 
                           p=0.01)
        
        self.b14 = RDDNeck(dilation=1, 
                           in_channels=64, 
                           out_channels=64, 
                           down_flag=False, 
                           p=0.01)
        
        
        # The second bottleneck
        self.b20 = RDDNeck(dilation=1, 
                           in_channels=64, 
                           out_channels=128, 
                           down_flag=True)
        
        self.b21 = RDDNeck(dilation=1, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False)
        
        self.b22 = RDDNeck(dilation=2, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False)
        
        self.b23 = ASNeck(in_channels=128, 
                          out_channels=128)
        
        self.b24 = RDDNeck(dilation=4, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False)
        
        self.b25 = RDDNeck(dilation=1, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False)
        
        self.b26 = RDDNeck(dilation=8, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False)
        
        self.b27 = ASNeck(in_channels=128, 
                          out_channels=128)
        
        self.b28 = RDDNeck(dilation=16, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False)
        
        
        # The third bottleneck
        self.b31 = RDDNeck(dilation=1, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False)
        
        self.b32 = RDDNeck(dilation=2, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False)
        
        self.b33 = ASNeck(in_channels=128, 
                          out_channels=128)
        
        self.b34 = RDDNeck(dilation=4, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False)
        
        self.b35 = RDDNeck(dilation=1, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False)
        
        self.b36 = RDDNeck(dilation=8, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False)
        
        self.b37 = ASNeck(in_channels=128, 
                          out_channels=128)
        
        self.b38 = RDDNeck(dilation=16, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False)
        
        
        # The fourth bottleneck
        self.b40 = UBNeck(in_channels=128, 
                          out_channels=64, 
                          relu=True)
        
        self.b41 = RDDNeck(dilation=1, 
                           in_channels=64, 
                           out_channels=64, 
                           down_flag=False, 
                           relu=True)
        
        self.b42 = RDDNeck(dilation=1, 
                           in_channels=64, 
                           out_channels=64, 
                           down_flag=False, 
                           relu=True)
        
        
        # The fifth bottleneck
        self.b50 = UBNeck(in_channels=64, 
                          out_channels=16, 
                          relu=True)
        
        self.b51 = RDDNeck(dilation=1, 
                           in_channels=16, 
                           out_channels=16, 
                           down_flag=False, 
                           relu=True)
        
        
        # Final ConvTranspose Layer
        self.fullconv = nn.ConvTranspose2d(in_channels=16, 
                                           out_channels=self.C, 
                                           kernel_size=3, 
                                           stride=2, 
                                           padding=1, 
                                           output_padding=1,
                                           bias=False)
        
        
    def forward(self, x):
        
        # The initial block
        x = self.init(x)
        
        # The first bottleneck
        x, i1 = self.b10(x)
        x = self.b11(x)
        x = self.b12(x)
        x = self.b13(x)
        x = self.b14(x)
        
        # The second bottleneck
        x, i2 = self.b20(x)
        x = self.b21(x)
        x = self.b22(x)
        x = self.b23(x)
        x = self.b24(x)
        x = self.b25(x)
        x = self.b26(x)
        x = self.b27(x)
        x = self.b28(x)
        
        # The third bottleneck
        x = self.b31(x)
        x = self.b32(x)
        x = self.b33(x)
        x = self.b34(x)
        x = self.b35(x)
        x = self.b36(x)
        x = self.b37(x)
        x = self.b38(x)
        
        # The fourth bottleneck
        x = self.b40(x, i2)
        x = self.b41(x)
        x = self.b42(x)
        
        # The fifth bottleneck
        x = self.b50(x, i1)
        x = self.b51(x)
        
        # Final ConvTranspose Layer
        x = self.fullconv(x)
        
        return x

class Enet_LowRes(nn.Module) : 
    def __init__(self, C, device):
        super().__init__()
        
        # Define class variables
        self.C = C
        
        # The initial block
        self.init = InitialBlock()
        
        
        # The first bottleneck
        self.b10 = RDDNeck_LowRes(dilation=1, 
                           in_channels=16, 
                           out_channels=64, 
                           down_flag=False, 
                           p=0.01, device=device)
        
        self.b11 = RDDNeck_LowRes(dilation=1, 
                           in_channels=64, 
                           out_channels=64, 
                           down_flag=False, 
                           p=0.01, device=device)
        
        self.b12 = RDDNeck_LowRes(dilation=1, 
                           in_channels=64, 
                           out_channels=64, 
                           down_flag=False, 
                           p=0.01, device=device)
        
        self.b13 = RDDNeck_LowRes(dilation=1, 
                           in_channels=64, 
                           out_channels=64, 
                           down_flag=False, 
                           p=0.01, device=device)
        
        self.b14 = RDDNeck_LowRes(dilation=1, 
                           in_channels=64, 
                           out_channels=64, 
                           down_flag=False, 
                           p=0.01, device=device)
        
        
        # The second bottleneck
        self.b20 = RDDNeck_LowRes(dilation=1, 
                           in_channels=64, 
                           out_channels=128, 
                           down_flag=False, device=device)
        
        self.b21 = RDDNeck_LowRes(dilation=1, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False, device=device)
        
        self.b22 = RDDNeck_LowRes(dilation=2, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False, device=device)
        
        self.b23 = ASNeck_LowRes(in_channels=128, 
                          out_channels=128, device=device)
        
        self.b24 = RDDNeck_LowRes(dilation=4, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False, device=device)
        
        self.b25 = RDDNeck_LowRes(dilation=1, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False, device=device)
        
        self.b26 = RDDNeck_LowRes(dilation=8, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False, device=device)
        
        self.b27 = ASNeck_LowRes(in_channels=128, 
                          out_channels=128, device=device)
        
        self.b28 = RDDNeck_LowRes(dilation=16, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False, device=device)
        
        
        # The third bottleneck
        self.b31 = RDDNeck_LowRes(dilation=1, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False, device=device)
        
        self.b32 = RDDNeck_LowRes(dilation=2, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False, device=device)
        
        self.b33 = ASNeck_LowRes(in_channels=128, 
                          out_channels=128, device=device)
        
        self.b34 = RDDNeck_LowRes(dilation=4, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False, device=device)
        
        self.b35 = RDDNeck_LowRes(dilation=1, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False, device=device)
        
        self.b36 = RDDNeck_LowRes(dilation=8, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False, device=device)
        
        self.b37 = ASNeck_LowRes(in_channels=128, 
                          out_channels=128, device=device)
        
        self.b38 = RDDNeck_LowRes(dilation=16, 
                           in_channels=128, 
                           out_channels=128, 
                           down_flag=False, device=device)
        
        
        # The fourth bottleneck
        self.b40 = UBNeck_LowRes(in_channels=128, 
                          out_channels=64, 
                          relu=True)
        
        self.b41 = RDDNeck_LowRes(dilation=1, 
                           in_channels=64, 
                           out_channels=64, 
                           down_flag=False, 
                           relu=True, device=device)
        
        self.b42 = RDDNeck_LowRes(dilation=1, 
                           in_channels=64, 
                           out_channels=64, 
                           down_flag=False, 
                           relu=True, device=device)
        
        
        # The fifth bottleneck
        self.b50 = UBNeck_LowRes(in_channels=64, 
                          out_channels=16, 
                          relu=True)
        
        self.b51 = RDDNeck_LowRes(dilation=1, 
                           in_channels=16, 
                           out_channels=16, 
                           down_flag=False, 
                           relu=True, device=device)
        
        
        # Final ConvTranspose Layer
        self.fullconv = nn.ConvTranspose2d(in_channels=16, 
                                           out_channels=self.C, 
                                           kernel_size=3, 
                                           stride=2, 
                                           padding=1, 
                                           output_padding=1,
                                           bias=False)
        
        
    def forward(self, x):
        
        # The initial block
        x = self.init(x)
        
        # The first bottleneck
        x = self.b10(x)
        x = self.b11(x)
        x = self.b12(x)
        x = self.b13(x)
        x = self.b14(x)
        
        # The second bottleneck
        x = self.b20(x)
        x = self.b21(x)
        x = self.b22(x)
        x = self.b23(x)
        x = self.b24(x)
        x = self.b25(x)
        x = self.b26(x)
        x = self.b27(x)
        x = self.b28(x)
        
        # The third bottleneck
        x = self.b31(x)
        x = self.b32(x)
        x = self.b33(x)
        x = self.b34(x)
        x = self.b35(x)
        x = self.b36(x)
        x = self.b37(x)
        x = self.b38(x)
        
        # The fourth bottleneck
        x = self.b40(x)
        x = self.b41(x)
        x = self.b42(x)
        
        # The fifth bottleneck
        x = self.b50(x)
        x = self.b51(x)
        
        # Final ConvTranspose Layer
        x = self.fullconv(x)
        
        return x

# Test function (#TODO : remove)
if __name__ == "__main__" :
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    model = Enet_LowRes(2, device)
    print(device, device == torch.device('cuda'))
    input = torch.ones((1, 3, 80, 80))
    input = input.to(device)
    model = model.to(device)
    output = model(input)
    print(input.size())
    print(output.size())