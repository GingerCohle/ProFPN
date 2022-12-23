import warnings
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from mmcv.cnn import ConvModule, constant_init, kaiming_init, xavier_init
from mmdet.core import auto_fp16
import math
from ..builder import NECKS
BN_MOMENTUM = 0.1

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class CropLayer(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        if self.rows_to_crop == 0 and self.cols_to_crop == 0:
            return input
        elif self.rows_to_crop > 0 and self.cols_to_crop == 0:
            return input[:, :, self.rows_to_crop:-self.rows_to_crop, :]
        elif self.rows_to_crop == 0 and self.cols_to_crop > 0:
            return input[:, :, :, self.cols_to_crop:-self.cols_to_crop]
        else:
            return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]

class GCA(nn.Module):#gca block
    def __init__(self, in_channels, out_channels):
        super(GCA, self).__init__()
        self.spatial = nn.Sequential(nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1, groups=1,dilation=1, bias=False),#conv1x1(in_channels, in_channels // 2, 1),
                                     nn.ReLU(),
                                     conv1x1(in_channels // 2, 1,1),
                                     )
        self.channel = nn.Sequential(conv1x1(out_channels, out_channels, 1),
                                     nn.ReLU(),
                                     )
    def forward(self, x,y):
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        att_map = F.softmax(self.spatial(x).view(b, 1, 1,-1), dim=-1)
        x_1 = y.view(b,c*2,-1,1)
        return self.channel(torch.matmul(att_map,x_1))

class Asymmetric_DConv(nn.Module):#asymmetric dilated convolution
    def __init__(self, in_c, out_c, padding, kernel_size, stride, dilations, padding_mode='zeros', use_affine=True):
        super(Asymmetric_DConv, self).__init__()
        center_offset_from_origin_border = padding - kernel_size // 2
        ver_pad_or_crop = (0, center_offset_from_origin_border)
        hor_pad_or_crop = (center_offset_from_origin_border, 0)
        if center_offset_from_origin_border >= 0:
            self.ver_conv_crop_layer = nn.Identity()
            ver_conv_padding = ver_pad_or_crop
            self.hor_conv_crop_layer = nn.Identity()
            hor_conv_padding = hor_pad_or_crop
        else:
            self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
            ver_conv_padding = (0, 0)
            self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
            hor_conv_padding = (0, 0)
        self.ver_conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(kernel_size, 1),
                                  stride=stride,
                                  padding=(dilations, 0), dilation=(dilations, 1), groups=1, bias=False,
                                  padding_mode=padding_mode)

        self.hor_conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(1, kernel_size),
                                  stride=stride,
                                  padding=(0, dilations), dilation=(1, dilations), groups=1, bias=False,
                                  padding_mode=padding_mode)
        self.ver_bn = nn.BatchNorm2d(num_features=out_c, affine=use_affine)
        self.hor_bn = nn.BatchNorm2d(num_features=out_c, affine=use_affine)

    def forward(self, x):
        vertical_outputs = self.ver_conv_crop_layer(x)
        vertical_outputs = self.ver_conv(vertical_outputs)
        vertical_outputs = self.ver_bn(vertical_outputs)
        horizontal_outputs = self.hor_conv_crop_layer(x)
        horizontal_outputs = self.hor_conv(horizontal_outputs)
        horizontal_outputs = self.hor_bn(horizontal_outputs)
        return horizontal_outputs + vertical_outputs



class MBAD(nn.Module):#multi-branch asymmetric dilated block
    def __init__(self, in_c, out_c,kernel_size=3):
        super(MBAD, self).__init__()
        self.stage1_conv = nn.Sequential(
            nn.Conv2d(in_c, out_c // 2, kernel_size=3, stride=1, padding=1, groups=1,dilation=1, bias=False),
            nn.BatchNorm2d(out_c // 2),
            nn.ReLU(True))
        self.stage2_drate_1 = Asymmetric_DConv(in_c=out_c // 2, out_c=out_c // 8, padding=1, stride=1, kernel_size=kernel_size, dilations=1)
        self.stage2_drate_2 = Asymmetric_DConv(in_c=out_c // 2, out_c=out_c // 8, padding=2, stride=1, kernel_size=kernel_size, dilations=2)
        self.stage2_drate_4 = Asymmetric_DConv(in_c=out_c // 2, out_c=out_c // 8, padding=4, stride=1, kernel_size=kernel_size, dilations=4)
        self.stage2_drate_8 = Asymmetric_DConv(in_c=out_c // 2, out_c=out_c // 8, padding=8, stride=1, kernel_size=kernel_size, dilations=8)
        self.identity_conv = conv1x1(in_c, out_c, stride=1)

    def forward(self, x):
        stage1_out = self.stage1_conv(x)
        stage2_out = torch.cat([self.stage2_drate_1(stage1_out), self.stage2_drate_2(stage1_out), self.stage2_drate_4(stage1_out), self.stage2_drate_8(stage1_out)],dim=1)
        return torch.cat([stage2_out, stage1_out], dim=1) + self.identity_conv(x)

class ProFPN(nn.Module):#progressive feature pyramid
    def __init__(self, upsample_cfg=dict(mode='nearest')):
        super(ProFPN, self).__init__()
        self.upsample_cfg = upsample_cfg.copy()
        #GCA blocks
        self.gca_lv4 = GCA(1024,2048)
        self.gca_lv3 = GCA(512,1024)
        self.gca_lv2 = GCA(256,512)

        #MBAD blocks
        #stage2_lv5
        self.stage2_lv5 = nn.ModuleList()
        self.stage2_lv5.append(MBAD(2048, 1024))
        self.upsample_stage2_lv5 = self.up_sampling(1)
        #stage2_lv4
        self.stage2_lv4 = nn.ModuleList()
        self.stage2_lv4.append(MBAD(1024, 512))
        self.upsample_stage2_lv4 = self.up_sampling(1)
        #stage2_lv3
        self.stage2_lv3 = nn.ModuleList()
        self.stage2_lv3.append(MBAD(512, 256))
        self.upsample_stage2_lv3 = self.up_sampling(1)
        #stage3_lv5
        self.stage3_lv5 = nn.ModuleList()
        self.stage3_lv5.append(MBAD(1024, 512))
        self.upsample_stage3_lv5 = self.up_sampling(1)
        #stage3_lv4
        self.stage3_lv4 = nn.ModuleList()
        self.stage3_lv4.append(MBAD(512, 256))
        self.upsample_stage3_lv4 = self.up_sampling(1)
        #stage4_lv5
        self.stage4_lv5 = nn.ModuleList()
        self.stage4_lv5.append(MBAD(512, 256))
        self.upsample_stage4_lv5 = self.up_sampling(1)


    def sum_tensor_list(self, feat_list):
        out = feat_list[0]
        for i in range(1, len(feat_list)):
            out+=feat_list[i]
        return out

    def up_sampling(self,i):
        return nn.Sequential(nn.Upsample(scale_factor=2**i, mode='nearest'))

    def forward(self, feats):
        #Bottom-up Interaction Module
        feats = list(feats)
        feats[1] = self.gca_lv2(feats[0], feats[1]) + feats[1]#C_2^1
        feats[2] = self.gca_lv3(feats[1], feats[2]) + feats[2]#C_3^1
        feats[3] = self.gca_lv4(feats[2], feats[3]) + feats[3]#C_4^1
        #Top-down Transfer Module
        #stage2_lv5
        c_2_5_list = []
        for ext_f in self.stage2_lv5:
            c_2_5_list.append(ext_f(feats[3]))
        c_2_5 = self.sum_tensor_list(c_2_5_list)
        #stage2_lv4
        c_2_4_pre = feats[2] + self.upsample_stage2_lv5(c_2_5)
        c_2_4_list = []
        for ext_f in self.stage2_lv4:
            c_2_4_list.append(ext_f(c_2_4_pre))
        c_2_4 = self.sum_tensor_list(c_2_4_list)
        #stage2_lv3
        c_2_3_pre = feats[1] + self.upsample_stage2_lv4(c_2_4)
        c_2_3_list = []
        for ext_f in self.stage2_lv3:
            c_2_3_list.append(ext_f(c_2_3_pre))
        c_2_3 = self.sum_tensor_list(c_2_3_list)
        #P2
        p2 = self.upsample_stage2_lv3(c_2_3) + feats[0]
        stage2_outs = [c_2_3, c_2_4, c_2_5]
        #stage3_lv5
        c_3_5_list = []
        for ext_f in self.stage3_lv5:
            c_3_5_list.append(ext_f(stage2_outs[2]))
        c_3_5 = self.sum_tensor_list(c_3_5_list)
        #stage3_lv4
        c_3_4_pre = stage2_outs[1] + self.upsample_stage3_lv5(c_3_5)
        c_3_4_list = []
        for ext_f in self.stage3_lv4:
            c_3_4_list.append(ext_f(c_3_4_pre))
        c_3_4 = self.sum_tensor_list(c_3_4_list)
        #f_2
        p3 = stage2_outs[0] + self.upsample_stage3_lv4(c_3_4)
        stage3_outs = [c_3_4, c_3_5]
        #stage4_5
        c_4_5_list = []
        for ext_f in self.stage4_lv5:
            c_4_5_list.append(ext_f(stage3_outs[1]))
        c_4_5 = self.sum_tensor_list(c_4_5_list)
        #stage3_f3
        p4 = self.upsample_stage4_lv5(c_4_5) + stage3_outs[0]
        outs = [p2, p3, p4, c_4_5]
        return outs

@NECKS.register_module()
class Profpn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(Profpn, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.ProFPN = ProFPN()
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,

                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)


    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        laterals = self.ProFPN(inputs)
        laterals.append(F.max_pool2d(laterals[-1], 1, stride=2))

        return tuple(laterals)


if __name__ == "__main__":

    torch.cuda.set_device(0)
    input_0 = torch.randn(4, 256, 200, 336).cuda()
    input_1 = torch.rand(4, 512, 100, 168).cuda()
    input_2 = torch.rand(4, 1024, 50, 84).cuda()
    input_3 = torch.rand(4, 2048, 25, 42).cuda()
    # print(input_3.transpose(-2,-1).shape)
    inputs = [input_0, input_1, input_2, input_3]#, input_1, input_2, input_3]
    net = Profpn(in_channels=[256, 512, 1024, 2048],out_channels=256, num_outs=5).cuda()
    # net = ASPP(2048,32 ).cuda()
    # print(FPN(input_3))
    for i in net(inputs):
        print(i.shape)
    # net = HRFPN().cuda()
    # print(net(inputs))

