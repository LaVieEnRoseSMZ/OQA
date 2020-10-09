from collections import OrderedDict

import torch.nn as nn
from utils import MyModule, build_activation, get_same_padding, SEModule, ShuffleLayer
from layers import My2DLayer
from elastic_nn.modules.lsq_conv import LsqConv

class MBInvertedLSQConvLayer(MyModule):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, expand_ratio=6, mid_channels=None, act_func='relu6', use_se=False, nbit_a=4, nbit_w=4):
        super(MBInvertedLSQConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.use_se = use_se

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                # ('conv', nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False)),
                ('conv', LsqConv(self.in_channels, feature_dim, 1,
                                 quan_name_w='lsq', quan_name_a='lsq', nbit_w=nbit_w, nbit_a=nbit_a,
                                 stride=1, padding=0, bias=False)),
                ('bn', nn.BatchNorm2d(feature_dim)),
                ('act', build_activation(self.act_func, inplace=True)),
            ]))

        pad = get_same_padding(self.kernel_size)
        depth_conv_modules = [
            # ('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad, groups=feature_dim, bias=False)),
            ('conv', LsqConv(feature_dim, feature_dim, kernel_size,
                             quan_name_w='lsq', quan_name_a='lsq', nbit_w=nbit_w, nbit_a=nbit_a,
                             stride=stride, padding=pad, groups=feature_dim, bias=False)),
            ('bn', nn.BatchNorm2d(feature_dim)),
            ('act', build_activation(self.act_func, inplace=True))
        ]
        if self.use_se:
            depth_conv_modules.append(('se', SEModule(feature_dim)))
            #depth_conv_modules.insert(2, ('se', SEModule(feature_dim)))
        self.depth_conv = nn.Sequential(OrderedDict(depth_conv_modules))

        self.point_linear = nn.Sequential(OrderedDict([
            # ('conv', nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
            ('conv', LsqConv(feature_dim, out_channels, 1,
                             quan_name_w='lsq', quan_name_a='lsq', nbit_w=nbit_w, nbit_a=nbit_a,
                             stride=1, padding=0, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
        ]))

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @property
    def module_str(self):
        if self.mid_channels is None:
            expand_ratio = self.expand_ratio
        else:
            expand_ratio = self.mid_channels // self.in_channels
        layer_str = '%dx%d_MBConv%d_%s' % (self.kernel_size, self.kernel_size, expand_ratio, self.act_func.upper())
        if self.use_se:
            layer_str = 'SE_' + layer_str
        layer_str += '_O%d' % self.out_channels
        return layer_str

    @property
    def config(self):
        return {
            'name': MBInvertedLSQConvLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expand_ratio': self.expand_ratio,
            'mid_channels': self.mid_channels,
            'act_func': self.act_func,
            'use_se': self.use_se,
        }

    @staticmethod
    def build_from_config(config):
        return MBInvertedLSQConvLayer(**config)


class LSQConvLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act', nbit_a=4, nbit_w=4):
        # default normal 3x3_Conv with bn and relu
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle
        self.nbit_a = nbit_a
        self.nbit_w = nbit_w

        super(LSQConvLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict()
        # weight_dict['conv'] = nn.Conv2d(
        #     self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=padding,
        #     dilation=self.dilation, groups=self.groups, bias=self.bias
        # )
        weight_dict['conv'] = LsqConv(
            self.in_channels, self.out_channels, kernel_size=self.kernel_size,
            quan_name_w='lsq', quan_name_a='lsq', nbit_w=self.nbit_w, nbit_a=self.nbit_a,
            stride=self.stride, padding=padding, dilation=self.dilation, groups=self.groups, bias=self.bias
        )
        if self.has_shuffle and self.groups > 1:
            weight_dict['shuffle'] = ShuffleLayer(self.groups)

        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                conv_str = '%dx%d_DilatedConv' % (kernel_size[0], kernel_size[1])
            else:
                conv_str = '%dx%d_Conv' % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                conv_str = '%dx%d_DilatedGroupConv' % (kernel_size[0], kernel_size[1])
            else:
                conv_str = '%dx%d_GroupConv' % (kernel_size[0], kernel_size[1])
        conv_str += '_O%d' % self.out_channels
        return conv_str

    @property
    def config(self):
        return {
            'name': LSQConvLayer.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
            **super(LSQConvLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return LSQConvLayer(**config)
