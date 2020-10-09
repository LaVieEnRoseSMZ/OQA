import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import horovod.torch as hvd


def gradscale(x, scale):
    yOut = x
    yGrad = x * scale
    y = (yOut - yGrad).detach() + yGrad
    return y


def signpass(x):
    yOut = x.sign()
    yGrad = x
    y = (yOut - yGrad).detach() + yGrad
    return y


def clippass(x, Qn, Qp):
    yOut = F.hardtanh(x, Qn, Qp)
    yGrad = x
    y = (yOut - yGrad).detach() + yGrad
    return y


def roundpass(x):
    yOut = x.round()
    yGrad = x
    y = (yOut - yGrad).detach() + yGrad
    return y


# v is input
# s in learnable step size
# p in the number of bits
def quantize(v, s, p, isActivation):
    if isActivation:
        Qn = 0
        Qp = 2 ** p - 1
        nfeatures = v.nelement()
        gradScaleFactor = 1 / math.sqrt(nfeatures * Qp)

    else:
        Qn = -2 ** (p - 1)
        Qp = 2 ** (p - 1) - 1
        if p == 1:
            Qp = 1
        nweight = v.nelement()
        gradScaleFactor = 1 / math.sqrt(nweight * Qp)

    s = gradscale(s, gradScaleFactor)
    v = v / s.abs()
    v = F.hardtanh(v, Qn, Qp)
    if not isActivation and p == 1:
        vbar = signpass(v)
    else:
        vbar = roundpass(v)
    vhat = vbar * s.abs()
    return vhat


class LsqConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, quan_name_w='lsq', quan_name_a='lsq', nbit_w=4, nbit_a=4,
                 stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(LsqConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        name_a_dict = {'lsq': quantize}
        name_w_dict = {'lsq': quantize}
        self.quan_w = name_w_dict[quan_name_w]
        self.quan_a = name_a_dict[quan_name_a]
        self.sa = nn.Parameter(torch.Tensor(1))
        self.sw = nn.Parameter(torch.Tensor(1))
        self.flag = 1
        #print('bit for a', nbit_a, 'bit for w', nbit_w)
        # self.logger.info('{} bits for activations and {} bits for weight'.format(self.nbit_a, self.nbit_w))
        # self.logger.info('gradscale for s is 1/math.sqrt(nfeatures*Qp), use abs scale, init use 2/sqrt(qmax)*mean')

    def get_quant_x_w(self, x, w):

        if self.flag == 0:
            self.q_max_a = 2 ** (self.nbit_a) - 1
            self.sa.data = 2.0 / math.sqrt(self.q_max_a) * x.abs().mean().detach().view(1)
            # TODO: fix allreduce error
            # world_size = hvd.size()
            # hvd.allreduce(self.sa.data)
            # self.sa.data /= world_size
            # self.logger.info('calculate input mean {} and sa {}'.format(input.abs().mean().data, self.sa.data))
            self.q_max_w = 2 ** (self.nbit_w - 1) - 1
            if self.nbit_w == 1:
                self.q_max_w = 1
            self.sw.data = 2.0 / math.sqrt(self.q_max_w) * w.abs().mean().detach().view(1)
            self.flag = 1
            # self.logger.info('calculate weight mean {} and sw {}'.format(self.weight.abs().mean().data, self.sw.data))

        if self.nbit_w < 32:
            w_quant = self.quan_w(w, self.sw, self.nbit_w, isActivation=False)
        else:
            w_quant = w

        if self.nbit_a < 32:
            x_quant = self.quan_a(x, self.sa, self.nbit_a, isActivation=True)
        else:
            x_quant = x

        return x_quant, w_quant

    def forward(self, x):
        # if self.sa.nelement() == 0:
        # if self.flag == 0:
        #     self.q_max_a = 2 ** (self.nbit_a) - 1
        #     self.sa.data = 2.0 / math.sqrt(self.q_max_a) * input.abs().mean().detach().view(1)
        #     # TODO: fix allreduce error
        #     # world_size = hvd.size()
        #     # hvd.allreduce(self.sa.data)
        #     # self.sa.data /= world_size
        #     # self.logger.info('calculate input mean {} and sa {}'.format(input.abs().mean().data, self.sa.data))
        #     self.q_max_w = 2 ** (self.nbit_w - 1) - 1
        #     if self.nbit_w == 1:
        #         self.q_max_w = 1
        #     self.sw.data = 2.0 / math.sqrt(self.q_max_w) * self.weight.abs().mean().detach().view(1)
        #     self.flag = 1
        #     # self.logger.info('calculate weight mean {} and sw {}'.format(self.weight.abs().mean().data, self.sw.data))
        #
        # if self.nbit_w < 32:
        #     w = self.quan_w(self.weight, self.sw, self.nbit_w, isActivation=False)
        # else:
        #     w = self.weight
        #
        # if self.nbit_a < 32:
        #     x = self.quan_a(input, self.sa, self.nbit_a, isActivation=True)
        # else:
        #     x = input

        x_quant, w_quant = self.get_quant_x_w(x, self.weight)
        #if hvd.rank() == 0:
        #    print('nbit for a and w', self.nbit_a, self.nbit_w)
        #print(x.size(), self.weight.size(), x_quant[0][0][0][0].detach()/self.sa.detach(), w_quant[0][0][0][0].detach()/self.sw.detach(),self.sa.data,self.sw.data)
        output = F.conv2d(x_quant, w_quant, None, self.stride, self.padding, self.dilation, self.groups)
        return output


if __name__ == '__main__':
    input = torch.randn(1, 3, 4, 4)
    layer = LsqConv(3, 2, 3, nbit_a=1, nbit_w=1)
    output = layer(input)
    print(output)
