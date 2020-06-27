'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import numbers
import numpy as np
from scipy import linalg

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        unnormed_tensor = torch.zeros_like(tensor)
        for i, (t, m, s) in enumerate(zip(tensor, self.mean, self.std)):
            unnormed_tensor[i] = t.mul(s).add(m)
            # The normalize code -> t.sub_(m).div_(s)
        return unnormed_tensor


def get_kappa_inv(n):
    column = [1] + [0.5] + [0] * (n-2)
    Kappa = linalg.toeplitz(column)
    Kappa_inv = linalg.pinv(Kappa)
    Kappa_inv = torch.tensor(Kappa_inv, dtype=torch.float32)
    return Kappa_inv


def get_kl_divergence(mu, logvar, Kappa_inv):
    n = Kappa_inv.size(0)
    kld = 0.5 * (Kappa_inv.diag().view(1,-1).matmul(logvar.exp()).matmul(Kappa_inv.diag().view(-1,1))
                 + mu.mul(Kappa_inv.matmul(mu).matmul(Kappa_inv)).sum(dim=[2,3], keepdim=True)
                 - n**2 + 2*n*(np.log(n+1)-n*np.log(2)) -logvar.sum(dim=[2,3], keepdim=True))
    kld = kld.mean() 
    return kld


# def get_kl_divergence(mu, logvar, Kappa_inv, B):
#     n = Kappa_inv.size(0)
#     kld = 0.5 * (Kappa_inv.diag().view(1,-1).matmul(logvar.exp()).matmul(Kappa_inv.diag().view(-1,1))
#                  + mu.mul(Kappa_inv.matmul(mu).matmul(Kappa_inv)).sum(dim=[2,3], keepdim=True)
#                  - n**2 + 2*n*np.log((n+1)/2**n) -logvar.sum(dim=[2,3], keepdim=True))
#     kld1 = kld[:-int(B/4)].mean() 
#     kld2 = kld[-int(B/4):].mean()
#     return kld1, kld2


def get_encoder_inputs(inputs, targets, num_classes, device):
    _,_,H,W = inputs.size()
    targets_onehot = torch.eye(num_classes)[targets]
    targets_onehot = targets_onehot.to(device)
    inputs_enc = torch.cat([inputs, 
                            targets_onehot.repeat(H,W,1,1).permute(2,3,0,1)],
                            dim=1)
    return inputs_enc


def get_baseline(baseline_, inputs, device):
    B,C,H,W = inputs.size()
    if baseline_ == 'mean':
        baseline = inputs.mean(dim=[2,3], keepdim=True).expand(B,C,H,W).contiguous()
    elif baseline_ == 'noise':
        baseline = inputs + torch.randn_like(inputs) * 0.2
    elif baseline_ == 'blur':
        kernel_size, kernel_std = 31, 15
        smoothing = GaussianSmoothing(C, kernel_size, kernel_std).to(device)
        baseline = smoothing(F.pad(inputs, [kernel_size//2]*4, mode='reflect'))
    elif baseline_ == 'random':
        random = torch.rand(1)
        if random >= 0.66:
            baseline = inputs.mean(dim=[2,3], keepdim=True).expand(B,C,H,W).contiguous()
            baseline += torch.randn(baseline.size()).to(device) * 0.3
        elif random >= 0.33:
            baseline = inputs + torch.randn_like(inputs) * 0.3
        else:
            kernel_size, kernel_std = 31, 15
            smoothing = GaussianSmoothing(C, kernel_size, kernel_std).to(device)
            baseline = smoothing(F.pad(inputs, [kernel_size//2]*4, mode='reflect'))
    elif baseline_ == 'random_w_noise':
        random = torch.rand(1)
        if random >= 0.8:
            baseline = inputs + torch.randn_like(inputs) * 0.2
        elif random >= 0.4:
            baseline = inputs.mean(dim=[2,3], keepdim=True).expand(B,C,H,W).contiguous()
        else:
            kernel_size, kernel_std = 15, 7 
            smoothing = GaussianSmoothing(C, kernel_size, kernel_std).to(device)
            baseline = smoothing(F.pad(inputs, [kernel_size//2]*4, mode='reflect'))
    elif baseline_ == 'zero':
        baseline = torch.zeros_like(inputs).to(device)
    return baseline


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.                                                          
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [   
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


# def get_mean_and_std(dataset):
#     '''Compute the mean and std value of dataset.'''
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
#     mean = torch.zeros(3)
#     std = torch.zeros(3)
#     print('==> Computing mean and std..')
#     for inputs, targets in dataloader:
#         for i in range(3):
#             mean[i] += inputs[:,i,:,:].mean()
#             std[i] += inputs[:,i,:,:].std()
#     mean.div_(len(dataset))
#     std.div_(len(dataset))
#     return mean, std
# 
# def init_params(net):
#     '''Init layer parameters.'''
#     for m in net.modules():
#         if isinstance(m, nn.Conv2d):
#             init.kaiming_normal(m.weight, mode='fan_out')
#             if m.bias:
#                 init.constant(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             init.constant(m.weight, 1)
#             init.constant(m.bias, 0)
#         elif isinstance(m, nn.Linear):
#             init.normal(m.weight, std=1e-3)
#             if m.bias:
#                 init.constant(m.bias, 0)
# 
# 
# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
# 
# TOTAL_BAR_LENGTH = 65.
# last_time = time.time()
# begin_time = last_time
# def progress_bar(current, total, msg=None):
#     global last_time, begin_time
#     if current == 0:
#         begin_time = time.time()  # Reset for new bar.
# 
#     cur_len = int(TOTAL_BAR_LENGTH*current/total)
#     rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
# 
#     sys.stdout.write(' [')
#     for i in range(cur_len):
#         sys.stdout.write('=')
#     sys.stdout.write('>')
#     for i in range(rest_len):
#         sys.stdout.write('.')
#     sys.stdout.write(']')
# 
#     cur_time = time.time()
#     step_time = cur_time - last_time
#     last_time = cur_time
#     tot_time = cur_time - begin_time
# 
#     L = []
#     L.append('  Step: %s' % format_time(step_time))
#     L.append(' | Tot: %s' % format_time(tot_time))
#     if msg:
#         L.append(' | ' + msg)
# 
#     msg = ''.join(L)
#     sys.stdout.write(msg)
#     for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
#         sys.stdout.write(' ')
# 
#     # Go back to the center of the bar.
#     for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
#         sys.stdout.write('\b')
#     sys.stdout.write(' %d/%d ' % (current+1, total))
# 
#     if current < total-1:
#         sys.stdout.write('\r')
#     else:
#         sys.stdout.write('\n')
#     sys.stdout.flush()
# 
# def format_time(seconds):
#     days = int(seconds / 3600/24)
#     seconds = seconds - days*3600*24
#     hours = int(seconds / 3600)
#     seconds = seconds - hours*3600
#     minutes = int(seconds / 60)
#     seconds = seconds - minutes*60
#     secondsf = int(seconds)
#     seconds = seconds - secondsf
#     millis = int(seconds*1000)
# 
#     f = ''
#     i = 1
#     if days > 0:
#         f += str(days) + 'D'
#         i += 1
#     if hours > 0 and i <= 2:
#         f += str(hours) + 'h'
#         i += 1
#     if minutes > 0 and i <= 2:
#         f += str(minutes) + 'm'
#         i += 1
#     if secondsf > 0 and i <= 2:
#         f += str(secondsf) + 's'
#         i += 1
#     if millis > 0 and i <= 2:
#         f += str(millis) + 'ms'
#         i += 1
#     if f == '':
#         f = '0ms'
#     return f
