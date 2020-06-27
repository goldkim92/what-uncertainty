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

