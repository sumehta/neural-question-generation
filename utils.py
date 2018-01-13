import os
import torch
import numpy as np
import random


def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var