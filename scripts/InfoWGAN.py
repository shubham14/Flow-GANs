import torch
import torchvision
import torch.nn as nn

class InfoGAN(nn.Module):
    def __init__(self):
        super(InfoGAN, self).__init__()
        