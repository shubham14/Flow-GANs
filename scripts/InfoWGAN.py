'''
Network component definition for InfoGAN with Wasserstein distance
'''
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from config import NetConfig
from torchvision import models

class Generator(nn.Module):
    '''
    cfg is the NetConfig object with hyperparameters
    '''
    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(cfg.input_dim)

class Quantizer(nn.Module):
    def __init__(self, levels=[i for i in range(-2, 3)], sigma=1.0):
        super(Quantizer, self).__init__()
        self.levels = levels
        self.sigma = sigma

    def forward(self, input):
        levels = input.data.new(self.levels)
        xsize = list(input.size())

        # Compute differentiable soft quantized version
        input = input.view(*(xsize + [1]))
        level_var = Variable(levels, requires_grad=False)
        dist = torch.pow(input-level_var, 2)
        output = torch.sum(level_var * nn.functional.softmax(-self.sigma*dist, dim=-1), dim=-1)

        # Compute hard quantization (invisible to autograd)
        _, symbols = torch.min(dist.data, dim=-1, keepdim=True)
        for _ in range(len(xsize)): levels.unsqueeze_(0)
        levels = levels.expand(*(xsize + [len(self.levels)]))

        quant = levels.gather(-1, symbols.long()).squeeze_(dim=-1)

        # Replace activations in soft variable with hard quantized version
        output.data = quant

        return output


class Encoder(nn.Module):
    '''
    latent_dim to which the encoder projects is same dimensional 
    as Generator noise dim
    Can switch instance norm to Batch Norm
    The parameters of Encoder will be dependent on the dataset
    C : bottleneck dimension
    input_dim : channels of input image
    '''
    def __init__(self, input_dim, C, filters=[60, 120, 240, 480, 960]):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.filters = filters
        self.cnn1 = nn.Sequential(nn.Conv2d(input_dim, filters[0], 7, stride=1),
                    nn.ReLU(),
                    nn.InstanceNorm2d(filters[0]))
        self.cnn2 = nn.Sequential(nn.Conv2d(filters[0], filters[1], 3, padding=1),
                    nn.ReLU(),
                    nn.InstanceNorm2d(filters[1]))
        self.cnn3 = nn.Sequential(nn.Conv2d(filters[1], filters[2], 3, padding=1),
                    nn.ReLU(),
                    nn.InstanceNorm2d(filters[2]))
        self.cnn4 = nn.Sequential(nn.Conv2d(filters[2], filters[3], 3, padding=1),
                    nn.ReLU(),
                    nn.InstanceNorm2d(filters[3]))
        self.cnn5 = nn.Sequential(nn.Conv2d(filters[3], filters[4], 3, padding=1),
                    nn.ReLU(),
                    nn.InstanceNorm2d(filters[4]))
        self.cnn6 = nn.Sequential(nn.Conv2d(filters[4], C, 3, stride=1),
                    nn.ReLU(),
                    nn.InstanceNorm2d(filters[2]))
        self.pad2 = nn.modules.padding.ReflectionPad2d(1)  
        self.pad1 = nn.modules.padding.ReflectionPad2d(3)   
        self.maxpool = nn.MaxPool2d(2, return_indices=True)
        
    def forward(self, x):
        out = self.pad1(x)
        out = self.cnn1(out)
        out, ind1 = self.maxpool(out)
        out = self.cnn2(out)
        out, ind2 = self.maxpool(out)
        out = self.cnn3(out)
        out, ind3 = self.maxpool(out)
        out = self.cnn4(out)
        out, ind4 = self.maxpool(out)
        out = self.cnn5(out)
        out = self.pad2(out)
        out = self.cnn6(out)
        return out, [ind1, ind2, ind3, ind4]

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

class InfoGAN(nn.Module):
    def __init__(self, gen, quant, dis):
        super(InfoGAN, self).__init__()
        self.gen = gen
        self.quant = quant
        self.dis = dis
    
    def forward(self, img):
        '''
        img is input to the GAN network
        '''
        gen_output = self.gen(img)
        quant_out = self.quant(gen_output)
        out = self.dis(quant_out)
        return out


