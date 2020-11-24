import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader



class expression_model(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))




'''
class SimpleResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        return self.relu2(out) + x # ReLU can be applied before or after adding the input
'''


def conv_block(in_channels, out_channels, pool = False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace = True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)



class ResNet_expression(expression_model):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64) #went from 1 to 64 channels(400*64*48*48)
        self.conv2 = conv_block(64, 128, pool=True)  # batchsize*128*24*24
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))  #residual block one(no change in num_of_channel no pooling)
        
        self.conv3 = conv_block(128, 256, pool=True)  #400*256*12*12
        self.conv4 = conv_block(256, 512, pool=True)   #400*512*6*6
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))    #400*512*6*6
        
        self.conv5 = conv_block(512, 1024, pool=True)  #400*1025*3*3
        
        self.res3 = nn.Sequential(conv_block(1024, 1024), conv_block(1024, 1024))    #400*1024*3*3
        
        self.classifier = nn.Sequential(nn.MaxPool2d(3), #400 * 1024 * 1 * 1
                                        nn.Flatten(), 
                                        nn.Linear(1024, num_classes)) 
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.res3(out) + out
        
        out = self.classifier(out)
        return out





def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


input_size = 48*48
output_size = 7

device = get_default_device()

model2 = to_device(ResNet_expression(1, output_size), device)

model2.load_state_dict(torch.load('/home/praveen/Desktop/Projects/technocolab_project_2/emotion_recognition/models/resnet-facial.pth'))