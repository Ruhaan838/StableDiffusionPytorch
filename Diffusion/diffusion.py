import torch
from torch import nn
from torch.nn import functional as F

from UNet.model import UNET,UNET_OutputLayer
from UNet.times import TimeEmbedding

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
    
    def forward(self, latent, context, time):

        time = self.time_embedding(time)
        
        output = self.unet(latent, context, time)
        
        output = self.final(output)
        
        return output
