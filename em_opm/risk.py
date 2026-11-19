"""Action-conditioned boundary and execution-risk model."""
import torch
from torch import nn
class RiskNet(nn.Module):
 def __init__(self,k=5):
  super().__init__();self.enc=nn.Sequential(nn.Conv2d(5,24,5,2,2),nn.ReLU(),nn.Conv2d(24,48,3,2,1),nn.ReLU(),nn.Conv2d(48,64,3,2,1),nn.ReLU(),nn.Conv2d(64,96,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));self.head=nn.Sequential(nn.Linear(96+3+2+2*k,192),nn.ReLU(),nn.Dropout(.1),nn.Linear(192,1))
 def forward(self,b,s,g,t):return self.head(torch.cat([self.enc(b).flatten(1),s,g,t.flatten(1)],1))
