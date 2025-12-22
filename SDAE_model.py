import torch
import torch.nn as nn

class SDAE_GCL(nn.module):
    def _init__(self,input_dim=2068,category=11,hidden_dims=[1000,500]):
        super(SDAE_GCL, self).__init__()
        # conpress 2068 dimenstion features
        self.encoder = nn.Sequential(
            
        )