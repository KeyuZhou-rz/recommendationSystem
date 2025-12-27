import torch
import torch.nn as nn
'''
11 categories: suit, sweater,
padding, shirt, tee, windbreak, mountainwear, fur, hoodies,
jacket and vest.
'''
class SDAE_GCL(nn.Module):
    def __init__(self,input_dim_pattern=2068,input_dim_color=20,category_dim=4,hidden_dims=[1000,500],latent_dim=2):
        super().__init__()
        # conpress 2048 dimenstion features into 128 to adjust its weight
        self.encoder = nn.Sequential(
            nn.Linear(input_dim_pattern, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512,128),
            nn.ReLU()
        )
        # 20 -> 128 dims
        self.color_encoder = nn.Sequential(
            nn.Linear(input_dim_color,64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64,128),
            nn.ReLU()
        )

        # Fusion layer which combines two 128 dims vectors
        self.fusion_encoder = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,latent_dim),
            nn.Tanh()
        )

        self.decoder_common = nn.Sequential(
            nn.Linear(latent_dim,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU()
        )
        # Post a doubt here how the method extract pattern and color
        self.decoder_pattern = nn.Sequential(
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,input_dim_pattern)
        )

        self.decoder_color = nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,input_dim_color)
        )

        self.decoder_category = nn.Sequential(
            nn.Linear(latent_dim,32),
            nn.ReLU(),
            nn.Linear(32,category_dim)
        )
        # Reconstruct Visual Features, trying to recreate the original ResNet+color features from 2 coordinates
        '''
        self.decoder_visual = nn.Sequential(
            nn.Linear(latent_dim,hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1],hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0],input_dim)

        )
        '''
        # Reconstruct Correlative Labels
        # Try to guess the category (eg. suit) from the 2 coordinates

  
    def forward(self,x):
        x_pattern = x[:,:2048]
        x_color = x[:,2048:]

        feat_p = self.pattern_encoder(x_pattern)
        feat_c = self.color_encoder(x_color)
        # why not use a concreante?
        combined = torch.cat([feat_p,feat_c],dim=1)
        latent_iss = self.fusion_encoder(combined)

        recon_common = self.decoder_pattern(latent_iss)

        x_hat_pattern = self.decoder_pattern(recon_common)
        x_hat_color = self.decoder_color(recon_common)

        x_hat = torch.cat([x_hat_pattern, x_hat_color],dim=1)

        c_hat = self.decoder_category(latent_iss)

        return x_hat, c_hat,latent_iss

        