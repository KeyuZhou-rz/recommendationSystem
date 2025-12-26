import torch
import torch.nn as nn
'''
11 categories: suit, sweater,
padding, shirt, tee, windbreak, mountainwear, fur, hoodies,
jacket and vest.
'''
class SDAE_GCL(nn.Module):
    def __init__(self,input_dim=2068,category_dim=4,hidden_dims=[1000,500],latent_dim=2):
        super().__init__()
        # conpress 2068 dimenstion features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0],hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1],latent_dim),
            nn.Tanh()
        )
        # Reconstruct Visual Features, trying to recreate the original ResNet+color features from 2 coordinates
        self.decoder_visual = nn.Sequential(
            nn.Linear(latent_dim,hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1],hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0],input_dim)

        )

        # Reconstruct Correlative Labels
        # Try to guess the category (eg. suit) from the 2 coordinates

        self.decoder_category = nn.Sequential(
            nn.Linear(latent_dim,category_dim),
            # nn.Softmax(dim=1) # Output probabilities for categories
        )

    def forward(self,x):
        # Input Visual Features (2068)
        # 1. Encode to latent space (wc-hs coord)
        latent_iss = self.encoder(x)
        # Decode back
        x_hat = self.decoder_visual(latent_iss)
        c_hat = self.decoder_category(latent_iss)

        return x_hat,c_hat,latent_iss
        