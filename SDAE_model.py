import torch
import torch.nn as nn
'''
11 categories: suit, sweater,
padding, shirt, tee, windbreak, mountainwear, fur, hoodies,
jacket and vest.
'''
class SDAE_GCL(nn.Module):
    def __init__(self, input_dim_pattern=2048, input_dim_color=20, category_dim=4, latent_dim=64):
        super().__init__()
        
        # FIX 1: Renamed from self.encoder to self.pattern_encoder
        self.pattern_encoder = nn.Sequential(
            nn.Linear(input_dim_pattern, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        
        self.color_encoder = nn.Sequential(
            nn.Linear(input_dim_color, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.fusion_encoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.Tanh()
        )

        self.decoder_common = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        self.decoder_pattern = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim_pattern)
        )

        self.decoder_color = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim_color)
        )

        self.decoder_category = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, category_dim)
        )

    def forward(self, x):
        x_pattern = x[:, :2048]
        x_color = x[:, 2048:]

        # 1. Encode
        feat_p = self.pattern_encoder(x_pattern)
        feat_c = self.color_encoder(x_color)

        # 2. Fuse
        combined = torch.cat([feat_p, feat_c], dim=1)
        latent_iss = self.fusion_encoder(combined)

        # 3. Decode Common (FIX 2: Don't skip this!)
        # Expand 64 -> 256
        recon_common = self.decoder_common(latent_iss) 

        # 4. Decode Specifics (FIX 3: Use recon_common as input)
        x_hat_pattern = self.decoder_pattern(recon_common)
        x_hat_color = self.decoder_color(recon_common)

        x_hat = torch.cat([x_hat_pattern, x_hat_color], dim=1)

        c_hat = self.decoder_category(latent_iss)

        return x_hat, c_hat, latent_iss