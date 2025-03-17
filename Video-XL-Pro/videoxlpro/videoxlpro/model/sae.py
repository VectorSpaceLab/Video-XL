import torch

from .sae_utils import SamePadConv3d,Normalize, SiLU, TemporalAttention, AttnBlock3D, RESMLP
import torch.nn as nn

class SiglipAE(nn.Module):
    def __init__(self):
        super().__init__()
        temporal_stride=2
        norm_type = "group"
                   
        self.temporal_encoding = nn.Parameter(torch.randn((4,1152)))
        #self.vision_tower=SigLipVisionTower('google/siglip-so400m-patch14-384')
        self.encoder=nn.Sequential(
            AttnBlock3D(1152),
            TemporalAttention(1152),
            
            SamePadConv3d(1152,1152,kernel_size=3,stride=(temporal_stride, 1, 1),padding_type="replicate"),
            
            AttnBlock3D(1152),
            TemporalAttention(1152),
            
            SamePadConv3d(1152,1152,kernel_size=3,stride=(temporal_stride, 1, 1),padding_type="replicate"),
            
        )
    def forward(self, x):
        b_,c_,t_,h_,w_=x.shape

        temporal_encoding = self.temporal_encoding.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        temporal_encoding = temporal_encoding.expand(b_, -1, -1, h_, w_)  # (B, T, C, H, W)
        temporal_encoding = temporal_encoding.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)  
        x = x + temporal_encoding
        
        x=self.encoder(x)
        return x
    
# image=torch.randn(1,1152,4,24,24).to('cuda')


# model = SiglipAE().to('cuda')
# model.load_state_dict(torch.load('encoder.pth'),strict=False)

# image=model(image)

# print(image.shape)