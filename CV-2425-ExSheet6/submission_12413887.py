"""Submission for exercise sheet 5

SUBMIT this file as submission_<STUDENTID>.py where
you replace <STUDENTID> with your student ID, e.g.,
submission_1234567.py
"""

import torch
import torch.nn as nn
from einops import repeat, rearrange
import numpy as np


# Exercise 6.1
class Encoder(nn.Module):
    def __init__(self, 
        in_channels: int = 3, 
        patch_size: int = 16, 
        emb_size: int = 768):
        
        super().__init__()
        self.emb_size=emb_size
        self.patch_size=patch_size
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels,     # Number of input channels
            out_channels=emb_size,       # Number of filters (embedding size)
            kernel_size=patch_size,      # Kernel size matches patch size
            stride=patch_size            # Non-overlapping patches
        )
        
        # # Uncomment and complete the line in case you solve Step 2
        self.token=nn.Parameter(torch.ones(1, emb_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fn_cl=self.conv_layer(x)
        batch_size, channels, height, width = fn_cl.shape
        
        # Reshape the tensor to (batch_size, num_patches, emb_size)
        fn_cl = fn_cl.view(batch_size, self.emb_size, height * height)
        
        out_1=rearrange(fn_cl, 'b c hw  -> b hw c ')
        token_repeated=repeat(self.token, '1 d -> b 1 d', b=batch_size)

        # Concatenate the token with the patch embeddings along dim=1
        out_2 = torch.cat((out_1, token_repeated), dim=1)
       
        return out_1, out_2
       