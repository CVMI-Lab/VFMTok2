from torch import nn
import torch, math, pdb
from copy import deepcopy
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from commons.engine.util import instantiate_from_config
from commons.modules.mlp import SwiGLUFFNFused, build_mlp
from commons.modules.attention import Attention4, MaskedCrossAttention, CrossAttention
from commons.modules.attention import MemoryEfficientCrossAttention, XFORMERS_IS_AVAILBLE

class Layer(nn.Module):
    ATTENTION_MODES = {
        "vanilla": CrossAttention,
        "xformer": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, dim_head, mlp_dim, num_head=8, dropout=0.0, xformer=True):
        super().__init__()
        attn_mode = "xformer" if XFORMERS_IS_AVAILBLE else "vanilla"
        
        attn_cls = self.ATTENTION_MODES[attn_mode]
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = attn_cls(query_dim=dim, heads=num_head, dim_head=dim_head, dropout=dropout)
        self.xformer = xformer
        if not xformer:
            self.attn1 = Attention4(dim, num_head, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffnet = SwiGLUFFNFused(in_features=dim, hidden_features=mlp_dim)
        
    def forward(self, x, slots=None, mask = None):

        if self.xformer:
            x = self.attn1(self.norm1(x)) + x
        else:
            x = self.attn1(self.norm1(x), attn_mask=mask) + x
            
        x = self.ffnet(self.norm2(x)) + x

        return x

class Transformer(nn.Module):

    def __init__(self, layer_type, dim, depth, num_head, dim_head, mlp_dim, dropout=0., xformer=False):
        super().__init__()
        self.depth = depth
        assert layer_type in ['normal',]
        layers = {'normal': Layer,}
        layer = layers[layer_type](dim, dim_head, mlp_dim, num_head, dropout, xformer)
        self.layers = _get_clones(layer, depth)
    
    def __len__(self):

        return self.depth
    def forward(self, x, slots = None, mask = None):
        
        for i, layer in enumerate(self.layers):

            x = layer(x, slots, mask)

        return x

def _get_clones(module, N):

    return nn.ModuleList([deepcopy(module) for i in range(N)])
