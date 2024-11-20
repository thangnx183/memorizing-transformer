import torch
import torch.nn as nn
import math
from einops import rearrange

class RelativePosition(nn.Module):
    def __init__(self, rp_scale, num_buckets=32, rp_max_distance=128, heads=8):
        super().__init__()
        self.scale = rp_scale
        self.num_buckets = num_buckets
        self.rp_max_distance = rp_max_distance
        self.relative_attention_embedding = nn.Embedding(num_buckets, heads)

    def relative_position_bucket(self, relative_position_matrix):
        n = -relative_position_matrix
        n = torch.max(n, torch.zeros_like(n))

        max_exact = self.num_buckets // 2

        is_small = n < max_exact
        val_if_large = max_exact + (torch.log(n.float() / max_exact) / math.log(self.rp_max_distance / max_exact) * (self.num_buckets - max_exact)).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, self.num_buckets - 1))

        return torch.where(is_small, n, val_if_large)

    def forward(self, sequence_length):

        sequence_pos = torch.arange(sequence_length, dtype=torch.long)
        #########
        context_pos = torch.arange(2 * sequence_length, dtype=torch.long)
        sequence_rel_pos = rearrange(sequence_pos, 'i -> i 1')
        context_rel_pos = rearrange(context_pos, 'j -> 1 j')
        rel_pos = context_rel_pos - sequence_rel_pos

        position_bucket_indices = self.relative_position_bucket(rel_pos)

        rp_values = self.relative_attention_embedding(position_bucket_indices)
        rp_values = rearrange(rp_values, 'i j h -> () h i j')
        return rp_values * self.scale