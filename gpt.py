import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np 
import random
import time
import math
from arxiv_text_model import load_and_process_data, train_model
from einops import rearrange, einsum


seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

sequence_length = 512
embedding_dimension = 300
head_dimension = 32
number_heads = 8
batch_size = 8
scaling_factor = head_dimension**-0.5

class XLMultiHeadsAttention(nn.Module):
    def __init__(self,embedding_dimension,heads=8,head_dimension=32):
        super().__init__()
        self.heads = heads
        self.head_dimension = head_dimension
        self.embedding_dimension = embedding_dimension
        self.scaling_factor = self.head_dimension**-0.5

        self.query_proj = nn.Linear(self.embedding_dimension,self.heads*self.head_dimension)
        self.key_proj = nn.Linear(self.embedding_dimension,self.heads*self.head_dimension)
        self.value_proj = nn.Linear(self.embedding_dimension,self.heads*self.head_dimension)
        self.output_proj = nn.Linear(self.heads*self.head_dimension, self.embedding_dimension)

    def forward(self,input,rel_pos=None,xl_memory=None):
        q = rearrange(self.query_proj(input),'b h (d k) -> b d h k',k=self.head_dimension )
        k = rearrange(self.key_proj(input),'b h (d k) -> b d h k',k=self.head_dimension)
        v = rearrange(self.value_proj(input),'b h (d k) -> b d h k',k=self.head_dimension)
        
        if xl_memory is not None:
            k_memory,v_memory = xl_memory.unbind(dim=-2)
            k_memory = rearrange(k_memory,'b h (d k) -> b d h k',k=self.head_dimension)
            v_memory = rearrange(v_memory,'b h (d k) -> b d h k',k=self.head_dimension)
            
            k = torch.cat((k_memory,k),dim=-2)
            v = torch.cat((v_memory,v),dim=-2)
            
            xl_seq_length = k_memory.shape[-2]
        
        # print(f'debug qk shape: {q.shape,k.shape}')
        qk = einsum(q,k,'b d i k, b d j k -> b d i j')
        i,j = qk.shape[-2:]
        if rel_pos is not None:
            # print('debug pos, qk shape: ',rel_pos.shape,qk.shape)
            qk = qk + rel_pos[...,-i:,-j:]
        qk = qk * self.scaling_factor
        
        
        mask = torch.ones((i,j), dtype=torch.bool).triu(j-i+1).to(input.device)
        qk_masked = qk.masked_fill(mask,float('-inf'))
        qk_softmax = F.softmax(qk_masked,dim=-1)

        # print(qk_softmax.shape, v.shape)
        qkv = einsum(qk_softmax,v,'b d i j,b d j k -> b i d k')
        qkv = rearrange(qkv,'b i d k -> b i (d k)')

        if xl_memory is None:
            key = rearrange(k,'b d h k -> b h (d k)')
            value = rearrange(v,'b d h k -> b h (d k)')
            current_kv_memory = torch.stack((key,value),dim=-2)
        else:
            key = rearrange(k,'b d h k -> b h (d k)')
            value = rearrange(v,'b d h k -> b h (d k)')
            kv_memory = torch.stack((key,value),dim=-2)
            current_kv_memory = kv_memory[:,xl_seq_length:]
        
        output = self.output_proj(qkv)

        return output, current_kv_memory

class KnnXLMultiHeadsAttention(nn.Module):
    def __init__(self,embedding_dimension,knn,heads=8,head_dimension=32):
        super().__init__()
        self.heads = heads
        self.head_dimension = head_dimension
        self.embedding_dimension = embedding_dimension
        self.scaling_factor = self.head_dimension**-0.5
        self.knn = knn

        self.query_proj = nn.Linear(self.embedding_dimension,self.heads*self.head_dimension)
        self.key_proj = nn.Linear(self.embedding_dimension,self.heads*self.head_dimension)
        self.value_proj = nn.Linear(self.embedding_dimension,self.heads*self.head_dimension)
        self.output_proj = nn.Linear(self.heads*self.head_dimension, self.embedding_dimension)
        self.gate = nn.Parameter(torch.randn(self.heads,1,1))

    def forward(self,input,rel_pos=None,xl_memory=None):
        q = rearrange(self.query_proj(input),'b h (d k) -> b d h k',k=self.head_dimension )
        k = rearrange(self.key_proj(input),'b h (d k) -> b d h k',k=self.head_dimension)
        v = rearrange(self.value_proj(input),'b h (d k) -> b d h k',k=self.head_dimension)
        
        if xl_memory is not None:
            k_memory,v_memory = xl_memory.unbind(dim=-2)
            k_memory = rearrange(k_memory,'b h (d k) -> b d h k',k=self.head_dimension)
            v_memory = rearrange(v_memory,'b h (d k) -> b d h k',k=self.head_dimension)
            
            k = torch.cat((k_memory,k),dim=-2)
            v = torch.cat((v_memory,v),dim=-2)
            
            xl_seq_length = k_memory.shape[-2]
        
        qk = einsum(q,k,'b d i k, b d j k -> b d i j')
        i,j = qk.shape[-2:]
        if rel_pos is not None:
            qk = qk + rel_pos[...,-i:,-j:]
        qk = qk * self.scaling_factor
        
        mask = torch.ones((i,j), dtype=torch.bool).triu(j-i+1).to(input.device)
        qk_masked = qk.masked_fill(mask,float('-inf'))
        qk_softmax = F.softmax(qk_masked,dim=-1)

        qkv = einsum(qk_softmax,v,'b h i j,b h j k -> b h i k')
        # qkv = rearrange(qkv,'b i d k -> b i (d k)')
        
        ### knn attention
        if self.knn.index.ntotal > 0:
            queries = rearrange(q,'b d h k -> b h (d k)')
            kv_external_memory = self.knn.search(queries)
            k_external_memory,v_external_memory = kv_external_memory.unbind(dim=-2)
            k_external_memory = rearrange(k_external_memory,'b s k (h d) -> b h s k d',d=self.head_dimension)
            v_external_memory = rearrange(v_external_memory,'b s k (h d) -> b h s k d',d=self.head_dimension)
            
            q = rearrange(q,'b s (h d) -> b h s d',d=self.head_dimension)
            qk_external_memory = einsum(q,k_external_memory,'b h s d, b h s k d -> b h s k')
            qk_external_memory = qk_external_memory * self.scaling_factor
            qk_external_memory = F.softmax(qk_external_memory,dim=-1)
            
            qkv_external_memory = einsum(qk_external_memory,v_external_memory,'b h s k, b h s k d -> b h s d')
            
            self.gate = torch.sigmoid(self.gate)
            qkv = qkv*self.gate + qkv_external_memory*(1-self.gate)
            qkv = rearrange(qkv,'b h s d -> b s (h d)')
        else:
            qkv = rearrange(qkv,'b h s d -> b s (h d)')
            
        if xl_memory is None:
            key = rearrange(k,'b d h k -> b h (d k)')
            value = rearrange(v,'b d h k -> b h (d k)')
            current_kv_memory = torch.stack((key,value),dim=-2)
        else:
            key = rearrange(k,'b d h k -> b h (d k)')
            value = rearrange(v,'b d h k -> b h (d k)')
            kv_memory = torch.stack((key,value),dim=-2)
            current_kv_memory = kv_memory[:,xl_seq_length:]
        
        output = self.output_proj(qkv)
        self.knn.add(current_kv_memory)

        return output, current_kv_memory
    
class RelativePosition(nn.Module):
  def __init__(
      self,
      rp_scale,
      num_buckets = 32,
      rp_max_distance = 128,
      heads = 8
    ):
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

    sequence_pos = torch.arange(sequence_length, dtype=torch.long,device=self.relative_attention_embedding.weight.device)
    context_pos = torch.arange(-sequence_length,sequence_length, dtype=torch.long,device=self.relative_attention_embeddinggit.weight.device)
    sequence_pos = sequence_pos.reshape(sequence_pos.shape[0], 1)
    rel_pos = context_pos - sequence_pos

    position_bucket_indices = self.relative_position_bucket(rel_pos)

    rp_values = self.relative_attention_embedding(position_bucket_indices)
      # Rearrange (sequence, context, heads) -> (1, heads, sequence, context)
    #   rp_values = rp_values.transpose(0,2)
    #   rp_values = rp_values.unsqueeze(0)
    rp_values = rearrange(rp_values,'i j h -> () h i j')
    return rp_values * self.scale

# if torch.backends.mps.is_available():
#     print("MPS is available")
#     device = torch.device("mps")
#     torch.manual_seed(seed)
# else
#     print("MPS is not available")
#     device = torch.device("cpu")

class Block(nn.Module):
    def __init__(self,embedding_dimension,heads=8,head_dimension=32,dropout=0.1):
        super().__init__()  
        # self.ln1 = nn.LayerNorm(embedding_dimension)
        self.ln2 = nn.LayerNorm(embedding_dimension)
        self.attn = XLMultiHeadsAttention(embedding_dimension,heads,head_dimension)
        self.attn_ln = nn.LayerNorm(embedding_dimension)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embedding_dimension),
            nn.Linear(embedding_dimension,embedding_dimension*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dimension*2,embedding_dimension),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,pos=None,xl_memory=None):
        residual = x 
        x,xl_memory = self.attn(self.attn_ln(x),pos,xl_memory)
        x = self.dropout(x)
        x = residual + x
        
        x = x + self.ffn(self.ln2(x))
        return x,xl_memory


class GPT(nn.Module):
    def __init__(self,embedding_dimension,vocab_size,heads=8,head_dimension=32,dropout=0.1,num_blocks=3):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.blocks = nn.ModuleList([Block(embedding_dimension,heads,head_dimension,dropout) for _ in range(num_blocks)])
        self.ln_final = nn.LayerNorm(embedding_dimension)
        self.lm_head = nn.Linear(embedding_dimension,self.vocab_size)
        
        self.relative_position = RelativePosition(rp_scale=head_dimension**-0.5)
    
    def forward(self,x,xl_memories):
        x = self.token_embedding(x)
        # x = self.blocks(x)
        # print('debug : ',x.shape)
        new_xl_memories = []
        for i,block in enumerate(self.blocks):
            x,new_xl_memory = block(x,self.relative_position(x.shape[1]),xl_memories[i])
            new_xl_memories.append(new_xl_memory)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits, new_xl_memories

heads = 8
head_dimension = 32
dropout = 0.1
num_blocks = 8
vocab_size = 128
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

model = GPT(embedding_dimension,vocab_size,heads,head_dimension,dropout,num_blocks).to(device)

# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(),lr=0.001)


# device = torch.device("cpu")
# input = torch.randn(batch_size,sequence_length,embedding_dimension).to(device)

# multi_heads_attention = MultiHeadsAttention(embedding_dimension,number_heads,head_dimension).to(device)

# start_time = time.time()
# output = multi_heads_attention(input)
# end_time = time.time()

# print(output.shape)
# print(f"Time taken: {end_time - start_time} seconds")

processed_data = load_and_process_data()
print(f"Processed data shape: {processed_data.shape}")
loader = DataLoader(processed_data,batch_size=batch_size,shuffle=True)
loader = iter(loader)

start_time = time.time()
train_model(model,loader,num_blocks,num_epochs=100)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")


