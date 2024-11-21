import numpy as np 
import faiss
import torch
from einops import rearrange

batch_size = 8
seq_len = 512
dim = 128
faiss.omp_set_num_threads(16)

file_name = 'external_memory.dat'

class KNN_Search:
    def __init__(self,file_name,batch_size,seq_len,dim):
        self.file_name = file_name
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.data = np.memmap(file_name, dtype=np.float32, mode='w+', shape=(self.batch_size * self.seq_len * 1000, 2, dim))
        self.ids = 0

    def add_data(self,data):
        data = rearrange(data,'b s c d -> (b s) c d')
        k,v = data.unbind(dim=-2)
        len_new_data = data.shape[0]
        self.data[self.ids:self.ids+len_new_data, :, :] = data.numpy()
        self.index.add(np.ascontiguousarray(k.numpy()))
        self.ids += len_new_data
        self.data.flush()
    
    def search(self,queries,k=3):
        queries = rearrange(queries,'b n d -> (b n) d')
        distances, indices = self.index.search(queries.numpy(), k=k)
        output = self.data[indices]
        output = rearrange(output,'(b n) k c d -> b n k c d',n=self.seq_len)
        output = torch.from_numpy(output)

        return output

    def clear(self):
        self.index = faiss.IndexFlatIP(self.dim)
        self.ids = 0
        self.data = np.memmap(self.file_name, dtype=np.float32, mode='w+', shape=(self.batch_size * self.seq_len * 1000, 2, self.dim))
        self.data.flush()


def main():
    knn = KNN_Search(file_name,batch_size,seq_len,dim)
    data = torch.randn(batch_size*10,seq_len,2,dim)
    knn.add_data(data)
    queries = torch.randn(batch_size,seq_len,dim)
    kv_memory = knn.search(queries,k=3)
    print(kv_memory.shape)
    knn.clear()

if __name__ == '__main__':
    main()