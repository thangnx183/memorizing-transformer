import numpy as np
import faiss
import torch
from einops import rearrange

batch_size = 8
seq_len = 512
dim = 128
faiss.omp_set_num_threads(16)

file_name = "external_memory.dat"


class KNN_Search:
    def __init__(self, file_name, batch_size, seq_len, dim):
        """Initialize the KNN_Search class.

        Args:
            file_name (str): The name of the file to store the data.
            batch_size (int): The size of the batches for processing.
            seq_len (int): The length of the sequences.
            dim (int): The dimensionality of the embeddings.
        """
        self.file_name = file_name
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.data = np.memmap(
            file_name,
            dtype=np.float32,
            mode="w+",
            shape=(self.batch_size * self.seq_len * 1000, 2, dim),
        )
        self.ids = 0
        self.device = "cpu"

    def add_data(self, data):
        """Add data to the KNN index and memory.

        Args:
            data (torch.Tensor): The input data tensor of shape (batch, sequence, 2, embed_dim).
        """
        self.device = data.device
        data = rearrange(
            data, "b s c d -> (b s) c d"
        )  # (batch,sequense,2,embed_dim) -> (batch*sequense,2, embed_dim)
        k, v = data.unbind(
            dim=-2
        )  # (batch*sequense, embed_dim), (batch*sequense,2, embed_dim)
        len_new_data = data.shape[0]
        self.data[self.ids : self.ids + len_new_data, :, :] = (
            data.detach().cpu().numpy()
        )
        self.index.add(
            np.ascontiguousarray(k.detach().cpu().numpy())
        )  # (batch*sequense, embed_dim)
        self.ids += len_new_data
        self.data.flush()

    def search(self, queries, k=3):
        """Search for the top-k nearest neighbors for the given queries.

        Args:
            queries (torch.Tensor): The input query tensor of shape (batch, sequence, embed_dim).
            k (int): The number of nearest neighbors to retrieve.

        Returns:
            torch.Tensor: The output tensor containing the top-k nearest neighbors.
        """
        queries = rearrange(
            queries, "b n d -> (b n) d"
        )  # (batch,sequense,embed_dim) -> (batch*sequense, embed_dim)
        distances, indices = self.index.search(
            queries.detach().cpu().numpy(), k=k
        )  # distance : (batch*sequense, topk)
        output = self.data[indices]
        output = rearrange(
            output, "(b n) k c d -> b n k c d", n=self.seq_len
        )  # (batch*sequense, topk, 2, embed_dim) -> (batch, sequense, topk, 2, embed_dim) | topk kv pair
        output = torch.from_numpy(output).to(self.device)

        return output

    def clear(self):
        """Clear the KNN index and reset the internal state."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.ids = 0
        self.data = np.memmap(
            self.file_name,
            dtype=np.float32,
            mode="w+",
            shape=(self.batch_size * self.seq_len * 1000, 2, self.dim),
        )
        self.data.flush()


def main():
    knn = KNN_Search(file_name, batch_size, seq_len, dim)
    data = torch.randn(batch_size * 10, seq_len, 2, dim)
    print(data.shape)
    knn.add_data(data)
    queries = torch.randn(batch_size, seq_len, dim)
    kv_memory = knn.search(queries, k=3)
    print(kv_memory.shape)
    knn.clear()


if __name__ == "__main__":
    main()
