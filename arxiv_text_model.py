import numpy as np 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import datasets

# Constants
SEGMENT = 10           # number of segments
SEGMENT_LENGTH = 512   # context window length
CHUNK_SIZE = SEGMENT * SEGMENT_LENGTH

def encode_text(s):
    # Using frombuffer instead of fromstring due to deprecation
    return np.frombuffer(s.encode(), dtype=np.uint8)

def decode_text(s):
    return ''.join([chr(i) for i in s])

def clip_article(article):
    remainder = len(article) % CHUNK_SIZE
    return article[:-remainder] if remainder else article

def load_and_process_data():
    # Load dataset
    dataset = datasets.load_dataset("ccdv/arxiv-summarization", split='train',streaming=True)
    raw_dataset = list(dataset.take(3500))
    
    # Extract and filter articles
    articles = [d['article'] for d in raw_dataset]
    articles = [a for a in articles if len(a) >= CHUNK_SIZE]
    
    # Process articles
    converted = [encode_text(article) for article in articles]
    clipped = [clip_article(article) for article in converted]
    chunked = [article.reshape(-1, CHUNK_SIZE) for article in clipped]
    processed_data = torch.tensor(np.concatenate(chunked), dtype=torch.long)
    
    return processed_data

def create_model():
    return nn.Sequential(
        nn.Embedding(128, 16),
        nn.Linear(16, 150),
        nn.ReLU(),
        nn.Linear(150, 150),
        nn.ReLU(),
        nn.Linear(150, 128)
    )

def train_model(model, data_loader, num_epochs=100):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    model.train()
    device = next(model.parameters()).device
    
    for epoch in range(num_epochs):
        chunk = next(data_loader)
        seq = chunk[:, :-1]
        labels = chunk[:, 1:]
        
        train_loss = 0
        for seq_segment, labels_segment in zip(seq.chunk(SEGMENT, dim=-1), 
                                             labels.chunk(SEGMENT, dim=-1)):
            optimizer.zero_grad()
            seq_segment = seq_segment.to(device)
            labels_segment = labels_segment.to(device)
            y_pred = model(seq_segment)
            loss = loss_fn(y_pred.transpose(2, 1), labels_segment)
            loss.backward()
            optimizer.step()
            train_loss += loss / SEGMENT
            
        print(f"Epoch {epoch}, Loss: {train_loss}")

def main():
    # Process data
    processed_data = load_and_process_data()
    print(f"Processed data shape: {processed_data.shape}")
    
    # Create data loader
    loader = DataLoader(processed_data, batch_size=8, shuffle=True)
    loader = iter(loader)
    
    # Create and train model
    model = create_model()
    train_model(model, loader)

if __name__ == "__main__":
    main() 