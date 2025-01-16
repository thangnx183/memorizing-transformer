import numpy as np 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import datasets

# Constants
SEGMENT = 10           # number of segments
SEGMENT_LENGTH = 512   # context window length
CHUNK_SIZE = SEGMENT * SEGMENT_LENGTH + 1

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
    dataset = datasets.load_dataset("ccdv/arxiv-summarization", split='train')
    raw_dataset = list(dataset.take(17000))
    
    # Extract and filter articles
    articles = [d['article'] for d in raw_dataset]
    articles = [a for a in articles if len(a) >= CHUNK_SIZE]
    
    # Process articles
    converted = [encode_text(article) for article in articles]
    clipped = [clip_article(article) for article in converted]
    chunked = [article.reshape(-1, CHUNK_SIZE) for article in clipped]
    train_chunked = chunked[:int(0.9*len(chunked))]
    val_chunked = chunked[int(0.9*len(chunked)):]
    processed_train_data = torch.tensor(np.concatenate(train_chunked), dtype=torch.long)
    processed_val_data = torch.tensor(np.concatenate(val_chunked), dtype=torch.long)
    
    return processed_train_data,processed_val_data

def create_model():
    return nn.Sequential(
        nn.Embedding(128, 16),
        nn.Linear(16, 150),
        nn.ReLU(),
        nn.Linear(150, 150),
        nn.ReLU(),
        nn.Linear(150, 128)
    )

def train_model(model, data_loader,val_loader,model_blocks, num_epochs=100):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    # model.train()
    device = next(model.parameters()).device
    best_loss = 10
    
    for epoch in range(num_epochs):
        model.train()
        chunk = next(data_loader)
        seq = chunk[:, :-1]
        labels = chunk[:, 1:]
        
        train_loss = 0
        xl_memories = [None] * model_blocks
        model.knn.clear()
        for i, (seq_segment, labels_segment) in enumerate(zip(seq.chunk(SEGMENT, dim=-1), 
                                             labels.chunk(SEGMENT, dim=-1))):
            # optimizer.zero_grad()
            # print(f'debug : {seq_segment.shape,labels_segment.shape}')
            seq_segment = seq_segment.to(device)
            labels_segment = labels_segment.to(device)
            y_pred,xl_memories = model(seq_segment,xl_memories)
            loss = loss_fn(y_pred.transpose(2, 1), labels_segment)
            # (loss/SEGMENT).backward()
            retain_graph = (i != SEGMENT - 1)
            (loss/SEGMENT).backward(retain_graph=retain_graph)
            # optimizer.step()
            train_loss += loss / SEGMENT
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch}, train loss: {train_loss}")
        
        if epoch % 5 == 0 and epoch != 0:
            model.eval()
            chunk = next(val_loader)
            seq = chunk[:, :-1]
            labels = chunk[:, 1:]
            
            val_loss = 0
            
            with torch.no_grad():
                xl_memories = [None] * model_blocks
                model.knn.clear()
                for i, (seq_segment, labels_segment) in enumerate(zip(seq.chunk(SEGMENT, dim=-1), 
                                                    labels.chunk(SEGMENT, dim=-1))):
                    # optimizer.zero_grad()
                    # print(f'debug : {seq_segment.shape,labels_segment.shape}')
                    seq_segment = seq_segment.to(device)
                    labels_segment = labels_segment.to(device)
                    y_pred,xl_memories = model(seq_segment,xl_memories)
                    loss = loss_fn(y_pred.transpose(2, 1), labels_segment)
                    # (loss/SEGMENT).backward()
                    # retain_graph = (i != SEGMENT - 1)
                    # (loss/SEGMENT).backward(retain_graph=retain_graph)
                    # optimizer.step()
                    val_loss += loss / SEGMENT
                if val_loss < best_loss:
                    best_loss = val_loss
                print(f"Epoch {epoch}, val loss: {val_loss}")
            
    print('best loss : ', best_loss)

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