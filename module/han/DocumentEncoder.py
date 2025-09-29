import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DocumentEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128, use_gru=True):
        super(DocumentEncoder, self).__init__()
        self.use_gru = use_gru
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        if use_gru:
            self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
            attn_dim = hidden_dim * 2
        else:
            attn_dim = embedding_dim

        self.attn_fc = nn.Linear(attn_dim, attn_dim)
        self.context_vector = nn.Parameter(torch.randn(attn_dim))

    def forward(self, x):  # x: [batch_size, num_paragraphs, embedding_dim]
        if self.use_gru:
            out, _ = self.gru(x)  # [batch, num_paragraphs, hidden_dim*2]
        else:
            out = x  # No contextual GRU

        u = torch.tanh(self.attn_fc(out))  # [batch, num_paragraphs, attn_dim]
        α = F.softmax(torch.matmul(u, self.context_vector), dim=1)  # [batch, num_paragraphs]
        doc_emb = torch.sum(out * α.unsqueeze(-1), dim=1)  # [batch, attn_dim]
        return doc_emb  # Use this for retrieval

def read_embeddings(folder_path, filename):
    file_path = os.path.join(folder_path, filename)
    array = np.load(file_path)
    tensor = torch.tensor(array)  # Convert to PyTorch tensor
    return tensor

if __name__ == '__main__':
    embedding_dim = 768
    hidden_dim = 128
    batch_size = 4
    num_paragraphs = 10
    # Create dummy paragraph embeddings
    # paragraph_embeddings = torch.randn(batch_size, num_paragraphs, embedding_dim)

    all_embeddings = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            array = np.load(file_path)
            tensor = torch.tensor(array)  # Convert to PyTorch tensor
            all_embeddings.append(tensor)


    # Initialize the encoder
    encoder = DocumentEncoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, use_gru=True)

    # Forward pass
    document_embeddings = encoder(paragraph_embeddings)  # [batch_size, output_dim]