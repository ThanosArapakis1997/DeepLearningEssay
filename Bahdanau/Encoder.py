import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        
        # Embedding layer to convert word indices into dense vectors
        self.embedding = nn.Embedding(input_size, embedding_dim)
        
        # Bidirectional GRU
        self.gru = nn.GRU(embedding_dim, hidden_size, bidirectional=True)
        
    def forward(self, src):
        # src = [src_len, batch_size]
        
        embedded = self.dropout(self.embedding(src))
        # embedded = [src_len, batch_size, emb_dim]
        
        outputs, hidden = self.gru(embedded)
        # outputs = [src_len, batch_size, enc_hid_dim * 2]
        # hidden = [n_layers * num_directions, batch_size, enc_hid_dim]
        
        return outputs, hidden
