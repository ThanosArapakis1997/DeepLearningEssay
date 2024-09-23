import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Badanau(nn.Module):
    def __init__(self,  source_vocab_size, target_vocab_size, d_model,
                 num_layers, max_seq_length, dropout):        
        super(Badanau, self).__init__()
        
    
    def forward(self, src):
        return 0;
    
    def train_model(self):
        return 0;