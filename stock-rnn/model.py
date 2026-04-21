"""
The RNN (LSTM) model
"""
import os
import random
import shutil
import time
import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class LSTMModel(nn.Module):
    def __init__(self,
                stock_count,
                input_size=1, 
                hidden_size=128,
                num_layer=1,
                embed_size=16,
                dropout=0.1,
                logs_dir="logs",
                 ):
        super(LSTMModel, self).__init__()
        self.stock_count = stock_count
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.embed_size = embed_size
        self.logs_dir = logs_dir

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layer,
                            batch_first=True,
                            dropout=dropout if num_layer > 1 else 0.0
                            )
        
        self.fc = nn.Linear(hidden_size, 1)

        if self.embed_size > 0 and self.stock_count > 1:
            # Create the embedding matrix
            self.stock_emb = nn.Embedding(self.stock_count, self.embed_size)
    
    def forward(self, x, stock_id):
        # price_ -> [B, T, 1], stock_id -> [B]
        batch_size, time_step, _ = x.shape

        emb = self.stock_emb(stock_id) #[Batch_size, emb_size]
        emb = emb.unsqueeze(1).expand(batch_size, time_step, emb.size(-1)) # [B, T, e]
        x = torch.cat([x, emb], dim=-1)
        
        h0 = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out

if __name__ == "__main__":
    lstm = LSTMModel(10, 1, 128, 1, 500)
    summary(lstm)
