import torch
from torch import nn 
import torch.functional as F


class Model(nn.Module):
    def __init__(
        self, input_size=128, 
        hidden_size = 128,
        embedding_dim = 128,
        num_layer = 1):
        
        super(Model, self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding = embedding_dim
        self.input_size = input_size
        self.num_layers  = num_layer 
        
        
        self.embed = nn.Embedding(num_embeddings=self.input_size, embedding_dim=self.embedding)
        
        self.LSTM = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers = self.num_layers,
            batch_first = True
            )
        
        
        self.fc = nn.Linear(self.hidden_size, self.input_size)
        
    def forward(self, x, hidden):
        embed = self.embed(x)
        output, hidden = self.LSTM(embed, hidden)
        
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.fc(output)
        return output, hidden

    def init_state(self, batch_size):
        return(
        torch.zeros(self.num_layers, batch_size , self.hidden_size),
        torch.zeros(self.num_layers, batch_size , self.hidden_size))
