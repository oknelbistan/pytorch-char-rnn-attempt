import torch
from torch import nn 



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
        
    def forward(self, x, prev_state):
        embed = self.embed(x)
        output, (state_h, state_c) = self.LSTM(embed.unsqueeze(1), prev_state)
                
        logits = self.fc(output.reshape(output.shape[0], -1))
        
        return logits, (state_h, state_c)

    def init_state(self, sequence_lenght):
        return(
        torch.zeros(self.num_layers, sequence_lenght , self.hidden_size),
        torch.zeros(self.num_layers, sequence_lenght , self.hidden_size))
