import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

def train(dataset, net, args):
    
    net.train()
    net.to(device)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    
    
    for epoch in range(args.max_epochs):
        
        state_h, state_c = net.init_state(args.sequence_length)
        state_h, state_c = state_h.to(device), state_c.to(device)
        
        for batch, (x,y) in enumerate(dataloader):
            optimizer.zero_grad()
            
            
            x, y = x.to(device), y.to(device)
            y_pred, (state_h, state_c) = net(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1,2), y)
            
            state_h = state_h.detach()
            state_c = state_c.detach()
            
            loss.backward()
            optimizer.step()
            
            torch.save(net.state_dict(), 'lstm_model.pt')
            
            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })
            




parser = argparse.ArgumentParser()

parser.add_argument('--max-epochs', type=int, default=3)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--sequence-length', type=int, default=128)
args = parser.parse_args()


dataset = Dataset(args)
model = Model()

train(dataset, model, args)

