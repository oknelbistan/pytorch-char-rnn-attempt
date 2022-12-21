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
    # net.to(device)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True,shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    
    
    for epoch in range(args.max_epochs):
        
        # state_h, state_c = state_h.to(device), state_c.to(device)
        
        for batch, (input_tensor, target_tensor) in enumerate(dataloader):
            
            
            hidden = net.init_state(input_tensor.size(0))
            
            optimizer.zero_grad()
            
            
            # input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
            
            y_pred, _ = net(input_tensor, hidden)
            
            loss = criterion(y_pred, target_tensor.view(-1))
            
            loss.backward()
            
            optimizer.step()
            
            torch.save(net.state_dict(), 'lstm_model.pt')
            
            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })
            




parser = argparse.ArgumentParser()

parser.add_argument('--max-epochs', type=int, default=3)
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--sequence-length', type=int, default=128)
args = parser.parse_args()


dataset = Dataset(args)
model = Model()

train(dataset, model, args)

