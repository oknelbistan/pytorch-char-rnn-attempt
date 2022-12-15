from model import Model
from dataset import Dataset
import torch
import numpy as np
import argparse


def predict(dataset, model, text, next_char=100):
    
    
    
    words = text.split(' ')
    model.eval()
    
    state_h, state_c = model.init_state(len(words))
    
    for i in range(0, next_char):
        x = torch.tensor([[dataset.stoi[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        
        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.itos[word_index])
        
    return words



parser = argparse.ArgumentParser()

parser.add_argument('--max-epochs', type=int, default=3)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--sequence-length', type=int, default=11)
args = parser.parse_args()


dataset = Dataset(args)
model = Model(dataset)

print(predict(dataset, model, text='how are you feel today ?'))