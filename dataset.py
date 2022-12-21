import torch
from collections import Counter
from torch import nn
import torch.nn.functional as F
        
class Dataset(torch.utils.data.Dataset):
    def __init__(self,args):
        
        self.args = args
        self.text = self.load_text()
        self.chars = self.get_uniq_char()
        
        self.stoi = {_str : idx + 1 for idx, _str in enumerate(self.chars)}
        self.itos = {idx : _str for _str, idx in self.stoi.items()}

        # all char in the text converted
        self.char_index = [self.stoi[char] for char in self.text]
        
        
    def load_text(self):
        with open('input.txt', "r", encoding="utf-8") as file:
            
            shakespeare_text = "".join(file.read().lower())
            

        return shakespeare_text
        
    def get_uniq_char(self):
                
        char_counts = Counter(self.text)
        return sorted(char_counts, key=char_counts.get, reverse=True)
    
    def encoder(self, sample):
        return torch.tensor([self.stoi[char] for char in sample])
    
    def decoder(self, sample):
        return ' '.join([self.itos[char] for char in sample])
    
    def __len__(self):
        return len(self.load_text())
    
    def __getitem__(self, index=0):
        
        
        inpt = torch.tensor(self.char_index[index:index+self.args.sequence_length], dtype=torch.long)
        
        target = torch.tensor(self.char_index[index+1:index+self.args.sequence_length+1], dtype=torch.long)
        

            
        return (inpt, target)


class Args():
    def __init__(self, sequence_length, batch_size):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        
args = Args(sequence_length=100, batch_size=256)
dtset = Dataset(args=args)

args.batch_size

dtset.__getitem__(-1)
# dtset.__len__()
dtset.get_uniq_char()
# dtset.char_index[:20]
print(dtset.stoi)
print(dtset.itos)

print(dtset.encoder('first'))
print(dtset.decoder([20,6,9,8,3]))