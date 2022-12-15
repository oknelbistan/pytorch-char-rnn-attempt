import torch
from collections import Counter
from torch.nn import functional as F
    
        
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
        
        
        inpt = torch.tensor(self.char_index[index:index+self.args.sequence_length])
        target = torch.tensor(self.char_index[index+1:index+self.args.sequence_length+1])

        
        return (inpt, target)


# class Args():
#     def __init__(self, sequence_length):
#         self.sequence_length = sequence_length
        
        
# args = Args(sequence_length=5)
# dtset = Dataset(args=args)


# dtset.__getitem__()
# # dtset.__len__()
# dtset.get_uniq_char()
# # dtset.char_index[:20]
# print(dtset.stoi)
# print(dtset.itos)

# print(dtset.encoder('first'))
# print(dtset.decoder([20,6,9,8,3]))