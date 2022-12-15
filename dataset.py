import torch
from collections import Counter

class Dataset(torch.utils.data.Dataset):
    def __init__(self,args):
        
        self.args = args
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()
        
        
        self.stoi = {_str : idx + 1 for idx, _str in enumerate(self.uniq_words)}
        self.itos = {idx : _str for _str, idx in self.stoi.items()}
        
        self.words_index = [self.stoi[word] for word in self.words]
        
        
    def load_words(self):
        with open('input.txt', "r", encoding="utf-8") as file:
            
            shakespeare_text = file.read().lower().split()
            
        return shakespeare_text
    
    def padding(self):
        pass
        
    def get_uniq_words(self):
                
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)
    
    def encoder(self, sample):
        return torch.tensor([self.stoi[char] for char in sample])
    
    def decoder(self, sample):
        return ' '.join([self.itos[char] for char in sample])
    
    def __len__(self):
        return len(self.load_words())
    
    def __getitem__(self, index=0):
        
        
        return(
            torch.tensor(self.words_index[index:index+self.args.sequence_length]),
            torch.tensor(self.words_index[index+1:index+self.args.sequence_length+1]),  
        )




class Args():
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        
        
args = Args(sequence_length=11)
dtset = Dataset(args=args)


#dtset.__getitem__()
# dtset.__len__()
#dtset.get_uniq_words()
# dtset.char_index[:20]
#print(dtset.stoi)


# dtset.words_index
print(dtset.encoder('first'))
print(dtset.decoder([20,6,9,8,3]))