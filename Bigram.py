import torch

#In this model, given a character, we aim to predict the next character that
#is likely to follow the first character
N = torch.zeros((28, 28), dtype=torch.int32) #declares a 28x28 array in pytorch with 
b = {} #we will store the count of all bigrams in names.txt in b
words = open('names.txt', 'r').read().splitlines() #gives us all the words as a python list of strings
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)} #mapping for each letter in the alph, to its pos in alphabet
N=torch.zeros((27, 27), dtype=torch.int32)
#rows = first char of bigram, columns = second char, entries tell us how often second char follows first in dataset
stoi['.']=0

for w in words:
    chs = ['.']+list(w)+['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2]+=1

itos = {i:s for s, i in stoi.items()}
g = torch.Generator().manual_seed(2258574758)
ix =0
P=(N+1).float() #to ensure no 0s in prob dist
P/=P.sum(1, keepdim=True)
for i in range (5):
    cout = []
    while True:
        p = P[ix]
        #here we are iteratively sampling from the model until we reach end
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        cout.append(itos[ix])
        if (ix == 0):
            break
        print(''.joint(cout))
    
#we now evaluate the quality of the model using MLE (and nll)
log_likelihood = 0.0
counter=0
for w in words:
    chs = ['.']+list(w)+['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood+=logprob
        counter+=1
nll=-log_likelihood
avgnll = nll/counter #this is the loss

