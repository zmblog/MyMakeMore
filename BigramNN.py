import torch
words = open('names.txt', 'r').read().splitlines() #gives us all the words as a python list of strings
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)} #mapping for each letter in the alph, to its pos in alphabet
N=torch.zeros((27, 27), dtype=torch.int32)
#rows = first char of bigram, columns = second char, entries tell us how often second char follows first in dataset
stoi['.']=0
#training set of bigrams xy
inp, targets = [], []
for w in words:
    chs = ['.']+list(w)+['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        inp.append(ix1)
        targets.append(ix2)
inp = torch.tensor(inp)
targets=torch.tensor(targets)
num = inp.nelement()
#forward pass
import torch.nn.functional as F
#grad descent
for k in range(100):
    InpEncoded=F.one_hot(inp, num_classes=27).float() #represents each char as e^ith vector, based on ith letter in alphabet
    g = torch.Generator().manual_seed(2258574758)
    W=torch.randn((27, 27), generator=g, requires_grad=True)
    # we now implement softmax
    logits = (InpEncoded@W).exp() #mat mult and exp to remove negs, equivalent to N
    prob = logits / logits.sum(1, keepdim=True) #probabilities for next char

    #our goal is to tune W (optimize to find such a W)
    #s.t the probabilities of next character are representative of sample
    #as we are doing classification we use nll instead of MSE (which is for regression)
    loss = -prob[torch.arange(num), targets].log().mean() + 0.01(W**2).mean()
    #backward pass
    W.grad = None #set grad to 0
    loss.backward()
    #update
    W.data+=-50*W.grad

