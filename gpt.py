#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import tiktoken
import pdb
from torch.nn import functional as F

block_size = 16
batch_size = 4
device = 'cuda'


# In[2]:


with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# In[3]:


enc = tiktoken.get_encoding("cl100k_base")
encoder = lambda s: torch.tensor(enc.encode(s)).to(device)
decoder = enc.decode
num_token = int(100e3)
print(encoder("hello world"))


# In[4]:


tensor = torch.tensor(encoder(text), dtype=torch.long)
n = int(0.8 * len(tensor))
data = tensor[:n]
eval = tensor[n:]


# In[5]:


def get_batch(split):
    if split == "training":
        training_data = data
    else:
        training_data = eval
    ix = torch.randint(len(training_data) - block_size - 1, (batch_size,))
    x = torch.stack([ training_data[i:i+block_size] for i in ix])
    y = torch.stack([ training_data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# In[6]:


class Head(torch.nn.Module):
    def __init__(self, head_size, n_embed):
        super().__init__()
        self.q = torch.nn.Linear(n_embed, head_size, bias=False)
        self.k = torch.nn.Linear(n_embed, head_size, bias=False)
        self.v = torch.nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, index):
        B,T,C = index.shape
        q = self.q(index) # B, T, head_size
        k = self.k(index) # B, T, head_size
        w = q @ k.transpose(-1, -2) # [B, T, head_size] @ [B, head_size, T] = [B, T, T]
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # B, T, T
        w = torch.nn.functional.softmax(w, dim=-1) # (B, T, T)
        w = self.dropout(w)
        v = self.v(index) # B, T, head_size
        out = w @ v
        return out

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_head, n_embed, head_size):
        super().__init__()
        self.heads = torch.nn.ModuleList([ Head(head_size, n_embed) for _ in range(n_head) ])
        self.proj = torch.nn.Linear(head_size * n_head, n_embed)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, index):
        x = torch.cat([h(index) for h in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class FeedFoward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            torch.nn.Linear(n_embed, 4 * n_embed),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embed, n_embed),
            torch.nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, n_embed, head_size)
        self.ffwd = FeedFoward(n_embed)
        self.ln1 = torch.nn.LayerNorm(n_embed)
        self.ln2 = torch.nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# In[7]:


class LLM(torch.nn.Module):
    def __init__(self, vocab_size, n_embed, n_head, n_layer):
        super().__init__()
        self.token_table = torch.nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = torch.nn.Embedding(block_size, n_embed)
        self.blocks = torch.nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        self.final_linear = torch.nn.Linear(n_embed, vocab_size)

    @torch.no_grad()
    def estimate_loss(self, eval_iters=200):
        out = {}
        self.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = self(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.train()
        return out

    def forward(self, index, targets):
        # B, T
        B, T = index.shape
        tok_emb = self.token_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x) 
        logits = self.final_linear(x)
        if targets is None:
            return logits, None
        B,T,C = logits.size()
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        try:
            loss = torch.nn.functional.cross_entropy(logits, targets)
        except:
            print(logits)
            print(targets)
            raise
        return logits, loss

    def generate(self, index, max_gen_token=500):
        position_embedding_dim = self.position_embedding_table.embedding_dim
        for i in range(max_gen_token):
            txt = index[:,:position_embedding_dim]
            logits, loss = self.forward(txt, None)
            logits = logits[:, -1, :]
            prob = torch.nn.functional.softmax(logits, dim=-1) 
            index_next = torch.multinomial(prob, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

    def train_and_update(self, get_batch, lr, epoch, eval_interval=1e3):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        for i in range(int(epoch)):
            if i % eval_interval == 0:
                losses = self.estimate_loss()
                print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            x, y = get_batch("training")
            logits, loss = self.forward(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


# In[8]:


llm = LLM(num_token, 384, 16, 1)
llm.to(device)
# llm.train_and_update(get_batch, 3e-4, 1e4)


# In[9]:


context = torch.tensor([0] * 16)
print(context)
context = context[None, :].to(device)
gen = llm.generate(context)
print(decoder(gen[0].tolist()))


# In[14]:


torch.save(llm.state_dict(), "C:/Users/linch/source/repos/linchuyuan/llm/model.pth")

