#!/usr/bin/env python
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import torch
import pdb
import os
import math

TICKERS = [
    "SMCI",
    "NVDA",
    "TSM"
]

TOKEN_OFFSET = 12
run_predict = input("run predict (y/n, default n)?:")
if not run_predict:
    run_predict = 'n'
max_gen_token = 100
batch_size = 1
block_size = 1024
n_embed = 2560
n_head = 12
n_layer = 12
lr = 1e-7
epoch = 3e4
cuda0 = "cuda"
cuda1 = "cuda"     

class DataFrame(object):
    def __init__(self):
        self.data = None
        for ticker in TICKERS:
            hist = yf.download(ticker, period="60d", interval="2m").to_numpy()
            if self.data is None:
                    self.data = hist[:, :-1]
            else:
                if len(self.data) > len(hist):
                    self.data = np.concatenate((self.data[:len(hist)], hist[:, :-1]), axis=1)
                else:
                    self.data = np.concatenate((self.data, hist[:len(self.data),:-1]), axis=1)
        self.data = self.data.astype(np.float32)
        self.data = torch.from_numpy(self.data)

    def getBatch(self, batch_size : int, block_size: int, split='training'):
        n = int(0.9 * len(self.data))
        data = self.data[:n]
        eval = self.data[n:]
        if split == "training":
            training_data = data
        else:
            training_data = data
        token_offset = TOKEN_OFFSET
        ix = torch.randint(len(training_data) - block_size - token_offset, (batch_size,))
        x = torch.stack([ training_data[i:i+block_size] for i in ix])
        y = torch.stack([ training_data[i+token_offset:i+block_size+token_offset] for i in ix])
        y = y[:,:,1].unsqueeze(2)
        x, y = x.to(cuda1), y.to(cuda0)
        return x, y

    def getInputWithIx(self, block_size: int, ix: int):
        token_offset = TOKEN_OFFSET
        i = ix
        x = self.data[i:i+block_size].unsqueeze(0)
        y = self.data[i+token_offset:i+block_size+token_offset].unsqueeze(0)
        x, y = x.to(cuda1), y.to(cuda0)
        return x, y

    def raw(self):
        return self.data

class Head(torch.nn.Module):
    def __init__(self, head_size, n_embed, block_size):
        super().__init__()
        self.q = torch.nn.Linear(n_embed, head_size, bias=False)
        self.k = torch.nn.Linear(n_embed, head_size, bias=False)
        self.v = torch.nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, index):
        B,T,C = index.shape
        q = self.q(index) # B, T, head_size
        # print("q weights %s" % torch.sum(self.q.weight))
        k = self.k(index) # B, T, head_size
        w = (q @ k.transpose(-1, -2)) / math.sqrt(C) # [B, T, head_size] @ [B, head_size, T] = [B, T, T]
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # B, T, T
        w = torch.nn.functional.softmax(w, dim=-1) # (B, T, T)
        v = self.v(index) # B, T, head_size
        out = w @ v
        return out

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_head, n_embed, head_size, block_size):
        super().__init__()
        self.heads = torch.nn.ModuleList([ Head(head_size, n_embed, block_size) for _ in range(n_head) ])
        self.proj = torch.nn.Linear(head_size * n_head, n_embed)
        self.dropout = torch.nn.Dropout(0.14)

    def forward(self, index):
        x = torch.cat([h(index) for h in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class FeedFoward(torch.nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        hiddenshape = 512
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embed, hiddenshape),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hiddenshape, n_embed),
        )

    def forward(self, x):
        return self.net(x)

class Block(torch.nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head, block_size):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, n_embed, head_size, block_size)
        self.ffwd = FeedFoward(n_embed)
        self.ln1 = torch.nn.LayerNorm(n_embed, eps=1e-6)
        self.ln2 = torch.nn.LayerNorm(n_embed, eps=1e-6)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=block_size):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Create matrix of positional embeddings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Add positional encoding to input tensor
        x = x / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise RuntimeError("Input sequence length exceeds maximum positional encoding length.")
        
        # Add positional encodings for each time step in input sequence
        x = x + self.pe[:seq_len, :]
        return x

class LLM(torch.nn.Module):

    def __init__(self, block_size, n_features, n_embed, n_head, n_layer):
        self.block_split = 0
        super().__init__()
        self.block_size = block_size
        self.embedding1 = torch.nn.Linear(n_features, n_embed).to(cuda1)
        self.position_embedding_table = PositionalEncoding(n_embed).to(cuda1)
        self.blocks = torch.nn.Sequential(*[Block(
            n_embed, n_head, block_size) for _ in range(n_layer)])
        for i, block in enumerate(self.blocks):
            if i < self.block_split:
                block.to(cuda1)
            else:
                block.to(cuda0)
        self.layernorm = torch.nn.LayerNorm(n_embed, eps=1e-6).to(cuda0)
        self.final_linear1 = torch.nn.Linear(n_embed, n_features).to(cuda0)
        self.final_linear2 = torch.nn.Linear(n_features, 1).to(cuda0)
        self.loss = torch.nn.MSELoss()

    @torch.no_grad() 
    def estimate_loss(self, get_batch, batch_size, block_size, eval_iters=200):
        out = {}
        self.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(batch_size, block_size, split)
                logits, loss = self.forward(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.train()
        return out

    def forward(self, index, targets):
        # B, T
        B, T, C = index.shape
        logits = self.embedding1(index)
        logits = self.position_embedding_table(logits)
        for i, block in enumerate(self.blocks):
            if i < self.block_split:
                logits = block(logits)
            else:
                if i == self.block_split:
                    logits = logits.to(cuda0)
                logits = block(logits)
        logits = logits + self.layernorm(logits)
        logits = self.final_linear1(logits)
        logits = self.final_linear2(logits)
        logits = logits[:,-TOKEN_OFFSET:].flatten()
        if targets is None:
            return logits, None
        targets = targets[:,-TOKEN_OFFSET:].flatten()
        loss = self.loss(logits, targets)
        return logits, loss
        

    @torch.no_grad() 
    def generate(self, index, max_gen_token=500, checkpoint_path=None):
        checkpoint = None
        print("Generating from path %s" %(checkpoint_path))
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise SystemError("No checkpoint available.")
        for i in range(max_gen_token):
            """
            offset = i * TOKEN_OFFSET
            tokens = index[:,offset:,:]
            logits, loss = self.forward(tokens, None)
            logits = logits[:, -TOKEN_OFFSET:, :].to(index.device)
            index = torch.cat((index, logits), dim=1)
            """
            logits, loss = self.forward(index, None)
            return logits
        return index

    def train_and_update(self, get_batch, batch_size, block_size, lr,
                         epoch, eval_interval=1e3, checkpoint_path=None):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        checkpoint = None
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            print("Resume from %s..." % checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            try:
                self.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                del checkpoint
                torch.cuda.empty_cache()
            except:
                print("Unable to restore from previous checkpoint, restaring...")
        else:
            print("Clean run starts")
        print("starting")
        for i in range(int(epoch)):
            if i % eval_interval == 0 and i != 0:
                losses = self.estimate_loss(get_batch, batch_size, block_size)
                print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                torch.save({
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                scheduler.step(losses['val'])  # Step the scheduler with the validation loss

            x, y = get_batch(batch_size, block_size)
            logits, loss = self.forward(x, y)
            print("Loop %s, %s" % (i, loss.item()), end='\r')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

def predict(llm, block_size, data, ix=0, checkpoint_path=None):
    # x, y = data.getBatch(1, block_size)
    x, y = data.getInputWithIx(block_size, 0)
    predict = llm.generate(x, max_gen_token=max_gen_token, checkpoint_path=checkpoint_path)
    print(predict)
    return predict

data = DataFrame()
x, y = data.getBatch(batch_size, block_size)
B, T, C = x.shape
n_features = C
path = "model_nhead%s_nlayer%s_nfeature%s_nembed%s_offset%s.pt" % (n_head,
    n_layer, n_features, n_embed, TOKEN_OFFSET)
llm = LLM(block_size, n_features, n_embed, n_head, n_layer)
if run_predict == 'y':
    ix = 0
    x = predict(llm, block_size, data, ix=ix, checkpoint_path=path)
    # x = x.reshape(block_size)
    predicted_data = torch.tensor(data.raw()[:,1])
    # predicted_data[ix+block_size:ix+block_size+TOKEN_OFFSET] = x[-TOKEN_OFFSET:]
    predicted_data[ix+block_size:ix+block_size+TOKEN_OFFSET] = x
    plt.plot(data.raw()[:,1].cpu().numpy()[ix+block_size-TOKEN_OFFSET:ix+block_size+2*TOKEN_OFFSET], label="Actual")
    plt.plot(predicted_data.cpu().numpy()[ix+block_size-TOKEN_OFFSET:ix+block_size+2*TOKEN_OFFSET], label="Predicted")
    plt.legend()
    plt.show()
else:
    llm.train_and_update(data.getBatch, batch_size, block_size, lr, epoch, checkpoint_path=path)
