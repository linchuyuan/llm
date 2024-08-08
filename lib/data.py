#!/usr/bin/env python3
import yfinance as yf
import torch
import numpy as np
import pdb

class DataFrame(object):
    def __init__(self, ticker_list, device):
        self.data = None
        for ticker in ticker_list:
            hist = yf.download(ticker, period="60d", interval="2m").to_numpy()
            hist = hist[:, :-1]
            if self.data is None:
                    self.data = hist
            else:
                if len(self.data) > len(hist):
                    self.data = np.concatenate((self.data[:len(hist)], hist), axis=1)
                else:
                    self.data = np.concatenate((self.data, hist[:len(self.data),:]), axis=1)
        self.data = self.data.astype(np.float32)
        self.data = torch.from_numpy(self.data).to(feature_device)
        print("data shape is ", self.data.shape)
        self.feature_device = feature_device
        self.label_device = label_device
        n = int(0.55 * len(self.data))
        self.train_data = self.data[:n]
        self.eval_data = self.data[n:]

    def getBatch(self, batch_size : int, src_block_size: int,
                 tgt_block_size, pred_block_size: int, split='training'):
        if split == "training":
            training_data = self.train_data
        else:
            training_data = self.eval_data

        ix = torch.randint(len(training_data) - src_block_size - pred_block_size, (batch_size,))
        x = torch.stack([ training_data[i:i+src_block_size] for i in ix])
        y = torch.stack(
            [ training_data[
                i+src_block_size-tgt_block_size:i+src_block_size+pred_block_size] for i in ix]
        )
        return x, y

    def getInputWithIx(self, src_block_size: int,
                       tgt_block_size: int, pred_block_size:int, ix: int):
        i = ix
        x = self.data[i:i+src_block_size].unsqueeze(0)
        y = self.data[
            i+src_block_size-tgt_block_size:i+src_block_size+pred_block_size].unsqueeze(0)
        x, y = x.to(self.feature_device), y.to(self.label_device)
        return x, y

    def getLatest(self, src_block_size: int,
                  tgt_block_size: int, pred_block_size:int):
        num_data = len(self.data)
        src_block_start = num_data - src_block_size
        tgt_block_start = num_data - tgt_block_size
        x = self.data[src_block_start:]
        y = torch.ones((tgt_block_size + pred_block_size, len(self.data[0])))
        y[:tgt_block_size,:] = self.data[tgt_block_start:]
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        return x, y

    def raw(self):
        return self.data
