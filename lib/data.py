#!/usr/bin/env python3
import yfinance as yf
import torch
import numpy as np
import pdb

class DataFrame(object):
    def __init__(self, ticker_list, feature_device, label_device, token_offset):
        self.data = None
        for ticker in ticker_list:
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
        print("data shape is ", self.data.shape)
        self.feature_device = feature_device
        self.label_device = label_device
        self.token_offset = token_offset

    def getBatch(self, batch_size : int, src_block_size: int,
                 tgt_block_size=None, split='training'):
        n = int(0.9 * len(self.data))
        data = self.data[:n]
        eval = self.data[n:]
        if split == "training":
            training_data = data
        else:
            training_data = data
        if tgt_block_size is None:
            tgt_block_size = src_block_size

        token_offset = self.token_offset
        ix = torch.randint(len(training_data) - src_block_size - token_offset, (batch_size,))
        x = torch.stack([ training_data[i:i+src_block_size] for i in ix])
        y = torch.stack(
            [ training_data[
                i+src_block_size-tgt_block_size+token_offset:i+src_block_size+token_offset] for i in ix]
        )
        x, y = x.to(self.feature_device), y.to(self.label_device)
        return x, y

    def getInputWithIx(self, src_block_size: int, tgt_block_size: int, ix: int):
        token_offset = self.token_offset
        i = ix
        x = self.data[i:i+src_block_size].unsqueeze(0)
        y = self.data[
            i+src_block_size-tgt_block_size+token_offset:i+src_block_size+token_offset].unsqueeze(0)
        x, y = x.to(self.feature_device), y.to(self.label_device)
        return x, y

    def raw(self):
        return self.data
