#!/usr/bin/env python3
import datetime
import yfinance as yf
import torch
import numpy as np
import pandas
import pytz
import pdb

DB_FILE = "db.pickle"
class DataFrame(object):
    def __init__(self, ticker_list, device):
        self.data = None
        if self.tradingAvailable() or self.data is None:
            for ticker in ticker_list:
                hist = yf.download(ticker, period="60d", interval="2m")
                hist = hist.add_prefix(ticker)
                if self.data is None:
                    self.data = hist
                else:
                    self.data = pandas.merge(
                        self.data, hist, on='Datetime', how='outer')
            self.data = self.data.fillna(0)
            db = self.load()
            if db is not None:
                pdb.set_trace()
                self.data = pandas.concat([db, self.data])
                self.data = self.data[~self.data.index.duplicated(keep='last')]
                self.data = self.data.fillna(0)
            self.dataFlush()
        self.data = self.data.to_numpy()
        self.data = self.data.astype(np.float32)
        self.data = torch.from_numpy(self.data).to(device)
        n = int(1 * len(self.data))
        self.data = self.data[:n]
        print("data shape is ", self.data.shape)
        # n = int(1 * len(self.data))
        # self.train_data = self.data[:n]
        # self.eval_data = self.data[n:]

    @property
    def db(self):
        if not hasattr(self, '_db'):
            self._db = pandas.read_pickle(DB_FILE)
        return self._db

    def load(self):
        try:
            return self.db
        except Exception as ex:
            print("Unable to load history from db: ", ex)
            return None

    def dataFlush(self):
        if self.data is not None:
            self.data.to_pickle(DB_FILE)

    def tradingAvailable(self):
        curr = datetime.datetime.now(pytz.timezone('US/Eastern'))
        start = datetime.time(9, 30)
        end = datetime.time(16, 0)
        if curr.time() < start or curr.time() > end:
            print("Stock outside of trading hours")
            return False
        return True

    def getBatch(self, batch_size : int, src_block_size: int,
                 tgt_block_size, pred_block_size: int, split='training'):
        """ 
        if split == "training":
            training_data = self.train_data
        else:
            training_data = self.eval_data
        """
        training_data = self.data
        ix_range = len(training_data) - src_block_size - pred_block_size
        ix = torch.randint(ix_range, (batch_size,))
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
            i+src_block_size-tgt_block_size:i+src_block_size].unsqueeze(0)
        return x, y

    def getLatest(self, src_block_size: int,
                  tgt_block_size: int):
        num_data = len(self.data)
        src_block_start = num_data - src_block_size
        tgt_block_start = num_data - tgt_block_size
        x = self.data[src_block_start:].unsqueeze(0)
        y = self.data[tgt_block_start:].unsqueeze(0)
        return x, y

    @staticmethod
    def padOnes(token_size, index):
        ones = torch.ones((1, token_size, len(index[0,0]))).to(index.device)
        return torch.concatenate((index, ones), dim=1)      

    def raw(self):
        return self.data
