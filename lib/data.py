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
        if self.tradingAvailable() or True:
            for ticker in ticker_list:
                hist = yf.download(ticker, period="60d", interval="2m")
                if hist is None or hist.empty:
                    continue
                hist = hist.add_prefix(ticker)
                if self.data is None:
                    self.data = hist
                else:
                    self.data = pandas.merge(
                        self.data, hist, on='Datetime', how='outer')
            self.data = self.data.fillna(0)
        if self.data is None:
            self.data = self.load()
        else:
            db = self.load()
            if db is not None:
                self.data = pandas.concat([db, self.data])
                self.data = self.data[~self.data.index.duplicated(keep='last')]
                self.data = self.data.fillna(0)
                self.dataFlush()
        self.addTemporalData(self.data)
        self.datetime = self.data.index
        self.data = self.data.to_numpy()
        self.data = self.data.astype(np.float32)
        self.data = torch.from_numpy(self.data).to(device)
        n = 5200
        print("data shape is ", self.data.shape)
        # n = int(1 * len(self.data))
        self.train_data = self.data[:n]
        self.eval_data = self.data[n:]

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
        if split == "training":
            training_data = self.train_data
        else:
            training_data = self.eval_data
        ix_range = len(training_data) - src_block_size - pred_block_size
        ix = torch.randint(ix_range, (batch_size,))
        x = torch.stack([ training_data[i:i+src_block_size] for i in ix])
        y = torch.stack(
            [ training_data[
                i+src_block_size-tgt_block_size:i+src_block_size+pred_block_size] for i in ix]
        )
        return x[:,:,:-6], x[:,:,-6:], y[:,:,:-6], y[:,:,-6:]

    def getInputWithIx(self, src_block_size: int,
                       tgt_block_size: int, pred_block_size:int, ix: int):
        i = ix
        x = self.data[i:i+src_block_size].unsqueeze(0)
        y = self.data[
            i+src_block_size-tgt_block_size:i+src_block_size].unsqueeze(0)
        return x[:,:,:-6], x[:,:,-6:], y[:,:,:-6], y[:,:,-6:]

    def getLatest(self, src_block_size: int,
                  tgt_block_size: int):
        num_data = len(self.data)
        src_block_start = num_data - src_block_size
        tgt_block_start = num_data - tgt_block_size
        x = self.data[src_block_start:].unsqueeze(0)
        y = self.data[tgt_block_start:].unsqueeze(0)
        return x[:,:,:-6], x[:,:,-6:], y[:,:,:-6], y[:,:,-6:]

    def raw(self):
        return self.data[:,:-6], self.data[:,-6:]

    @staticmethod
    def genTimestamp(start, periods):
        start_datetime = pandas.DataFrame({
            'year': start[0].item(),
            'month': start[1].item(),
            'day': start[3].item(),
            'hour': start[4].item(),
            'minute': start[5].item(),
        }, index=[0])
        start_datetime = pandas.to_datetime(start_datetime).iloc[0]
        valid_timestamps = []
        current_datetime = start_datetime
        while len(valid_timestamps) <= periods + 1:
            sequence = pandas.date_range(
                start=current_datetime, periods=1000, freq='2min')  # Generate a large chunk
            trading_hours_sequence = sequence[
                (sequence.time >= pandas.Timestamp("09:30").time()) &
                (sequence.time < pandas.Timestamp("16:00").time())]
            valid_timestamps.extend(trading_hours_sequence[1:])
            current_datetime = valid_timestamps[-1]
        valid_timestamps = valid_timestamps[:periods]
        buf = pandas.DataFrame(index=valid_timestamps)
        DataFrame.addTemporalData(buf)
        return torch.from_numpy(buf.to_numpy()).unsqueeze(0)

    @staticmethod
    def addTemporalData(data):
        data["year"] = data.index.year
        data["month"] = data.index.month
        data["weekday"] = data.index.weekday
        data["day"] = data.index.day
        data["hour"] = data.index.hour
        data["minute"] = data.index.minute

    @staticmethod
    def padOnes(token_size, index):
        ones = torch.ones((1, token_size, len(index[0,0]))).to(index.device)
        return torch.concatenate((index, ones), dim=1)      
