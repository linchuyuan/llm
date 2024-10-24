#!/usr/bin/env python3
import datetime
import yfinance as yf
import torch
import numpy as np
import pandas
import pytz
import pdb
import os

from lib.polygon import getStockHistory

class Tokenizer(object):

    def __init__(self, items:set):
        self.stoi = dict()
        items = sorted(items)
        for v, k in enumerate(items):
            self.stoi[k] = v
 
    def convert(self, string:str):
        return [self.stoi[string]]

    def size(self):
        return len(self.stoi)

class DataFrame(object):
    def __init__(self, ticker_list:list, device:str, is_option:bool, db_file:str):
        self.is_option = is_option
        self.db_file = db_file
        self.device = device
        self.data = None
        self.n_unique_ticker = None
        if 'update_db' in os.environ:
            for ticker in ticker_list:
                hist = getStockHistory(ticker)
                if hist is None or hist.empty:
                    continue
                if not self.is_option:
                    hist = hist.add_prefix(ticker)
                if self.data is None:
                    self.data = hist
                else:
                    if self.is_option:
                        self.data = pandas.concat([hist, self.data])
                        self.data = self.data.sort_index()
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
                # self.data = self.data[~self.data.index.duplicated(keep='last')]
                self.data = self.data.drop_duplicates()
                self.data = self.data.fillna(0)
            self.dataFlush()
        self.addTemporalData(self.data)
        self.data = self.data[self.data.index >= '2024-08-22 11:08:00']
        self.data_frame = self.data
        self.data_option_label = None
        if self.is_option:
            self.data_option_label = self.data['Symbol'].to_list()
            tokenizer = Tokenizer(set(self.data_option_label))
            self.n_unique_ticker = tokenizer.size()
            self.data_option_label = torch.tensor(
                [tokenizer.convert(l) for l in self.data_option_label])
            self.data_option_label = self.data_option_label.to(device)
            self.data = self.data.drop('Symbol', axis=1)
        self.data = self.data.to_numpy()
        self.data = self.data.astype(np.float32)
        self.data = torch.from_numpy(self.data).to(device)
        self.splitTrainingEval()

    def splitTrainingEval(self):
        print("data shape is ", self.data.shape)
        n = int(0.2 * len(self.data))
        self.train_data = self.data[n:]
        print("training shape is ", self.train_data.shape)
        self.eval_data = self.data[:n]
        print("eval shape is ", self.eval_data.shape)

    @property
    def db(self):
        if not hasattr(self, '_db'):
            self._db = pandas.read_pickle(self.db_file)
        return self._db

    def load(self):
        try:
            return self.db
        except Exception as ex:
            print("Unable to load history from db: ", ex)
            return None

    def dataFlush(self):
        if self.data is not None:
            self.data.to_pickle(self.db_file)

    def tradingAvailable(self):
        curr = datetime.datetime.now(pytz.timezone('US/Eastern'))
        start = datetime.time(9, 30)
        end = datetime.time(16, 0)
        if curr.time() < start or curr.time() > end:
            print("Stock outside of trading hours")
            return False
        return True

    """
    align option db and stock db
    """
    def align(self, to, encoder_block_size:int = 5000):
        del self.data
        self.data = list()
        data_option_label = list()
        i = 0
        for index, row in to.data_frame.iterrows():
            print("Process ", index, end='\r')
            buf = self.data_frame[self.data_frame.index <= index].iloc[
                -encoder_block_size:].drop('Symbol', axis=1)
            T, C = buf.shape
            data = np.zeros((encoder_block_size, C), dtype=np.float32)
            data[:T, :] = buf
            self.data.append(data)
            _, C = self.data_option_label.shape
            label = np.zeros((encoder_block_size, C), dtype=np.int32)
            self.data_option_label = self.data_option_label.to('cpu')
            buf = self.data_option_label[i:i+T]
            label[:len(buf)] = buf
            i += 1
            data_option_label.append(label)
        self.data_option_label = torch.from_numpy(
            np.array(data_option_label)).to(self.device)
        self.data = torch.from_numpy(np.array(self.data)).to(self.device)
        # set years to 2020 as default
        years = self.data[:, :, -6]
        years = torch.where(years == 0, years + 2020, years)
        self.data[:, :, -6] = years
        self.splitTrainingEval()

    def getOptionBatch(self, seed:list, split='training'):
        if split == "training":
            training_data = self.train_data
        else:
            training_data = self.eval_data
        x = training_data[:,:,:-6]
        y = training_data[:,:,-6:]
        x = torch.stack([ x[i] for i in seed ])
        y = torch.stack([ y[i] for i in seed ])
        z = torch.stack([ self.data_option_label[i] for i in seed ])
        return x, y, z


    def getBatch(self, batch_size:int, tgt_block_size:int, pred_block_size:int, split='training'):
        if split == "training":
            training_data = self.train_data
        else:
            training_data = self.eval_data
        ix_range = len(training_data) - tgt_block_size - pred_block_size
        ix = torch.randint(ix_range, (batch_size,))
        x = torch.stack([ training_data[i:i+tgt_block_size+pred_block_size] for i in ix])
        return x[:,:,:-6], x[:,:,-6:], ix

    def getInputWithIx(self, tgt_block_size:int, pred_block_size:int, ix:int):
        i = ix
        x = self.data[i:i+tgt_block_size].unsqueeze(0)
        return x[:,:,:-6], x[:,:,-6:]

    def getLatest(self, tgt_block_size: int):
        num_data = len(self.data)
        tgt_block_start = num_data - tgt_block_size
        x = self.data[tgt_block_start:].unsqueeze(0)
        return x[:,:,:-6], x[:,:,-6:]

    def raw(self):
        # is option db
        if len(self.data.shape) == 3:
            return self.data[:, :, :-6], self.data[:, :, -6:]
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
        ones = torch.zeros((1, token_size, len(index[0,0]))).to(index.device)
        return torch.concatenate((index, ones), dim=1)      
