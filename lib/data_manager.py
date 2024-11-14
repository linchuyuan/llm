from lib.data import DataFrame
import random
import pdb

class DataFrameManager(object):
    
    def __init__(self, stock_data_frame:DataFrame,
            option_data_frame:DataFrame, tgt_block_size,
            pred_block_size):
        self.stock_data_frame = stock_data_frame
        self.option_data_frame = option_data_frame
        self.option_data_frame.align(self.stock_data_frame)
        self.splitTrainingEval(tgt_block_size, pred_block_size)

    def splitTrainingEval(self, tgt_block_size, pred_block_size):
        T, C = self.stock_data_frame.data.shape
        self.ix = [i for i in range(T)]
        self.ix = self.ix[(tgt_block_size + pred_block_size):]
        self.train_ix = self.ix[::5]
        self.eval_ix = [item for i, item in enumerate(self.ix) if i % 5 != 0]
        if len(list(set(self.train_ix) & set(self.eval_ix))):
            raise ValueError("Training and eval set overlapped")
        print("train data size is ", len(self.train_ix))
        print("eval data size is ", len(self.eval_ix))

    def getBatch(self, batch_size:int, src_block_size:int,
                 tgt_block_size:int, pred_block_size:int, split='training'):
        if split == 'training':
            ix = self.train_ix
        else:
            ix = self.eval_ix
        ix = random.sample(ix, batch_size)
        target, target_mark = self.stock_data_frame.getBatch(
            batch_size, tgt_block_size, pred_block_size, ix)
        for i in range(len(ix)):
            ix[i] = ix[i] - pred_block_size -1
        option_data, option_data_mark, ticker = \
            self.option_data_frame.getOptionBatch(ix)
        return option_data, option_data_mark, ticker, target, target_mark

    def getInputWithIx(self, tgt_block_size:int, pred_block_size:int, ix:int):
        option_data, option_data_mark, ticker = self.option_data_frame.getOptionBatch([
            ix-pred_block_size-1])
        target, target_mark = self.stock_data_frame.getInputWithIx(
            tgt_block_size, pred_block_size, ix)
        return option_data, option_data_mark, ticker, target, target_mark

    def getLatest(self, tgt_block_size:int):
        option_data, option_data_mark, ticker = self.option_data_frame.getOptionBatch([-1])
        target, target_mark = self.stock_data_frame.getLatest(tgt_block_size)
        return option_data, option_data_mark, ticker, target, target_mark

    def raw(self):
        return self.stock_data_frame.raw()
