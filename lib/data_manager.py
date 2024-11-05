from lib.data import DataFrame
import pdb

class DataFrameManager(object):
    
    def __init__(self, stock_data_frame:DataFrame, option_data_frame:DataFrame):
        self.stock_data_frame = stock_data_frame
        self.option_data_frame = option_data_frame
        self.option_data_frame.align(self.stock_data_frame)

    def getBatch(self, batch_size:int, src_block_size:int,
                 tgt_block_size:int, pred_block_size:int, split='training'):
        target, target_mark, seed = self.stock_data_frame.getBatch(
            batch_size, tgt_block_size, pred_block_size, split)
        if isinstance(seed, list):
            for i in range(seed):
                seed[i] = seed[i] - pred_block_size -1
        else:
            seed = seed - pred_block_size -1
        option_data, option_data_mark, ticker = self.option_data_frame.getOptionBatch(
            seed, split)
        return option_data, option_data_mark, ticker, target, target_mark

    def getInputWithIx(self, tgt_block_size:int, pred_block_size:int, ix:int):
        option_data, option_data_mark, ticker = self.option_data_frame.getOptionBatch([ix])
        target, target_mark = self.stock_data_frame.getInputWithIx(
            tgt_block_size, pred_block_size, ix)
        return option_data, option_data_mark, ticker, target, target_mark

    def getLatest(self, tgt_block_size:int):
        option_data, option_data_mark, ticker = self.option_data_frame.getOptionBatch([-1])
        target, target_mark = self.stock_data_frame.getLatest(tgt_block_size)
        return option_data, option_data_mark, ticker, target, target_mark

    def raw(self):
        return self.stock_data_frame.raw()
