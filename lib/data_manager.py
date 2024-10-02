from lib.data import DataFrame
import pdb

class DataFrameManager(object):
    
    def __init__(self, stock_data_frame:DataFrame, option_data_frame:DataFrame):
        self.stock_data_frame = stock_data_frame
        self.option_data_frame = option_data_frame

    def getBatch(self, batch_size:int, src_block_size:int,
                 tgt_block_size:int, pred_block_size:int, split='training'):
        target, target_mark,= self.stock_data_frame.getBatch(
            batch_size, tgt_block_size, pred_block_size, split)
        option_data, option_data_mark = self.option_data_frame.getOptionBatch(batch_size)
        return option_data, option_data_mark, target, target_mark

