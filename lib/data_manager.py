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
        self.seed = [i for i in range(T)]
        self.seed = self.seed[(tgt_block_size + pred_block_size):]
        random.seed(1)
        random.shuffle(self.seed)
        random.seed(None)
        n = int(0.2 * len(self.seed))
        self.train_seed = self.seed[n:]
        self.eval_seed = self.seed[:n]
        print("train data size is ", len(self.train_seed))
        print("eval data size is ", len(self.eval_seed))

    def getBatch(self, batch_size:int, src_block_size:int,
                 tgt_block_size:int, pred_block_size:int, split='training'):
        if split == 'training':
            seed = self.train_seed
        else:
            seed = self.eval_seed
        seed = random.sample(seed, batch_size)
        target, target_mark = self.stock_data_frame.getBatch(
            batch_size, tgt_block_size, pred_block_size, seed)
        for i in range(len(seed)):
            seed[i] = seed[i] - pred_block_size -1
        option_data, option_data_mark, ticker = \
            self.option_data_frame.getOptionBatch(seed)
        return option_data, option_data_mark, ticker, target, target_mark

    def getInputWithIx(self, tgt_block_size:int, pred_block_size:int, ix:int):
        option_data, option_data_mark, ticker = self.option_data_frame.getOptionBatch([
            ix-pred_block_size-1], split="training")
        target, target_mark = self.stock_data_frame.getInputWithIx(
            tgt_block_size, pred_block_size, ix)
        return option_data, option_data_mark, ticker, target, target_mark

    def getLatest(self, tgt_block_size:int):
        option_data, option_data_mark, ticker = self.option_data_frame.getOptionBatch([-1])
        target, target_mark = self.stock_data_frame.getLatest(tgt_block_size)
        return option_data, option_data_mark, ticker, target, target_mark

    def raw(self):
        return self.stock_data_frame.raw()
