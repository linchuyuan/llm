#!/usr/bin/env python3

import data

def test_data():
    data_gen = data.DataFrame(
        ticker_list = ["SMCI"],
        feature_device = "cpu",
        label_device = "cpu",
        token_offset = 1
    )
    feature, label = data_gen.getBatch(batch_size = 1, block_size = 8)
    print("feature is %s, \n label is %s" % (feature, label))

if __name__ == "__main__":
    test_data()
