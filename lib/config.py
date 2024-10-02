#!/usr/bin/env python3

import torch
FORCE_CPU = False

dropout = 0.1

class Config(object):

    def __init__(self, config=None, *args, **kargs):
        if config is not None:
            for k, v in config.__dict__.items():
                self.__setitem__(k, v)
        for item in args:
            k = item[0]
            v = item[1]
            self.__setitem__(k, v)

        for k, v in kargs.items():
            self.__setitem__(k, v)

        for i in range(8):
            self.__setitem__(
                "cuda%s" % (i), self.cuda(i))

    def __setitem__(self, k, v):
        self.__dict__[k] = v

    def __str__(self):
        return str(self.__dict__)

    def cpu(self):
        return "cpu"

    def cuda(self, idx: int):
        cuda_device_count = torch.cuda.device_count()
        if cuda_device_count == 0 or FORCE_CPU:
            return "cpu"
        return "cuda:%s" % (idx % cuda_device_count)

    def informerCheckpointPath(self):
        return "model_nDH%s_nDL%s_nDB%s_nEH%s_nEL%s_nEB%s_nembed%s_nfeature%s_predict%s.pt" % (
            self.n_decoder_head,
            self.n_decoder_layer,
            self.n_decoder_block_size,
            self.n_encoder_head,
            self.n_encoder_layer,
            self.n_encoder_block_size,
            self.n_embed,
            self.n_decoder_features,
            self.n_predict_block_size,
        )
