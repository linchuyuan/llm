#!/usr/bin/env python3

import torch
FORCE_CPU = False

class Config(object):

    def __init__(self, *args, **kargs):
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

    def cuda(self, idx: int):
        cuda_device_count = torch.cuda.device_count()
        if cuda_device_count == 0 or FORCE_CPU:
            return "cpu"
        return "cuda:%s" % (idx % cuda_device_count)

    def decoderOnlyInformerCheckpointPath(self):
        return "model_nhead%s_nlayer%s_nfeature%s_nembed%s_offset%s.pt" % (
            self.n_head,
            self.n_layer,
            self.n_features,
            self.n_embed,
            self.token_offset
        )

    def informerCheckpointPath(self):
        return "model_nDH%s_nDL%s_nDB%s_nEH%s_nEL%s_nEB%s_nembed%s_nfeature%s_offset%s.pt" % (
            self.n_decoder_head,
            self.n_decoder_layer,
            self.n_decoder_block_size,
            self.n_encoder_head,
            self.n_encoder_layer,
            self.n_encoder_block_size,
            self.n_embed,
            self.n_features,
            self.token_offset,
        )
