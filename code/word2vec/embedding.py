# -*- coding: utf-8 -*-
# +
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import Embedding, SigmoidWithLoss


# class Embedding:
#     def __init__(self, W):
#         self.params = [W]
#         self.grads = [np.zeros_like(W)]
#         self.idx = None
        
#     def forward(self, idx):
#         W, = self.params
#         output = W[self.idx]
#         return output
    
#     def backward(self, dout):
#         dW, = self.grads
#         dW[...] = 0 # 
# #         # 좋지 않은 예. idx의 원소가 중복인 경우 문제가 생김
# #         dW[self.idx] = dout 
# #         for i, word_id in enumerate(self.idx):
# #             dW[word_id] += dout[i]
#         np.add.at(dW, self.idx, dout)
#         return None

class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None
        
    def forward(self, h, idx):
        target_W = self.embed.forward(idx) # row 추출
        out = np.sum(target_W * h, axis=1)
        
        self.cache = (h, target_W)
        return out
    
    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh
