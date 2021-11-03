# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from embedding import Embedding
from negative_sampling import NegativeSamplingLoss

class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size
        
        # initialize weight
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(V, H).astype('f')
        
        # layer 생성
        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.neg_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)
        
        # layer, parameter 정리
        layers = self.in_layers + [self.neg_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
            
        # 단어의 분산 표현 저장
        self.word_vecs = W_in
          
    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
#         h /= len(self.in_layers)
        h *= 1 / len(self.in_layers)
        loss = self.neg_loss.forward(h, target)
        return loss
    
    def backward(self, dout=1):
        dout = self.neg_loss.backward(dout)
#         dout /= len(self.in_layers)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
            
        return None
