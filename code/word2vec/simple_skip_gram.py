import numpy as np
import sys
sys.path.append('..')
from common.layers import MatMul, SoftmaxWithLoss

class SimpleSkipGram:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        # initialize weight
        W_in = 0.01 * np.random.randn(V, H).astype('f') 
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # layer 생성
        self.in_layer = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer1 = SoftmaxWithLoss()
        self.loss_layer2 = SoftmaxWithLoss()
        
        #list에 모으기
        layers = [self.in_layer, self.out_layer, self.loss_layer1, self.loss_layer2]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
            
        # 단어의 분산표현 저장
        self.word_vecs = W_in
        
    def forward(self, contexts, target):
        h = self.in_layer.forward(target)
        score = self.out_layer.forward(h)
        loss = self.loss_layer1.forward(score, contexts[:, 0]) + self.loss_layer2.forward(score, contexts[:, 1])
        return loss
    
    def backward(self, dout=1):
        dscore = self.loss_layer1.backward(dout) + self.loss_layer2.backward(dout)
        dh = self.out_layer.backward(dscore)
        self.in_layer.backward(dh)
        return None