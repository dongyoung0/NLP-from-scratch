import sys
sys.path.append(',,')
from common.np import *  # import numpy as np
from common.layers import Embedding, SigmoidWithLoss
#from embedding import EmbeddingDot
import collections

# class UnigramSampler:
#     '''
#     corpus에서 target에 대한 negative sampling
#     '''
#     def __init__(self, corpus, power, sample_size):
#         self.corpus = corpus
#         self.power = power
#         self.sample_size = sample_size
        
#     def get_negative_sample(self, target):
#         corpus = self.corpus
#         #neg_sample = []
#         batch_size = target.shape[0]
#         neg_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)
#         for i, t in enumerate(target):
#             # words : target을 제외한 단어 집합
#             words = list(range(max(corpus)+1))
#             words.remove(t)
#             # 확률분포 계산
#             p = [list(corpus).count(word)/len(corpus) for word in words]
#             new_p = np.power(p, self.power)
#             new_p /= np.sum(new_p)
#             #neg = np.random.choice(words, self.sample_size, p = new_p, replace=False)
#             neg_sample[i, :] = np.random.choice(words, self.sample_size, p = new_p, replace=False)
            
#         return neg_sample


class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0
                p /= p.sum()
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        else:
            # GPU(cupy）로 계산할 때는 속도를 우선한다.
            # 부정적 예에 타깃이 포함될 수 있다.
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)

        return negative_sample

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

class NegativeSamplingLoss:
    def __init__(self, W, corpus, power = 0.75, sample_size = 5):
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.sample_size = sample_size # 샘플링 횟수
        self.loss_layers = [SigmoidWithLoss() for _ in range (sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads
            
    def forward(self, h, target):
        batch_size = target.shape[0]
        neg_sample = self.sampler.get_negative_sample(target)
        
        # positive
        score = self.embed_dot_layers[0].forward(h, target)
        pos_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, pos_label)

        # negative
        neg_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            neg_target = neg_sample[:, i]
            score = self.embed_dot_layers[1+i].forward(h, neg_target)
            loss += self.loss_layers[1+i].forward(score, neg_label)
            
        return loss
    
    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
            
        return dh