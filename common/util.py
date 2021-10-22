# -*- coding: utf-8 -*-
import numpy as np

def preprocess(text):
    '''
    text를 corpus로 전처리
    symbols : 처리할 특수문자 리스트
    '''
    text = text.lower()
    symbols = [',', '.', '?', '!']
    for s in symbols:
        text = text.replace(s, ' ' + s + ' ')
    words = text.split(' ')
    words = [i for i in words if i != '']
    
    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
            
    corpus = np.array([word_to_id[w] for w in words])
    
    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size=1):
    '''
    동시발생 행렬(co-occurence matrix)
    corpus내의 각 단어마다 주변에 어떤 단어가 몇번씩 사용되는지 행렬로 표현
    '''
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            idx_left = idx - i
            idx_right = idx + i
            
            if idx_left >= 0:
                left_word_id = corpus[idx_left]
                co_matrix[word_id, left_word_id] += 1
            
            if idx_right < corpus_size:
                right_word_id = corpus[idx_right]
                co_matrix[word_id, right_word_id] += 1
                
    return co_matrix


def cos_similarity(x, y, eps=1e-8):
    '''
    cos similarity 계산
    분모에 eps를 넣어서 0으로 나누기 방지
    '''
    nx = x / np.sqrt(np.sum(x**2) + eps)
    ny = y / np.sqrt(np.sum(y**2) + eps)
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    '''
    query : 검색어(단어)
    query와 유사한 단어를 상위 top(=5)개 만큼 출력하는 함수
    '''
    if query not in word_to_id:
        print(f'{query}를 찾을 수 없습니다.')
        return
    
    print('\n[query]: ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]
    
    # Cosine similarity
    vacab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
        
    #유사도 순으로 출력
    cnt = 0
    for i in (-1*similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(f'{id_to_word[i]}: {round(similarity[i],4)}')
        
        cnt += 1
        if cnt >= top:
            return

def ppmi(C, verbose=False, eps=1e-8):
    '''
    Positive Pointwise Mutual Information
    C : co-occurence matrix
    verbose : 진행상황 출력 여부
    eps : 0으로 나누기 방지
    '''
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0
    
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i,j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)
            
            if verbose: 
                cnt += 1
                if cnt % (total//10) == 0:
                    print(f'{100*cnt/total}% 완료')
    return M


def create_contexts_target(corpus, window_size=1):
    '''
    corpus를 context, target으로 변환
    '''
    target = corpus[window_size:-window_size]
    contexts = []
    
    for i in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[i + t])
        contexts.append(cs)
    
    return np.array(contexts), np.array(target)


def convert_one_hot(corpus, vocab_size):
    '''
    corpus를 one-hot vector로 변환
    '''
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot
