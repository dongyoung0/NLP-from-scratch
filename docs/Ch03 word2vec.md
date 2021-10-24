# Ch 03 word2vec

### 기존 통계 기반 기법의 문제점

- 단어의 개수가 N개일 때 Co-occurence matrix의 크기는 NxN으로 너무 큼.
- SVD를 사용하여 차원 축소를 진행하는데, 이때 SVD를 적용하는 비용도 $O(n^3)$으로 너무 큼.

### 추론 기반 기법

- Neural Network처럼 학습 데이터를 미니배치로 나눠서 순차적으로 학습.

![fig 3-1.png](Ch%2003%20word2vec%20f42969ee61314ded846ab1434b454bff/fig_3-1.png)

## CBOW(Continuous Bag-Of-Words)

- 주변 단어의 맥락(context)이 주어졌을때 무슨 단어(target)가 들어갈지 추측하는 과정.

![fig 3-2.png](Ch%2003%20word2vec%20f42969ee61314ded846ab1434b454bff/fig_3-2.png)

- 학습시 사용한 말뭉치(corpus)에 따라 얻게 되는 단어의 분산 표현이 다름.
- 가중치를 다시 학습할 수 있어서, 단어의 분산표현 갱신이나 새로운 단어 추가를 효육적으로 수행할 수 있음

### 학습 과정

![fig 3-12.png](Ch%2003%20word2vec%20f42969ee61314ded846ab1434b454bff/fig_3-12.png)

1. context, target을 one-hot vector로 변환
2. 해당 vector들을 input 으로 넣음 (이 때 input layer의 개수는 입력시키는 단어의 개수와 같음)
3. hidden layer에서 각 input layer들의 값의 평균을 계산
4. Softmax를 통해 해당 단어가 해당 자리에 나타날 확률을 계산
5. Cross-entropy error를 통해 Loss 계산, 학습

### 가중치

- CBOW 모델에는 가중치가 2개 존재 : W_in, W_out

![fig 3-15.png](Ch%2003%20word2vec%20f42969ee61314ded846ab1434b454bff/fig_3-15.png)

- W_in의 각 행(row)이 각 단어의 분산표현을 나타냄
- W_out에도 각 단어의 의미가 열(column)으로 저장됨
- 이 때 word2vec에서는 보통 W_in만 사용할 때 결과가 좋음(특히 skip-gram)
- GloVE에서는 두 가중치를 더해서 사용했을 때 더 좋은 결과를 얻음

### Skip-gram

- CBOW와 다르게 target으로부터 주변 context를 추측하는 모델

![fig 3-23.png](Ch%2003%20word2vec%20f42969ee61314ded846ab1434b454bff/fig_3-23.png)

- CBOW보다 계산 비용이 크지만 corpus가 커질수록 성능이 뛰어난 경향을 보임.

### 통계 기반 vs 추론 기반

1. 단어의 분산 표현을 수정하고 싶을 때
    - 통계 기반 기법은 co-occurence matrix를 새로 만들어야 하는 반면
    - 추론 기반 기법(word2vec)은 기존 가중치를 초깃값으로 다시 학습하면 되기 때문에 효율적으로 갱신 가능함
2. 성능면에서는 통계 기반 / 추론 기반 기법의 우열을 따지기 어려움
3. 두 기법은 연관이 있음
    - skip-gram과 네거티브 샘플링을 이용한 모델은 co-occurence matrix에 특수한 행렬 분해를 적용한 것과 같음
    - GloVe : 두 기법을 융합한 기법