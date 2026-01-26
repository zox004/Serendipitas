# **AlphaHoldem 개발을 위한 포괄적 연구 보고서: 심층 기술 분석 및 60일 개발 로드맵**

## **1\. 서론: 불완전 정보 게임과 AlphaHoldem의 패러다임 전환**

텍사스 홀덤(Texas Hold'em), 특히 헤즈업 노리밋(Heads-Up No-Limit, HUNL) 포커는 인공지능(AI) 연구에서 불완전 정보 게임(Imperfect Information Game, IIG)의 성배로 여겨져 왔다. 체스나 바둑과 같은 완전 정보 게임과 달리, 포커는 상대방의 패를 볼 수 없다는 점과 허세(Bluffing)라는 심리적 요소가 결합되어 있어 기존의 탐색 기반 알고리즘으로는 해결하기 어려운 복잡성을 지닌다. Libratus와 DeepStack과 같은 선행 연구들은 이러한 문제를 해결하기 위해 **반사실적 후회 최소화(Counterfactual Regret Minimization, CFR)** 알고리즘에 의존했다.1 CFR은 게임 트리를 순회하며 모든 가능한 상황에서의 후회 값을 계산하여 내쉬 균형(Nash Equilibrium)에 근사하는 전략을 찾아낸다. 그러나 이 방식은 막대한 계산 자원을 필요로 하며, 실시간 의사결정 시에도 서브게임(Subgame) 해결을 위한 반복적인 연산이 요구되어 실제 애플리케이션에 적용하기에는 무리가 있었다.3

이러한 배경 속에서 등장한 **AlphaHoldem**은 기존의 CFR 기반 접근 방식에서 탈피하여, 딥 러닝(Deep Learning)과 강화 학습(Reinforcement Learning, RL)을 결합한 엔드투엔드(End-to-End) 학습 프레임워크를 제안함으로써 새로운 지평을 열었다. AlphaHoldem은 **의사 샴 네트워크(Pseudo-Siamese Network)** 아키텍처를 도입하여 입력 상태 정보로부터 행동(Action)을 직접 추론하며, **Trinal-Clip PPO**라는 새로운 손실 함수를 통해 학습의 안정성을 확보하고, **K-Best Self-Play** 알고리즘으로 다양한 전략을 가진 과거의 자신과 경쟁하며 성능을 고도화한다.1

본 보고서는 AlphaHoldem의 아키텍처를 재현하고 이를 능가하는 고성능 포커 AI를 개발하기 위한 구체적인 60일간의 기술 로드맵을 제시한다. 단순한 작업 목록을 넘어, 각 개발 단계에서 고려해야 할 이론적 배경, 기술적 난제, 그리고 이를 해결하기 위한 엔지니어링 전략을 심층적으로 분석한다. 특히, 단일 GPU 환경에서도 수 밀리초 내에 초인적인 의사결정을 내릴 수 있는 경량화 모델을 구축하는 것을 목표로 한다.1

## ---

**2\. 개발 준비 및 환경 구축 (Week 1-2)**

AlphaHoldem 개발의 첫 단계는 견고한 시뮬레이션 환경을 구축하고, 신경망이 이해할 수 있는 형태의 고차원 상태 표현(State Representation)을 설계하는 것이다. 이는 건물의 기초를 다지는 과정과 같으며, 데이터의 품질과 학습 속도를 결정짓는 핵심적인 단계이다.

### **Day 1: 프로젝트 아키텍처 설계 및 라이브러리 선정**

**목표:** 전체 시스템의 청사진을 그리고, 핵심 라이브러리를 선정하여 개발 환경을 셋업한다. **상세 분석:** AlphaHoldem은 고속의 환경 상호작용과 대규모 병렬 처리를 필요로 한다. 따라서 Python 기반의 RLCard 라이브러리를 기본 환경으로 채택하되, 성능 최적화를 위해 커스텀 래퍼(Wrapper)를 구현해야 한다.4 딥러닝 프레임워크로는 PyTorch를 사용하여 동적인 연산 그래프와 유연한 텐서 조작을 지원하도록 한다. 분산 학습을 위해서는 Ray 프레임워크를 도입하여, 데이터 수집(Actor)과 모델 학습(Learner)을 비동기적으로 처리할 수 있는 구조를 설계해야 한다.

| 구성 요소 | 기술 스택 | 선정 이유 |
| :---- | :---- | :---- |
| **환경 (Environment)** | RLCard (Customized) | 포커 게임의 룰(Rule) 구현이 검증되어 있으며, 상태 표현의 커스터마이징이 용이함.4 |
| **딥러닝 (Deep Learning)** | PyTorch | 복잡한 손실 함수(Trinal-Clip PPO) 구현과 텐서 연산의 직관성 제공. |
| **분산 처리 (Distributed)** | Ray | 단일 머신 내 멀티 코어 활용 및 향후 클러스터 확장성을 고려한 Actor 모델 지원. |
| **데이터 관리** | NumPy, TensorBoard | 고효율 수치 연산 및 학습 과정의 실시간 시각화. |

### **Day 2: RLCard 기반 HUNL 환경 분석 및 벤치마킹**

**목표:** RLCard의 No-Limit Texas Hold'em 환경을 분석하고, 시뮬레이션 속도를 벤치마킹한다.

**기술적 통찰:**

기본 RLCard 환경은 학습용으로 설계되었으나, AlphaHoldem이 요구하는 초고속 롤아웃(Rollout)을 충족시키지 못할 수 있다. env.step() 함수의 내부 로직을 프로파일링(Profiling)하여 병목 구간을 식별해야 한다. 특히, 덱(Deck)을 섞거나 승패를 판정하는 부분에서 불필요한 객체 생성을 최소화하고, NumPy 기반의 연산으로 대체하여 초당 프레임(FPS)을 극대화해야 한다.

* **작업 2-1:** rlcard.envs.nolimitholdem 소스 코드 분석. 딜러, 플레이어, 칩 관리 로직의 흐름 파악.6  
* **작업 2-2:** 랜덤 에이전트(Random Agent)를 이용한 기본 벤치마킹 수행. 단일 코어 기준 초당 핸드 수(Hands/sec) 측정. 목표치는 코어당 1,000 hands/sec 이상이다.  
* **작업 2-3:** 게임 룰의 정확성 검증. 사이드 팟(Side Pot), 스플릿 팟(Split Pot), 올인(All-in) 상황에서의 칩 계산 로직이 정확한지 단위 테스트(Unit Test)를 통해 검증한다.

### **Day 3: 행동 공간(Action Space)의 이산화 및 추상화**

**목표:** 연속적인 베팅 금액을 신경망이 출력할 수 있는 이산적인 행동 공간으로 변환한다. **상세 분석:** 노리밋 홀덤은 이론적으로 무한한 베팅 크기를 가질 수 있지만, 강화 학습의 효율성을 위해 이를 제한된 수의 행동으로 추상화해야 한다. AlphaHoldem 논문과 RLCard 문서를 참고하여 다음과 같은 행동 집합을 정의한다.4

1. **Fold**: 게임 포기.  
2. **Check/Call**: 상대의 베팅에 응수하거나 턴을 넘김.  
3. **Raise Pot**: 팟 크기만큼 베팅.  
4. **Raise Half-Pot**: 팟 크기의 50%만큼 베팅.  
5. **All-In**: 전 재산을 베팅.  
   이러한 5가지 핵심 행동 외에, 전략적 다양성을 위해 0.75 팟, 1.5 팟 등을 추가할 수 있으나, 초기 단계에서는 복잡도를 낮추기 위해 필수적인 행동만으로 구성한다.

### **Day 4: 보상 함수(Reward Function) 설계**

**목표:** 에이전트의 학습 신호가 될 보상 함수를 정의한다. **기술적 통찰:** 단순한 승패(Win/Loss)는 분산이 너무 크다. 100BB(Big Blind)를 잃는 것과 1BB를 잃는 것은 질적으로 다르다. 따라서 \*\*핸드당 획득한 빅 블라인드(mbb/h, milli-big-blinds per hand)\*\*를 기준으로 보상을 설정해야 한다.4 또한, 학습 초기에는 0이 아닌 보상을 자주 제공하여 학습을 가속화하기 위해, 팟의 변화량(Potential-based Reward Shaping)을 고려할 수 있으나, 이는 최적 정책을 왜곡할 수 있으므로 주의해야 한다. 최종적으로는 정규화된 칩의 획득량을 보상으로 사용한다.

### **Day 5: 병렬 환경 래퍼(Vectorized Environment Wrapper) 구현**

**목표:** 다수의 게임 환경을 동시에 실행하여 데이터 수집 효율을 극대화하는 VectorEnv 클래스를 구현한다.

**상세 분석:**

PPO 알고리즘은 온-폴리시(On-Policy) 방식이므로, 현재 정책으로 생성된 대량의 데이터가 필요하다. Python의 multiprocessing이나 Ray를 사용하여 ![][image1]개의 환경을 병렬로 실행하고, 각 환경의 상태를 하나의 배치(Batch) 묶어서 반환하는 래퍼를 작성한다.

* **구현 포인트:** 프로세스 간 통신 오버헤드를 줄이기 위해 공유 메모리(Shared Memory)를 활용하거나, Ray의 Object Store를 적극 활용한다.  
* **목표 성능:** 단일 머신에서 32개 이상의 환경을 동시에 구동하여 초당 수만 개의 트랜지션(Transition)을 생성.

### **Day 6: 상태 표현(State Representation) \- 카드 정보 인코딩**

**목표:** AlphaHoldem의 핵심 기여 중 하나인 "추상화 없는 상태 표현"을 위한 텐서 구조를 설계한다.2 **기술적 통찰:** 기존 AI는 카드를 클러스터링하여 추상화했지만, AlphaHoldem은 원시 카드 정보를 그대로 사용한다. 이를 위해 **원-핫 인코딩(One-Hot Encoding)** 기반의 텐서를 설계한다.

* **차원 설계:** ![][image2]장의 카드를 ![][image3] (무늬 ![][image4] 숫자) 매트릭스로 표현한다.  
* **채널 분리:** 내 패(Private Hand), 플랍(Flop), 턴(Turn), 리버(River)를 각각 다른 채널에 할당하여 시간적 정보를 보존한다. 예를 들어, ![][image5] 형태의 텐서에서 ![][image6]는 각 라운드의 카드 정보를 담는 채널이 된다.3

### **Day 7: 상태 표현 \- 베팅 히스토리 인코딩**

**목표:** 베팅의 흐름과 액션의 순서를 텐서에 인코딩한다.

**상세 분석:**

포커에서 "누가, 언제, 얼마를 베팅했는가"는 카드 정보만큼이나 중요하다. AlphaHoldem은 이를 이미지처럼 처리하기 위해 베팅 히스토리를 텐서의 추가적인 채널로 인코딩한다.

* **구현 방안:** 각 베팅 라운드(프리플랍, 플랍, 턴, 리버)별로 플레이어와 상대방의 액션을 기록하는 매트릭스를 생성한다. 예를 들어, 칩의 양을 정규화하여 해당 위치에 값을 채워 넣거나, 액션의 유형을 원-핫으로 인코딩하여 채널을 구성한다. 이는 CNN이 베팅 패턴(예: 체크-레이즈)을 시각적 패턴으로 인식하게 한다.2

### **Day 8: 통합 텐서 구조 설계 및 구현**

**목표:** 카드 정보와 베팅 히스토리를 결합하여 최종 입력 텐서를 완성한다.

**기술적 통찰:**

AlphaHoldem의 입력 텐서는 ![][image7] 형태를 띤다. 여기서 ![][image8] (Suits), ![][image9] (Ranks)로 고정하고, 모든 정보를 ![][image10] 차원으로 쌓아 올린다(Stacking).

* **채널 구성(예시):**  
  * Ch 0-1: 내 패 (2장)  
  * Ch 2-4: 커뮤니티 카드 (플랍, 턴, 리버)  
  * Ch 5-N: 각 라운드별 베팅 히스토리 (내 액션, 상대 액션)  
  * Ch M: 현재 팟 사이즈 및 스택 사이즈 (전체 영역에 동일한 값으로 브로드캐스팅).3 이러한 고차원 텐서 표현은 의사 샴 네트워크가 공간적 특징(Spatial Features)을 추출하는 데 최적화되어 있다.

### **Day 9: 데이터 전처리 및 정규화 파이프라인**

**목표:** 신경망 학습의 안정성을 위해 모든 입력 데이터를 정규화(Normalization)한다.

**상세 분석:**

칩 개수나 팟 사이즈와 같은 수치형 데이터는 그 범위가 매우 넓다(0 \~ 수천). 이를 그대로 신경망에 넣으면 그라디언트 폭주가 발생할 수 있다. 따라서 모든 칩 관련 수치를 빅 블라인드(BB) 단위로 나누거나, 로그 스케일로 변환하여 $$ 또는 ![][image11] 범위로 정규화하는 전처리 파이프라인을 구축한다.

### **Day 10: 상태 표현 검증 및 시각화 도구 개발**

**목표:** 설계한 텐서가 실제 게임 상태를 정확하게 반영하는지 검증한다.

**작업 내용:**

* 랜덤하게 생성된 게임 상태를 텐서로 변환한 후, 이를 다시 사람이 읽을 수 있는 형태(텍스트 로그)로 복원하는 '역변환(Decode)' 기능을 구현한다.  
* Matplotlib을 사용하여 텐서의 각 채널을 히트맵(Heatmap)으로 시각화한다. 카드가 있는 위치에 1이 표시되는지, 베팅 금액이 올바른 채널에 인코딩되었는지 육안으로 확인한다. 이는 디버깅의 핵심 도구가 된다.

## ---

**3\. 신경망 아키텍처 및 알고리즘 구현 (Week 3-4)**

환경과 데이터 파이프라인이 준비되었다면, 이제 AlphaHoldem의 두뇌에 해당하는 \*\*의사 샴 네트워크(Pseudo-Siamese Network)\*\*와 이를 학습시킬 **Trinal-Clip PPO**를 구현할 차례이다. 이 단계는 연구 논문의 수식을 코드로 변환하는 고난도 작업이 포함된다.

### **Day 11: 의사 샴 네트워크(Pseudo-Siamese Network) 개요 및 설계**

**목표:** AlphaHoldem의 독창적인 신경망 구조인 의사 샴 네트워크의 전체 구조를 설계한다. **이론적 배경:** 일반적인 샴 네트워크는 두 개의 입력에 대해 동일한 가중치(Shared Weights)를 갖는 네트워크를 사용한다. 그러나 포커에서는 '나의 정보(Private Info)'와 '공개 정보(Public Info)'의 성격이 다르다. 의사 샴 네트워크는 이 두 정보를 처리하는 초기 레이어(Stem)의 구조는 유사하지만 가중치를 공유하지 않거나, 혹은 입력을 합치기 전에 서로 다른 경로로 처리하는 구조를 의미한다.9

* **구조:** 입력 텐서를 받아 특징을 추출하는 **ResNet Backbone**과, 추출된 특징을 바탕으로 정책(Policy)과 가치(Value)를 출력하는 **Dual Head** 구조로 나뉜다.

### **Day 12: ResNet 백본(Backbone) 구현**

**목표:** 깊은 층에서도 학습이 잘 되도록 잔차 연결(Residual Connection)이 포함된 컨볼루션 네트워크를 구현한다.

**상세 분석:**

AlphaHoldem은 1D 벡터가 아닌 2D/3D 텐서를 입력으로 받으므로, Conv2d 레이어를 핵심으로 사용한다.

* **ResBlock 구현:** Conv2d \-\> BatchNorm \-\> ReLU \-\> Conv2d \-\> BatchNorm \-\> Add Input \-\> ReLU 구조의 블록을 정의한다.  
* **다운샘플링 전략:** 포커 보드 크기(![][image3])는 작기 때문에, 풀링(Pooling)을 과도하게 사용하면 정보가 손실된다. 스트라이드(Stride)를 조절하여 공간 정보를 최대한 보존하거나, 다일레이션(Dilation) 컨볼루션을 사용하여 수용 영역(Receptive Field)을 넓힌다.3

### **Day 13: 어텐션 모듈 (CBAM) 통합**

**목표:** 중요한 정보(예: 현재 보드에 깔린 에이스 카드)에 가중치를 부여하기 위해 CBAM(Convolutional Block Attention Module)을 통합한다.9 **기술적 통찰:** 단순한 CNN은 모든 위치를 동일하게 처리하지만, 포커에서는 특정 카드의 조합(플러시 가능성 등)이 훨씬 중요하다. CBAM은 \*\*채널 어텐션(Channel Attention)\*\*과 \*\*공간 어텐션(Spatial Attention)\*\*을 순차적으로 적용하여, 네트워크가 "어떤 채널(어떤 종류의 정보)이 중요한지", "어떤 위치(어떤 카드)가 중요한지"를 스스로 학습하게 돕는다. 이를 ResBlock 사이사이에 삽입하여 특징 추출 능력을 극대화한다.

### **Day 14: 정책 헤드(Policy Head) 및 가치 헤드(Value Head) 구현**

**목표:** 백본에서 추출된 특징 벡터를 받아 최종 행동 확률과 승률 예측값을 출력하는 헤드를 구현한다.

**구현 방안:**

백본의 출력(Feature Map)을 Flatten 하거나 GlobalAveragePooling을 통해 1D 벡터로 변환한다. 이후 두 갈래로 나뉜다:

1. **Policy Head:** FC Layer \-\> ReLU \-\> FC Layer \-\> Softmax. 출력 차원은 행동의 개수(예: 5\~10개).  
2. **Value Head:** FC Layer \-\> ReLU \-\> FC Layer \-\> Tanh. 출력은 ![][image11] 사이의 값(기대 수익).  
* **가중치 초기화:** Orthogonal Initialization을 적용하여 학습 초기의 안정성을 확보한다.

### **Day 15: 모델 파라미터 수 최적화 및 순전파(Forward Pass) 테스트**

**목표:** 모델의 크기를 조절하여 실시간 추론(Inference) 속도를 확보한다. **상세 분석:** AlphaHoldem은 단일 GPU에서 2.9ms 내에 추론을 마쳐야 한다.1 모델이 너무 크면 연산 비용이 증가하고, 너무 작으면 표현력이 부족하다. 파라미터 수를 100만\~500만 개 수준으로 유지하며, 더미 데이터(Dummy Data)를 통과시켜 추론 속도를 측정한다. 필요시 채널 수(Width)나 레이어 수(Depth)를 조정한다.

### **Day 16: PPO (Proximal Policy Optimization) 기본 구현**

**목표:** 가장 기본적인 PPO 알고리즘을 구현하여 베이스라인을 잡는다. **이론적 배경:** PPO는 정책의 변화폭을 클리핑(Clipping)하여 과도한 업데이트를 방지하는 알고리즘이다. 기본 손실 함수는 다음과 같다 11:

![][image12]  
여기서 $r\_t(\\theta)$는 확률 비율, ![][image13]는 어드밴티지 함수이다.

### **Day 17: 일반화된 어드밴티지 추정(GAE) 구현**

**목표:** 분산과 편향의 트레이드오프를 조절하여 안정적인 학습을 돕는 GAE(Generalized Advantage Estimation)를 구현한다. **상세 분석:** 포커는 보상(Reward)이 희소(Sparse)하고 지연(Delayed)되어 나타난다(핸드가 끝나야 결과가 나옴). GAE는 ![][image14] 파라미터를 통해 ![][image15]\-step 리턴을 지수적으로 가중 평균하여 어드밴티지를 계산한다.12

![][image16]  
이때 $\\delta\_t^V \= r\_t \+ \\gamma V(s\_{t+1}) \- V(s\_t)$이다. 올바른 마스킹(Masking) 처리를 통해 게임이 끝난 시점 이후의 보상이 섞이지 않도록 주의해야 한다.

### **Day 18: Trinal-Clip PPO 구현 \- 이론 및 수식**

**목표:** AlphaHoldem의 핵심인 Trinal-Clip 손실 함수를 코드로 구현한다. **심층 분석:** 일반 PPO(Dual-Clip 포함)는 정책 비율만 클리핑하지만, Trinal-Clip은 \*\*가치 함수(Value Function)\*\*의 오차까지 클리핑 범위에 포함하거나, 어드밴티지가 음수일 때(즉, 예상보다 나쁜 결과가 나왔을 때)의 업데이트를 더욱 보수적으로 제한하는 세 번째 클리핑 메커니즘을 도입한다.3

* **필요성:** 포커에서는 올바른 플레이를 했음에도 운에 의해 질 수 있다(Bad Beat). 이때 가치 함수가 너무 크게 업데이트되면 정책이 망가질 수 있다. Trinal-Clip은 이러한 "운에 의한 큰 손실"이 학습을 방해하는 것을 막아준다.  
* **구현:** 가치 손실 함수에 clip 연산을 추가한다:  
  ![][image17]

### **Day 19: 손실 함수 통합 및 역전파(Backpropagation) 파이프라인**

**목표:** 정책 손실, 가치 손실, 엔트로피 보너스를 결합한 최종 목적 함수를 정의한다.

**작업 내용:**

**![][image18]**

* **![][image19]**: Trinal-Clip이 적용된 정책 손실.  
* ![][image20]: 클리핑된 가치 손실 (Mean Squared Error 기반).  
* ![][image21]: 정책의 다양성을 유지하기 위한 엔트로피 보너스. 포커에서는 초기에 다양한 전략을 탐색하는 것이 중요하므로 ![][image22] 값을 적절히 설정해야 한다 (예: 0.01).13  
* loss.backward()를 호출하고 optimizer.step()으로 가중치를 업데이트하는 전체 루프를 완성한다.

### **Day 20: Leduc Hold'em을 이용한 알고리즘 검증**

**목표:** 복잡한 HUNL에 적용하기 전에, 단순화된 포커 게임인 Leduc Hold'em에서 알고리즘이 내쉬 균형에 수렴하는지 확인한다. **기술적 통찰:** Leduc Hold'em은 카드 수가 적어 최적 전략을 계산하기 쉽다.14 구현한 AlphaHoldem 에이전트가 Leduc 환경에서 "착취 가능성(Exploitability)"이 0에 가깝게 줄어드는지 확인한다. 만약 여기서 수렴하지 않는다면, HUNL에서도 실패할 것이 자명하므로 디버깅(하이퍼파라미터 튜닝, GAE 계산 오류 수정 등)에 집중해야 한다.

## ---

**4\. Self-Play 시스템 및 대규모 학습 (Week 5-7)**

강화 학습 에이전트가 초인적인 성능을 내기 위해서는 강력한 상대와의 대전이 필수적이다. **K-Best Self-Play**는 이를 위한 핵심 메커니즘이다.

### **Day 21: 모델 풀(Model Pool) 아키텍처 구축**

**목표:** 과거의 에이전트 모델들을 저장하고 관리하는 저장소를 구축한다. **상세 분석:** 단순히 최신 모델하고만 대결하면 전략이 순환(Cyclic)하거나 특정 전략에만 과적합(Overfitting)될 수 있다. 이를 방지하기 위해 학습 과정에서 일정 주기(예: 100회 업데이트)마다 모델의 스냅샷(Checkpoint)을 저장하고, 이를 '모델 풀'에 등록한다.3

### **Day 22: K-Best 선정 알고리즘 및 Elo Rating 시스템**

**목표:** 모델 풀 내의 에이전트들 간의 우열을 가리기 위한 평가 시스템을 만든다. **기술적 통찰:** 풀에 있는 모든 모델을 상대로 학습할 수는 없다. 가장 강력한(Best) ![][image23]개의 모델을 선별해야 한다. 이를 위해 백그라운드에서 모델끼리 리그전을 벌이고, Elo Rating 시스템을 도입하여 점수를 매긴다. 점수가 높은 상위 ![][image23]개의 모델이 학습 상대로 선정될 확률을 높인다.15

* **K값 설정:** 초기에는 작게 시작하여(K=1), 풀이 커질수록 늘려나간다(K=5\~10).

### **Day 23: 상대 모델 샘플링 및 로딩 메커니즘**

**목표:** 학습 환경(Environment)이 시작될 때마다 모델 풀에서 상대를 랜덤하게 로드하는 기능을 구현한다.

**구현 방안:**

Ray의 Actor 기능을 활용하여, 각 환경 워커(Worker)가 에피소드 시작 시 ModelPool 액터에게 "상대 모델 가중치"를 요청하도록 한다.

* **확률적 샘플링:** ![][image24]. 즉, 강한 상대일수록 더 자주 선택되도록 하되, 약한 상대(과거의 자신)도 가끔 선택하여 기본적인 플레이 방법을 잊지 않도록(Catastrophic Forgetting 방지) 한다.16

### **Day 24: 비동기 학습 아키텍처 (Actor-Learner) 최적화**

**목표:** 데이터 수집(CPU)과 모델 업데이트(GPU)의 속도 차이를 극복하기 위한 비동기 파이프라인을 구축한다.

**상세 분석:**

단일 GPU를 사용하는 경우, GPU가 쉬지 않고 돌게 하는 것이 핵심이다.

* **구조:** 다수의 Actor들이 롤아웃 데이터를 공유 큐(Shared Queue)에 밀어 넣는다(Push). Learner는 큐에서 배치를 꺼내(Pull) 학습하고, 최신 가중치를 Parameter Server(또는 Ray Object Store)에 게시한다. Actor들은 주기적으로 가중치를 동기화한다.3

### **Day 25: GPU 배치 추론(Batch Inference) 구현**

**목표:** 개별 환경에서의 추론 요청을 하나로 묶어 처리함으로써 GPU 활용률을 극대화한다. **기술적 통찰:** 각 Actor가 개별적으로 신경망을 호출하면 오버헤드가 크다. "Inference Server"를 두어, 여러 Actor의 상태 입력을 모아 큰 배치(Large Batch)로 만든 뒤 GPU에 한 번에 태우고, 결과를 다시 각 Actor에게 분배하는 방식을 사용한다. 이는 전체 시스템의 처리량(Throughput)을 10배 이상 향상시킬 수 있다.17

### **Day 26-30: 하이퍼파라미터 튜닝 및 초기 학습**

**목표:** 본격적인 장기 학습에 앞서 학습률(Learning Rate), 클리핑 비율(![][image25]), 배치 크기(Batch Size), GAE 계수(![][image14]) 등을 최적화한다.

**작업 내용:**

* **Day 26:** 학습률 스케줄러(Learning Rate Scheduler) 도입. 초기에는 웜업(Warm-up) 후 선형 감쇠(Linear Decay) 또는 코사인 어닐링(Cosine Annealing) 적용.  
* **Day 27:** 엔트로피 계수 튜닝. 정책 붕괴(Policy Collapse \- 항상 Fold만 하거나 All-in만 하는 현상)를 막기 위해 엔트로피 보너스를 적절히 조절한다.  
* **Day 28:** 배치 크기 실험. 너무 작으면 노이즈가 심하고, 너무 크면 학습이 느리다. 4096\~16384 사이에서 최적점을 찾는다.  
* **Day 29:** 24시간 연속 학습 테스트. 메모리 누수(Memory Leak) 여부 점검.18  
* **Day 30:** 초기 모델 성능 평가. 랜덤 에이전트를 압도적으로 이기는지 확인.

### **Day 31-40: 집중 학습 (Intensive Training) 및 커리큘럼 러닝**

**목표:** 수십억 핸드(Billions of hands) 규모의 대규모 학습을 수행한다.

**상세 분석:**

* **커리큘럼 러닝:** 처음에는 스택 사이즈가 작은 게임(Short Stack)부터 시작하여 점차 딥 스택(Deep Stack)으로 난이도를 높여가는 방식을 고려할 수 있다. 이는 복잡한 의사결정 공간을 점진적으로 탐색하게 돕는다.  
* **로그 모니터링:** TensorBoard를 통해 정책 손실, 가치 손실, 승률(mbb/h)을 실시간으로 감시한다. 가치 손실이 급증하면 학습이 불안정하다는 신호이므로 학습률을 낮추거나 클리핑을 강화한다.

### **Day 41-45: 전략 분석 및 디버깅**

**목표:** 학습된 에이전트의 플레이 스타일을 분석하고 약점을 보완한다.

**작업 내용:**

* **Day 41:** 프리플랍 레인지(Pre-flop Range) 시각화. 에이전트가 어떤 핸드로 오픈(Open), 3-Bet, 콜(Call)을 하는지 히트맵으로 그려서 포커 이론(GTO)과 비교한다.  
* **Day 42:** 사후 분석(Post-mortem). 에이전트가 크게 패배한 핸드들을 추출하여 로그를 분석한다. 명백한 실수(예: 너츠를 들고 폴드)가 없는지 확인한다.  
* **Day 43:** 착취 전략 대응. 특정 패턴(예: 지나친 블러핑)을 보이는지 확인하고, 이를 교정하기 위해 모델 풀에 '카운터 전략'을 가진 에이전트를 인위적으로 추가할 수 있다.

## ---

**5\. 최적화 및 최종 평가 (Week 8-9)**

학습된 모델을 실제 배포 가능한 수준으로 경량화하고, 공인된 벤치마크(Slumbot 등)를 통해 성능을 검증한다.

### **Day 46: 모델 경량화 및 증류(Distillation)**

**목표:** 거대한 학습용 모델의 지식을 작은 추론용 모델로 압축한다. **기술적 통찰:** 학습 시에는 표현력이 좋은 큰 모델(Teacher)을 사용하고, 실제 게임(서비스) 시에는 이를 모방하는 작은 모델(Student)을 사용한다. Student 모델은 Teacher의 출력(Softmax 분포)을 따라 하도록 학습(Knowledge Distillation)되므로, 훨씬 적은 파라미터로도 비슷한 성능을 낼 수 있으며 추론 속도는 빨라진다.1

### **Day 47: 추론 엔진 최적화 (C++ Integration 고려)**

**목표:** Python의 오버헤드를 줄여 2.9ms라는 극한의 속도를 달성한다. **작업 내용:** 모델 가중치를 ONNX(Open Neural Network Exchange)나 TensorRT로 변환하여 C++ 기반의 추론 엔진에서 실행하는 것을 테스트한다. Python 환경에서도 TorchScript를 활용하여 JIT(Just-In-Time) 컴파일을 적용하면 속도를 비약적으로 높일 수 있다.19

### **Day 48-50: Slumbot 대전 및 성능 평가**

**목표:** 2018년 컴퓨터 포커 대회 우승자인 Slumbot과 대결하여 객관적인 성능 지표를 확보한다.3 **상세 분석:**

* Slumbot API를 연동하여 에이전트와 대결시킨다.  
* 최소 10,000 핸드 이상을 플레이하여 통계적 유의성을 확보한다.  
* AlphaHoldem 논문에서는 100,000 핸드 기준 Slumbot을 상대로 양의 mbb/h를 기록했다.3 이를 재현하는 것이 최종 목표이다.

### **Day 51: 로컬 베스트 리스폰스(LBR) 테스트**

**목표:** 에이전트의 착취 가능성을 근사적으로 측정한다. **기술적 통찰:** LBR은 에이전트의 전략을 고정한 상태에서, 국소적으로 최적의 대응(Best Response)을 찾는 탐색 알고리즘을 돌려 에이전트가 얼마나 취약한지 평가하는 방법이다.20 LBR 수치가 낮을수록 내쉬 균형에 가깝다는 의미이다.

### **Day 52-55: 최종 하이퍼파라미터 튜닝 및 재학습**

**목표:** 평가 결과를 바탕으로 미세 조정(Fine-tuning)을 수행한다.

**작업 내용:**

평가에서 발견된 약점(예: 리버 블러핑 빈도가 너무 낮음 등)을 보완하기 위해 보상 함수를 미세하게 조정하거나, 특정 상황(리버 상황)의 데이터를 더 많이 수집하도록 커리큘럼을 수정하여 짧게 재학습한다.

### **Day 56: 코드 리팩토링 및 문서화**

**목표:** 프로젝트의 지속 가능성을 위해 코드를 정리하고 문서를 작성한다.

**작업 내용:**

* 연구용 스파게티 코드를 모듈화된 구조로 정리한다.  
* 각 클래스와 함수의 Docstring을 작성한다.  
* 설치 및 실행 가이드(README)를 작성하고, 재현 가능한 실험 스크립트를 포함시킨다.

### **Day 57-59: 최종 100,000 핸드 대규모 벤치마크**

**목표:** 최종 모델의 성능을 확정 짓는 대규모 실험을 수행한다. **상세 분석:** Slumbot 뿐만 아니라, DeepStack(가능하다면), 그리고 자체적으로 학습된 다른 버전의 AlphaHoldem들과 리그전을 벌인다. 이 결과를 표(Table)로 정리하여 보고서의 결론을 뒷받침할 데이터로 삼는다. 95% 신뢰 구간을 반드시 명시해야 한다.3

### **Day 60: 프로젝트 회고 및 향후 연구 방향 수립**

**목표:** 60일간의 여정을 마무리하고 성과를 분석한다.

**내용:**

* 목표 달성 여부(승률, 속도, 자원 사용량) 평가.  
* 기술적 한계(예: 6인용 테이블로의 확장성 부족 등) 분석.  
* 향후 연구 과제(예: 상대방 모델링(Opponent Modeling) 모듈 추가 등) 도출.

## ---

**6\. 결론**

본 보고서는 AlphaHoldem의 핵심 기술인 **의사 샴 네트워크**, **Trinal-Clip PPO**, **K-Best Self-Play**를 기반으로, 실제 구현 가능한 수준의 상세 로드맵을 제시하였다. 이 로드맵은 단순한 구현을 넘어, 포커 AI가 갖추어야 할 고속 추론, 분산 학습, 그리고 안정적인 수렴성을 달성하기 위한 공학적 해법을 담고 있다.

제시된 60일의 일정은 도전적이지만, RLCard와 Ray 같은 현대적인 오픈소스 도구를 적절히 활용하고, 본 보고서에서 분석한 기술적 난제(분산 관리, 손실 함수 안정화)에 미리 대비한다면 충분히 달성 가능하다. 성공적으로 개발된 AlphaHoldem은 단일 GPU만으로도 세계 최고 수준의 포커 AI 성능을 보여줄 것이며, 이는 거대 자본이 필요한 AI 연구의 장벽을 낮추고 불완전 정보 게임 연구의 새로운 표준을 제시할 것이다.

| 주요 성과 지표 (KPI) | 목표치 | 비고 |
| :---- | :---- | :---- |
| **승률 (Win Rate)** | \> 50 mbb/h (vs Slumbot) | 10만 핸드 기준 3 |
| **추론 속도 (Latency)** | \< 4ms | 단일 GPU 배치 추론 시 |
| **학습 속도** | 3일 내 수렴 | 1 PC 기준 (논문 재현) 1 |
| **하드웨어 요구사항** | Consumer GPU (RTX 3080급) | 클라우드 클러스터 불필요 |

이로써 AlphaHoldem 개발을 위한 연구 보고서를 마친다. 이 로드맵이 차세대 게임 AI 개발의 나침반이 되기를 기대한다.

#### **참고 자료**

1. AlphaHoldem: High-Performance Artificial Intelligence for Heads-Up No-Limit Poker via End-to-End Reinforcement Learning, 1월 22, 2026에 액세스, [https://ojs.aaai.org/index.php/AAAI/article/view/20394](https://ojs.aaai.org/index.php/AAAI/article/view/20394)  
2. Papers \- AAAI2022, 1월 22, 2026에 액세스, [https://aaai-2022.virtualchair.net/papers.html?filter=authors\&search=%20University%20of%20Chinese%20Academy%20of%20Sciences)Institute%20of%20Automation](https://aaai-2022.virtualchair.net/papers.html?filter=authors&search=+University+of+Chinese+Academy+of+Sciences\)Institute+of+Automation)  
3. AlphaHoldem: High-Performance Artificial Intelligence for Heads-Up No-Limit Poker via End-to-End Reinforcement Learning \- AAAI, 1월 22, 2026에 액세스, [https://cdn.aaai.org/ojs/20394/20394-13-24407-1-2-20220628.pdf](https://cdn.aaai.org/ojs/20394/20394-13-24407-1-2-20220628.pdf)  
4. Games in RLCard, 1월 22, 2026에 액세스, [https://rlcard.org/games.html](https://rlcard.org/games.html)  
5. RLCard: A Toolkit for Reinforcement Learning in Card Games — RLcard 0.0.1 documentation, 1월 22, 2026에 액세스, [https://rlcard.org/](https://rlcard.org/)  
6. rlcard.games.nolimitholdem, 1월 22, 2026에 액세스, [https://rlcard.org/rlcard.games.nolimitholdem.html](https://rlcard.org/rlcard.games.nolimitholdem.html)  
7. AlphaHoldem: High-Performance Artificial Intelligence for Heads-Up No-Limit Poker via End-to-End Reinforcement Learning \- AAAI-2022, 1월 22, 2026에 액세스, [https://aaai-2022.virtualchair.net/poster\_aaai2268](https://aaai-2022.virtualchair.net/poster_aaai2268)  
8. Papers \- AAAI2022, 1월 22, 2026에 액세스, [https://aaai-2022.virtualchair.net/papers.html?filter=authors\&search=%20Renye%20Yan](https://aaai-2022.virtualchair.net/papers.html?filter=authors&search=+Renye+Yan)  
9. A Multi-Scale Pseudo-Siamese Network with an Attention Mechanism for Classification of Hyperspectral and LiDAR Data \- MDPI, 1월 22, 2026에 액세스, [https://www.mdpi.com/2072-4292/15/5/1283](https://www.mdpi.com/2072-4292/15/5/1283)  
10. Cross-Modal Pseudo-Siamese Networks \- Emergent Mind, 1월 22, 2026에 액세스, [https://www.emergentmind.com/topics/cross-modal-pseudo-siamese-network](https://www.emergentmind.com/topics/cross-modal-pseudo-siamese-network)  
11. Proximal Policy Optimization — Spinning Up documentation \- OpenAI, 1월 22, 2026에 액세스, [https://spinningup.openai.com/en/latest/algorithms/ppo.html](https://spinningup.openai.com/en/latest/algorithms/ppo.html)  
12. ChengTsang/PPO-clip-and-PPO-penalty-on-Atari-Domain \- GitHub, 1월 22, 2026에 액세스, [https://github.com/ChengTsang/PPO-clip-and-PPO-penalty-on-Atari-Domain](https://github.com/ChengTsang/PPO-clip-and-PPO-penalty-on-Atari-Domain)  
13. (PDF) Alternative Loss Functions in AlphaZero-like Self-play \- ResearchGate, 1월 22, 2026에 액세스, [https://www.researchgate.net/publication/339406650\_Alternative\_Loss\_Functions\_in\_AlphaZero-like\_Self-play](https://www.researchgate.net/publication/339406650_Alternative_Loss_Functions_in_AlphaZero-like_Self-play)  
14. RLCard: Building Your Own Poker AI in 3 Steps | Towards Data Science, 1월 22, 2026에 액세스, [https://towardsdatascience.com/rlcard-building-your-own-poker-ai-in-3-steps-398aa864a0db/](https://towardsdatascience.com/rlcard-building-your-own-poker-ai-in-3-steps-398aa864a0db/)  
15. A Survey on Self-play Methods in Reinforcement Learning \- arXiv, 1월 22, 2026에 액세스, [https://arxiv.org/pdf/2408.01072](https://arxiv.org/pdf/2408.01072)  
16. Self Play And Autocurricula In The Age Of Agents | Amplify Partners, 1월 22, 2026에 액세스, [https://www.amplifypartners.com/blog-posts/self-play-and-autocurricula-in-the-age-of-agents](https://www.amplifypartners.com/blog-posts/self-play-and-autocurricula-in-the-age-of-agents)  
17. Open source code for AlphaFold 2\. \- GitHub, 1월 22, 2026에 액세스, [https://github.com/google-deepmind/alphafold](https://github.com/google-deepmind/alphafold)  
18. bupticybee/AlphaNLHoldem: An unoffical implementation of AlphaHoldem. 1v1 nl-holdem AI. \- GitHub, 1월 22, 2026에 액세스, [https://github.com/bupticybee/AlphaNLHoldem](https://github.com/bupticybee/AlphaNLHoldem)  
19. pokerengine \- PyPI, 1월 22, 2026에 액세스, [https://pypi.org/project/pokerengine/](https://pypi.org/project/pokerengine/)  
20. RLCard: A Toolkit for Reinforcement Learning in Card Games \- Daochen Zha, 1월 22, 2026에 액세스, [https://dczha.com/files/rlcard-a-toolkit.pdf](https://dczha.com/files/rlcard-a-toolkit.pdf)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAYAAAD3Va0xAAAA3UlEQVR4XmNgGAWkgnlA/BmI/0PxAhRZCPjLgJAHYWdUaVSArBAb2AfEKuiC6IARiLcD8XoGiEFBqNJggMsCFJAPxCZQNi5X/UEXwAbeIrE/MEAM4kMSUwPiTiQ+ToDsAlA4gPg3kcSWATEPEh8rAIXPZjQxdO9h8yoGQA4fZDGQ5m4o/xeSHE7wDl0ACmCu0gbiFjQ5rACXs3czQOTuATEnmhwGYAHiveiCUMDEgBlWWAEzEL8B4pPoEkjgGxB/RxdEBquA+CMDJP2A0g0oL2ED+kCcjS44CkYBEAAABi803bhnVOIAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAXCAYAAADpwXTaAAABE0lEQVR4XmNgGAXkgkIgXgnEWlC+BhAvAeICuAoE2AXE/4H4PRAnocmBQQsDRAEyPo+iAgJA4qxQdiSU/wohDQH1QDwJiBcCcQ0QM6NKg8F2IN6AJraVAWKgH7IgyABHZAEs4A8DRGMIkpg2VOwlkhhDNQNhw6SBeBGamAMDxLCzyIKVQNwKlZgPpWciK8ABdjJA1OogCxZBJZABSFEzmhgyAIUrSM1pdAlsABaruMB3BjTvwQAjugAQ/GXAbdhVIF6FLggDIE1vsYhhM2wNAyR8kcFjZA5IUymyAFQM3bAyIM5EExMH4qnIAt+AWBSJ78AAMUgdScwZKoYNuyOpAwOQN5EVKKFKYxiAjJmQ1I2CYQEAobFNJdu5G+wAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAXCAYAAACrggdNAAABnUlEQVR4Xu2VO0sEMRSFr4WVKL5tFFsbGwvBQsHKX2Djn9DK/QNWgmArWIiNlWAnWIhgKdhY2aqIhYWCL3zfQ5KdmztkNlnZbcwHBzInh2ROZnaWKJPJtINx1o8220gvVe+/yvomk9nxp8IgXLVoKxhiXVGxd2j/S9asuH6mcLbOEUUGW0hVKT23bK/3hOcxyjpg3VN4Uc2SNhSdrG5tNkDfuAQ+Dt1Rs96G8DzcQimlRliP2rR0sd61GUFVKQ32RrZDT4B91pgdp5QCeMJPykOhD+XFEltqnUxuUk+AAdaxuE4tBXAg7rVAoU8xl0qjUsOsTdYp64bM/ZfQCzRTCrhifykEGpWSrJHJzklzizUhDWq+VD+ZV+5VTySSUgq/pVL+kMxjlHIhjLeLaCWuEBhkvYi5VEo3aZkh408rP5T3iAoJ+qj8UUAx+elNIbT/BRn/RPmhvEdUyNLDetOmBcX0VzGG0P5TVD68XTLZBeXXOWfdsa6tMD7zEmVWtKHAHy/KxfDAuqVif4zhSRbJlMBhfdnxvJfIZDKZf8kv68J+KGtOfGoAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAAAdklEQVR4XmNgGAVDEEgCMRO6IBrwRBdABv+BmAVdEAp+ATEXuiA6ABnAiib2G4h50MRwAmQDQBp5keSIAiADQBr50CWIAf+gmB1dghAAaYL5EcRmQ5LDC5A1IouhByIG+MuAOzpABuCKRgZlBsLOi0UXGAX0AgD/zhE4ENMyggAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAE0AAAAYCAYAAAC/SnD0AAAC0klEQVR4Xu2Yy8tNURjGn9zvhWRCoZARBiRKTDARycjAlxATDOQy8Be4lEsihiZGyMBtIjEyUkpicD65hMjAPdf3sfZ21nnstc46++zvfOk7v3o75zzPu9+1v3evtfbeH9ClS3+yQoUBxDwVUthusV/FAgZbrLU4Y7FRvMXy+3/jlwoxplq8UFGYC1f0p8Vei1kWmzJtePY54W926xy3uK9ii7xSwWOSRQ3uPB/Bnb8y3eKdiiFYaISKHlfgcnapkUGvpaskjII7vkzT7qE+fugc5lvc9H5vgcvd4Wk5Hy1Wq6hwSX1V0eMI3AAcOMRbuBlYFh5btmk5lxFu2hc4b6SnhZo8E8V6A98Q3ss2wBXg/hXjosUxFRM5bLEAfdu08/jXCzWNUOfsD6JXwCdW2Oe6xXgVE+Be+DD73pdNU7i6mHtQjYyYh7EID7QNzrutRoVwlud0qmkL4fJuqeFxA5EtaznCAz2F81aqUREHLJZ5vzvRtEMW5+DyesTzOY1IrfyRoQjqIa9dBlk8Ea0TTcsZCpfbK3rOPkRqsdshM7VpJyxGq9gE3taVTjaN5H/fbDWMPYjUWoSwWUPY8/mhQgLcJzU4FpvJ70vqqcnEmsa6d0X7DpfPbUI5hXCtP3e8kDkNzjspus8DuKleBN8ahqgYITTTUuuEmjYR9Vnlk2tzRCfXLD6p6MMDh6mYcQHOX68G3I1ihooZO+GOe69GBOY/E62VOnfgcvkYo1DnK1LOuEyreZoPvaMq+jBht4oea1C/Knwv4+elhoxieEJcAs3g0uF7Ly8Cm/Yaja8xzep8tngJdyxrPLd4Y7HZy+E/GZjHc+/NPq96vkI/+tzJ6f9BxYrg4FVQVZ0UpiBxPCal7ButwGVyVsUSVFUnFa6mdSoWscrisYptUuauWkRVdVKYDLfUk+G71lYV22CMCiWpqk4KSctS6VFhALFUhS5d+off1zjPteJ6/BUAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAYCAYAAAAlBadpAAAAsUlEQVR4XmNgGJZAGogLgHgmECshiVshsTHAYiD+D8S3gdgbiFWBeBoQPwdiS6gcVgCS+AfE/OgSQFDJAJG/hC4BAn8Y8JgKBSD5IHTBD1AJTnQJNIBhuC5U8Ba6BBaAofkvVBCbPwkCkEYME4kFZGtmZoBofIkugQVgtYAYmy2AOAFdEATuMkA0g1yBDYDEX6ELIgOQZlAiQTfACIhfo4lhBbsZEF74CqVTUVSMgqEIAG1gK0HBSgf2AAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAASMAAAAYCAYAAACx6pv3AAAKlElEQVR4Xu2bB6wkRxGGi2hyzjLckbEJBoFMEPiOZIKJNkHEExKYIMDI5JwxJouchM+AjWSC4UQOwgiBEEjkIOKJDEaYZHLsT911U/tvz2zP7tvjvfN8Uuvt/NU909PTobpmntnExMTExMTExMQwR6owMTEx0cgNVViWR6T0FBW3CBdI6TAVN4hLpnRsSm9O6fCgXz78ds6V0jVV3MKcP6XrqXgOg+d/oLbBpVM6RMUBWsbZf1UYy5VT+oWKgd9YvoinP6T025T+ErRD9+Xev/zVujpsJM+0fE7u/T4pbS/a31O6XLFFziqa6luVX9r674c+9zfrrvOvlH5SbA+22TYlnVlsY/hmShdVsYF7pPQfy9f9itg2iuMt96d4j+8ptjdYbo9oI7/zb7E5J6X0oHBcg8XVy+0RWx+t4+yqKf1OxTFwAWa9Id5iOd951GBtlVwE5e+sYiOU/bqKK+ADpObl3NyyjY6qfMNWb4fNBPfyDxU3mPPa8PNbpW891nJZPIBlofzRKlYgz7L1HLrHIdvPUzo4HOOZk/fFQRuCvDetaC8RzRl6TpE/p3RXFVu4heXZeRG+StT4omXbbdXQyPUtlz+3Ghqh7D1VXJJfW/99OthfraJl/WsqbmG4nxepuMEcZ/k6d1NDAdu6J8Q+jrHFfcHx3cMyUK6vbJ/tIimdouIIHmnz5/XJrC/20zrOWMT13E3809piRZwct7lGX4O1crotX36VFUn5quVzXVsNAnkupaJl/e4qblFwt7mfC6thg2Hr1ff8WCixPV8N+wm8gL66KeR7v4qN9I2f71q/jRDJKtQWXV8YaowdZ+S9kIqLoNAFVRQubjnf49SQOM2y7dFqKGxP6VUp3UV0uKPlrRnl2RodldKtZnJkuP5rUnpZSpcQW+wwzPbvSOnqnbmZgyyfh336ImoPJT4sJiTa5QGdeQ7u6YWWYyM1aC/azblSSq9I6YSgOdT93Tbrcj80pZNTukrQFOrw0pSeazlYHdlt9ft0uBZBfeIPq9A32OBTNtw/h+p/f+vfbji0Ed5FbWEBrv1lyxPy8yx7wzFMQbvTf70PP6n8Pl/I00KtDdhavrXHRttr3yLIfpIN7y6ebV2f4pyfLL95i069fYKi72nIZOw4I++i9p+BwJ7eaA0mAfLhflGBa6V0B8teFd6SThDOj1P6Uvl9X+uCYM7xlh8g2qnl+HbBDuyLf2RdI2t9Of5jSnstd8jLFO26MVMD1JVyy2412Z5RnsnM31D4fSmfsPzg4To23y7UnXbxON2elJ5TbB8pWsQnUHRiWj+wHIu5QtHwciK0JSvrhy23GQNMz8lx3/bo2ym9vvwm1tgygffBdX5mOcbAALhTSNi0XrCo/gTf2cYQ5P1M0B0GOvmPKMcEXDnWrTfat6zbDtE/43XYzqDxPNH5TRrLT23+PhlbUGsDjVeeaLm9XmDzeYExhc6kzsTr5/SFnzo/sWiEXDh+SLE52MaMs49bW/hnH7e2euUVj9zf3vJgJTEZ8aC4YM2VjzOpw/GfRPN4EftVBW+JN3eOu60Rjn1gR23UrGzdA1qWWnlvt4hPMBEte3b563lpd4dBG/PSoXwfj65vRdGeUdE0tlWrE51bYdWPeT8nx2N4lOWyeL2PD+kJKT252Godeqj+vBl+WNCon4JOcNvxyekaQeNtGhoD3WESRNPtxyrxIvi05fLuUTG2uD5orJY3uurt+hvI2guUQ4vGX4dJV/N5vIjxWAPbmHH2Rpu/xiDMfi0FyNP3uk4bC/gGB+3hoqM9VbT3Fl15umWdFT4SJy22Q7WyaEy0Y6BM7VytUJZPAFTTlRntXRXtY+GYQeq6eidsx2I9PfDLtbX+3sFuHDTc65gPL/eHNhuo3mY5jw46uIllG95qbes9hjNtvs6Ox4vYHkUW1X9n+esr92XLseOfLESY/FTjdb5qD6xogPY+FUfgiw5eMrCIOXuLzdHFhme8vfwm3ys70z5N68xuQ7XHVDRnmXHmi0kzu2xxAXfreGA12FNjZ3vg8G2Hnndb0XRyQeObJaXWiIpfO8IEqFoLLdcDnRyACUHL+kQQ4zivLZoGyLX9HPQ4SbimKxSwddI6MPGr5vfJ6oh954w18zabLxfxBchTjKOMYajNcfOxabyopf5Q6xuApv0N71vzcoznH6kNYn/Oh4k+Bl4gcQ68XmJgfMfmENfBxseXn7X+mFCtD7KYoNUWvzNEG/Luam25aJz5tq+Zm9niArjQ5OnrcNj0HDWtr4Oj1QLj6Ky+Q5BHOwwDpebaL6JWZ4UOsVtFy3ExLVvbjvn3SxEC2KqBx3uIfTgeG+mbuL5Q0eI21zXdKivk6WtD90xZVPBayOsf6o3Btzx09Bp9z6Ol/kA+jQEBOl63anjoqt27osVtG/DiplbPMdzL8jmI1Wiff1Ox7bDhdmaR9DiTc6zlsrXFTz0atL63gdjGjjM+2hzVLgyuRQWw9+Vx9+3zoqNpw6AxGP038AaA3z7b38a6ADb67vI7EgcneWodhjcpEN3dG9hs7EXxyVL345F4vkjtYaF9r/x2T6bmMbLfd41ArlN7mK8LGm8i6cQOOrGGCNqR5bd3HDRiC0qM+5HHPTK2NY5/+hDh+GjRCL6rB6z41mjs90Ut9Y/bCryDOCmh7wjHvp2jLzq0rd5nPCfe2tvLb/UodPJvaYsbWT6HXhPcw6jZItiPKb/9tb9PchHeILpGbO1q1nl3bMGBeSHGC7G1jjPno7bE5weclOh4DV+9tIHReRjYPiQ2eJrNNsKecny65dXdZ+Cjiu7EuJS76RHeGPy+/CaIrnbiB7GhvXEBXfMr/laLgGYEL0Un1whlvCNEjbdDBCU/WDSPgzgeF/P7pmM76D55O2db94Djebi23pvHdoD9uwcwCdxq3lvabNtjx2veZbNeGPoJ4fiQokX8/Iu8F+5Dyzr3s2x7uRqsrf7vtC4P8aQI/cdfaV/Rcj49H9sh1U4N2t6go/kicpzlujutbXExy/lOVoN1k6DGXyME3r1uvADgfA46QX0gJhXv1yd7JvJ4v/Ft3dhx5mCPn6Y0QSFuQGEg+NugmNCwsbcd+p+fUyzn58a2WfdVpq523y86q67yAeuue5Z1bxiAG9XBCh6gxEWNEN3/js2/5lbcIyHRcfn7rJkcs7C/J4/ik7VuQ9wjIO20bl+v5+A4ej6wveikGMinzrpggK/afFYR8eAiiUHMYI749y0MSgUvz8ueMWvax17rf91Pm/KKmOdJYqB6n8CL4Jg6Edchb+08i+rv21mSD8QIfZO+g9fI9bTt8SJPE80XZlIMiuMleQxtV9CdobaIaB0cHzeLoC3JxwQY2VZ0EiEX4P44PtwzWReb+lXQYOw4c7DhYY2CDsCKe06A15D++nRivbQMoM0A9cSTWidbpS02ioNthXum4KJ97YGAemUT6+EgywH8zQbeA3E35wjLfT/GITeazdoW6wRvVeOIzRCwY7t0IEN8ZoeKE2uhFtD8f+P/beAfiBIL45iA7jrZjG2xTvjGULd6o2ELQzDqQCV+XTuxXtbpaawCgVa+ueGlC2+p9gebtS3WxdLbM2WXChMTExONsO2dmJiYmJiYmJhYmv8BWwp//0EZw0QAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAYCAYAAABa1LWYAAABV0lEQVR4Xu2Vvy4FQRTGP1Eh8Q63UXoA4gGQXIVEKSoKiUbiAeg09wEIQeEhFAqVRLSiRoH4EyQohO/knJHrZHf2Kpgl80t+uXfnm03mZGbOAplM5ifZoA/0vc1Ly7rolctu6ZjlKZE1T/pBT1h0EfvQrN8HiZiDridalOyITDr2gRErOAXX6KCoeeikpg8MyV79YCLu7beyKLlDZTsxAs2WfZCAaTpj/yuLCsdrgo5DG8GoeWhZz+fsdEhzCHRU1ClddC5ZVraLRQzSnRK36RbdhHbddbqmr1Uip0nufiBaVLhPZW26DvdJTs+CG4sWFTpJEUPQbMUHv8yzH0BFUbHjtQfNen0QoUFXv2kVB84j6LpO7PkL3dDwr3yfAgOI7FQLGk75ANrt6lrUMHRds+2Du/SR3tEb6AftzbI++mRjksmcF2i7rwOyngt6Rs/tOZPJZDL/hw9FqXDmwf9DSQAAAABJRU5ErkJggg==>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEIAAAAYCAYAAABOQSt5AAAB2ElEQVR4Xu2Wu0v1QBDFB9TCQkFBVGzstPMfUESw+grtLVSw0MpCSytB8NEJdoofvgoRK1uxsxAUBEXQRnwgvhoFH/ieYXavk0myJI1JsT845O45c+/dTDbZAHg8Ho+bItQD6lvo3mSNqCeVnZiMuFRZl8iyYFcbghLUHvA8b1BtwfiXZ+CiKOyJRjGAGtbmH7IDwYsRRTnqXIybgGvXhVdgH+J/yPUn79rIiCGIn+MWcNYivNhzWgMOapXfjfoymWYGVanNjHA1glYtZVXCi23EOIS7Rryitk1WrLIjNc4SVyM0pcC1dFuF6AMOe4W3gipDLZusQWSH4nMeSNqIeuC6C+UXaAUuGBMePWWJUZP9M+MK4NsiLUsxWkQtoP6j5lFzqFnznaQkacQk8O9T3YjKCtQBF6ya8ZnIekw2aMZvIssLSRohodoXbVoopFVQg5oQfrPJplHtqE6R5YW0jbDvTv06ICh4RH0qn3YSyjYisjRMpVQaXI24guAKJzaB6+kYggISXXWNzap1kBNcjbBzl9i3YtpaQ1AQd/9TdqrNHEEriOYo3xUsH6gO5VGt8xlBO0IUuqN5gS7cNfAVpi2RjnfAu5HkFvgcjs3xIBh7PB6Px+NJyQ/pT5WR+f+FDQAAAABJRU5ErkJggg==>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEsAAAAYCAYAAACyVACzAAADCElEQVR4Xu2XS6iNURTHV97PJM8YXCmPPEpGSKHEgCQTE4WBRJJuSsSAgZTII0JCSZIBGShJSRhg4JkiopQ8yivv5/rfvdY56/y/3b0XE0ffr/6db//W/p5nn72/I1JSUvKPM1CzQrNXMzj4CWHbaacZwbKO6aIZwzLHYc1PzX3NDM0QzW7NU814q0XemGNfr7yUVt4POvzQ9OCCslpS/SYXJO2D/C/gPi+yjHyTlp8m6nNYSvLbWdYxuJ9pLJ3Xkjp05gKRe5i9JfmeXKhTFkr+PpsYLal4jwsZcgfZIVU/T3NMM6laLjBIs00zk7yzVLMhtLFw7NesDc7pqzkq6ZgAC02jZo+mq7kcmGY2a9ZrOlDtkeTvs4nvkoq5eao1YK7C/m81vTQdrb04djIea67a9lzNR6m9sCWSVlt8ce81dyV90+CZpP6RJ5JWbZ8ezpufYo5po/mgOS3pIbWVYj+0L5GrgCLv8DtgXzxwdlfIYWHg86D9jtoAKzG2B4TaFnPOOk0fqT6sc6EG4CZm3I2M4/ZUchX+5mFhnsK+/cjDYYg7qMPxaIPDKuvMCp5XI4yseJ3L7BOvNXz9I81h5Dj+SuQM1TzQbAxuvhSPVcGHIS6kJXIH2SpFP8xcnAtum4s0mMM8w8DzagR3nByA/0rujPkI2gjmWHxBk2uqiYdS3K8GP0hzjNMsYCnV+S7iP6FI7hwHMg5gLmM/ylwn8gB+U8bhgbGLP/kc6MMjugYMRXSKQzYC/5ylgf12Zdy+sO2f/O3DfQrbDiZ1flh3gsOD8bmsv/lu1gZYIeG6W/uWfcL5diSumuiDEY2FAKtsFnTCSyk/sLGaF+QcvJNhP0yyEThcwHDNSnNrzDunrH1C0jlPhhr8hdB25yMl/lPYabUIzukOq157214evIMF4FVoo47rOSjNv3rIWUmdESzb+FxU06OW2VI8ObgmyR8if8Q8brZB0n9OtL/ETubiH3cw3fxn8hiF18kBHBP98fONrDKP4CHxannZanhPKykpKSkpKSn5U34BEezhjuE4KOgAAAAASUVORK5CYII=>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADYAAAAXCAYAAABAtbxOAAABZUlEQVR4Xu2XvS4FURSFt0ahEYlKr0AIovEIRPQeASGEx/AWCoVKq9qFylMQGrwACuyTvY+77zLjzjFn3HuZL1mZZK2ZyV7JOfND1DK8bJlWMRhgpqkzdymvohnRFAYDzATpzNcYeJ7RyMSxaBvNH3BKWqIIRsOTs9gZ6Qp4N+10x5W5Er1R5z6z3fEnjIYnZzFPnWKRZWqLfaUtlsifLraLZiKx2BwGBqPhKSq2kqAywkB7aCYSi81jYDAanqJimwkqIwy0j2YisdgCBgaj4SkqloMw0AGaicRiixgYjIanyWKHaBprVL5vPLHYEgYGo+Fpotgk6UAnGAgjpFlQL9ZJz9vAwGA0PDmLnYueRPeiOzs+kH5meS5EN+B5Hkmv8/cJ3q0/iX6xWArffplXhNHw9KtYlaXYC0bD049il6T7sC6MhudFNC4aw6BBwh9wHUZJZw6/N6UcmcIjeFgI77U49//gA/MLYlUtQuD6AAAAAElFTkSuQmCC>

[image12]: <data:image/png;base64,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>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAYCAYAAAD6S912AAABBElEQVR4XmNgIAxMgPg/EJehS5ADLIH4G5R9Hoi7kOTIAmvQ+NPR+IMHSALxP3RBSgDIMFDEUAWkMkAMo5qBMMOoYuBeIFYA4hcMVDCQE4gfQNn7GKhgICxBg8BUBoiB6khiJAFnIO5B4hcwQAwMQRIjCaB7zxsq1ogmDgMs6ALIYC4Qf0LDnxkgBq5DUgcDIJ9MQBeEASYg/oguCAUgA++gCzJAEr0wuiAMvAFiVnRBKEBPi1lIYiCcgSQHdtlRqAQ2ALIE3UAQEGDAksfDGCDefAvEX4H4N6o0WANMHkSD5Iuhcn0MeMKPHPCXAU/4kQNgQRCPIkoBAOX1g0Csgy4xggAA0kBBjqHDSgoAAAAASUVORK5CYII=>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAZCAYAAADnstS2AAAAhUlEQVR4XmNgGPrgPxA/AGIWNHGcAKQBhIkCExggijnQJXABkOKN6IK4AElOyWCAKFZGl8AFQIpvoQtiA+IMJDjlOhDfZYAoxhvmD4B4JRAzM0AUL0ORRQIvGVDdidMpH4D4O5pYIQNEsRSy4GeoIDYAEr8E48hABUBuxAb2MOA2aBQQBwC0ciJVn07c0AAAAABJRU5ErkJggg==>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAXCAYAAAA/ZK6/AAAAlUlEQVR4XmNgGAWDCegC8Twg5obyeYG4AYgnADETVAwO2IF4KxBHA/F/IG4G4gVQuXqoGArYC6VhGhqR5EA2YWgohdLXGDAls7GIwQFIoh2L2GU0MTCQYIBIgpwAA3xQMQUofypCCsJBtxpZrBqIlZDkGP4C8VdkASAoYIBo0AfiS2hyDBZAzIouyACJHwN0wVFACAAA3qgdBAlcrcAAAAAASUVORK5CYII=>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAA2CAYAAAB6H8WdAAAFoElEQVR4Xu3dW6htVRkH8GF2tYtZmWSlpmalqAVC9ZRJQZlaDwli+lCEkeANFQp9ih5MS8US6UJY0YWgonzoIUiMepASKerBJKUgb6WRElamNf7NOVtjj9Zee23d52z3Xr8ffMwxx5znnLn05WNcvlEKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA7Dj71Li2xo9qvKbGBTWOrfHW9iUAALbPh5v2P2sc39y/rGkDALBN3ty0f1HjHzWeVYaRt2c2zwAA2EZfqvGTsf3sGj+r8YXZYwAAAAAAAGD3em6Nq2q8vsZ7umfxyab9phrvbOKk5tmnmvZ6zujuP9PdAwCstCRYn+/6zqrxobF9SI3vNs8iCdXDXd+jTfun4/XLTd8iWejfJnn7N+2tkt+Qb/5TjXubuG/s+0uNv82Jt+QPAwBsl5PLkJgdOka8qMav//fGoB9he12Nfzf3B9W4eWw/o8bZY3ta0B9HNu15/tjd79vdP1VJCh8vw3envZHUY/trjSf6B1vswDLsUF3kszXOK7PvfqDGS2ePAYBV8JymnZpkU2KQMhfRJjiXjNc2YUtCcXSNd4/3SdpiqnV269iX5PCAsa/35zIkfpPLmvZWyTfku9tv38iryzDl2zq8u9+MK/qO6iN9R/W8pn1wjc+N7eOafgBghbQJzD1NO6UukkhNCdjlZSgqm3hseqnMRqEyKvbaMkv0zh2vk9Q3S+mMXk4cSILSTs3269q2yi/L8Ht/3z9Y4FVNO/9NvtPcrydJbpLfPkE9qqz9bR9v2pNbyvBevnMa+bxrvL53vAIAK+SOsjZhS+KUtV3xjTIb2TlivI8kMEnYcjpACtBm/VeOdXpbjdvHd+J7TTuSxGTKNf41Xq8vs+nSR8ZrLDNt+WRNU6Nn9g+WkIRv2W/7WJmfoLbJ4t+b9iRJci/fO60NBABWyBvKbCRnT+iTlRc07f2a9nbY7NRoZITw7r5zgfzeef9GpoaT3EaeZ01aKyNw3+/68t4Luz4AYAX8arzOSyq2yulNO2VCJtNauHlu6zv2gCRN+d3ZCbqsjATe0NxnqjijZfl7Thn7MqrWykjiO7q+uHO8zttckc0amWY+oenbU1PEAMDTWNZJTTbapbhbXVyGZOvq/sE6Lqzx/ua+TSyn6d/2jNLIhou+BEqslyRP/dmI0I9QAgAr5odlmHZLJGFb1QPPU07kB33nOi4ts5G0eb7Z3f+mDGsCk4Q9v3s2L2FLuZQcRB+n1fhq82wj6sUBwC5zZXefab2cbDC5uWk/HWXn5dcWxGb068cWeXsZRuXWk92ukw/WOGZsf6XGF5tnkY0Pvaxru25sZ+p1KuHx9fG6yLzNCwDADpREJ+vI2im6d5VhV+JHy1CPLQvb7y9rE7jdKrsu29pvG3l5WTxNOe2ATeHg3zX92VXbFuA9tKxdCzdJyZA/lGFH7vTv5P9JTpFo67LNc2PfAQDsXi8uT67cxU7TlhDZjPU2KUy7Ppfx8zKrVbeMc/qOTjY/pMAvALAiPtF37EK/Lf+/pmyRbzftnHKQtWy99hiuRTJS9um+cwNT4d7sKM0oXBtZe5jzYAGAFZLzM19Zdm/Nr5vK5o6WygH2L+n6PtDdx7JJ04/7jg0cVmP/MhxPtZ6LyuaTQABgh1u2kv9OkxGprNlbxhvLsJ5s3m7OvW1erTYAgF0pRWyzwaKPB8d4qAwL/JOkTfGt//5JAAAAAAAAAAAAeOpSr2w6gWCR1DYDAGAbpG7Z+/rOzknj9ca2EwCAveOasvig+/3KrFDtvPM+AQDYw1K6I1Lb7LQxTq1xShlqtGXK9BXjO4+NVwAA9qI2YTuxi5wHmrVrhw2vrDmwHQCAveSCsvGmg3PLcDj7+f0DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA2CL/Adfr0ZkkaoJUAAAAAElFTkSuQmCC>

[image17]: <data:image/png;base64,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>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAAiCAYAAADiWIUQAAAFWklEQVR4Xu3ceah1UxjH8cc8ZZ7na8o8Z5a3l8zTP8hMSBIp5S8yJYVkjBK9hcjwB38os0SSCBFlzJDMQzJkfn53PY+zznqPum/OOXcfvp96Wnuvfe777n3Wrf3cZ629zQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB02+Yeu3ts4rGTx9oel/V9oue5tiMc5nFg2zlCS3n87LGix/Ye8/oPT1vE4/1q/+lqu8t03n96LBf7y3qc2js8bWGPJ5q+rlve41mPlTxWs/6xaT3YdgAA8H93WrQXVX1rVdsbRrupx15V//7V9mwkQ79Gu4bHM1X/0dX2o9EeUPXJIdHu2NfbHe96LBTb10a7c7RJybXs09fbM7ftmGVK0paxkrDJQR5reizpMeWxWPRvZyUJly2tJOdK9lKOr8Zd39GRsb9utLJotQ0AwES51eNHjz88PvZYv//wdFWndryVm+e+Vm6cD1nvRvh5tJk0fRntMB3h8ZvHLx4feCxdHdvGSlLzgsdZ0aeq00uxrc/r3OfE/sFWEgKd/4seV3ls67GRx5nWS0rHSUnJZ1a+94+sPyl50kpCcmnsZ/L1crRbR3uCx7FWqqQnR5/GWVaNdpyUQOmadL73edxfHTvD4zyPL6o+jc8Psf1ptK9Fu4qVRG4JK38QaPuUOKYx1/W9Gfu7eSzu8YCVMb4p+gEAmEhKcC5pO63caL9u+jKBu9xKQvNd7G9lpbqmn9EUl1wT7bDpHFSJab1qpfJSU+KlKqDoOu+ycrOXPPc7o023R5tVrHFTdeiTttPdbKWCtnHs/x7t99FeEa28Fe0r0eqzW8T2uD3vsXK1/1O1reRblGxrXNaxUhVTMi05RkrQk8Y05XWKKqRTVqbElagr+VblTUm9pud3+PuTAABMICVAmzV9unkqMXjKetNw8m20eaNUe6jHClaSJSUbJ3rs4bGr9U9JDUtb9RNNqykRUFvTGrr1PK60MtWmKtsNcUxVJ8U5Viox+g50s38jjo+iQjgTSlJubDvdMR6vV/uqZOraVH1SkqnELZM2Vai07k0JkZIhVVFlNpI2jVdObcqH0Wo9ns5LY6axyfVrGh8dU3VRSZemfVUB1e+V5O+gPG7lOrOiqqRc06nnxv4t0SppBABgYk3Z4ASoqzS1panbUVHipsraBh6nN8f+DU1bas2cppEfa47VlBxPynhoGlNJ+j02f6JcU7Kla1Io6RylrNilR6xU2O5o+gEAmCia2sxps1RPX42KqiYnNaG1V8d5HFV9rqU1XpoarKmiMixKmLSWTWvlhkXrqvRvih4G2CW2s60pAWqTjnGMx4LSWjRVVUVVLH1veihFa9IGUQVNU+RK2pQUj8IFHtdZqbgl/Z9nV/sAAEwkrZXKRdtpNqbNZupeK9OAtdWb/VFTgnj3P8S86nMpF8zX9rbB/dfb/BVEVbBGTWv72mvJaBNkGVQFvM1KEpeL/uXCaltyChQAACwA3XjzlQqi6la9Zm1ULray3mhQ1K/jaLWJwqBkomu+avanom0rm6Lry7Vacr6Vd+B1TTsOWuR/uJW1anoiNLXJpp5GBgAAM6QkQGuQdOPV4no9DaonCfXi2S7S6zi+sXLD13lrYb6mDrU+quu07u49K+9/u7rqrxM2PR2pxE7Xp1ZjovF4u/pMl+zp8bCVBHtu1V8/tanr3s/jHSuVOo3bOP4YAAAAGJpBFbZJNidavVcNAABg4ukluKqi6TUd/5WqUz4Jmi/yBQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQ/YXFD3dnVE9Us4AAAAASUVORK5CYII=>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAYCAYAAABa1LWYAAACBElEQVR4Xu2WPUiVURjHH8saTDdxskDCRRAaxc1cMggJp6CtnJzc3Z0sxEEnByXXwkETQReFCKOihhwE0UFFBRXN74//33OO97mP74eL3ju8P/hxz3nO836c856PK5JxjR64BzdgH+yHZ/CDTorhPTyFb0x8y9QLwjl8ERF7ZGJRME9TAttMrCDYF7vnYw9MPAp77a0wCHfEPSy4LfHT6Zlcf7EZOGtiP+GIuKn50Mf4dReuMkTG4CYsVTF+uTU4BJd9bBz+k1zeE/jZlxMJHUrjCzyGf+C8uAFpycvIv08tPPDlb/CtL1fBp/AHbPUxdkhfy3IFLBP3vNc+PgHfhaQ4ws34gDSYZxe6phf+VXX9ZaMGTcc4SJw5UdjOptIhLvGlbYgg7YZsf6XqX+En1abhF2ZHAmyvVPVADTxUdXufSDiHb5LYIOl5c5K/m4X8RrgIm3NNl9PyMZxS9fJcs3T7X95v2pc58Ku+nAgfnPayfOAu/C/urEliHU5K7mUDPN+46AOd8DusVjGef6PiBofLIsANhjF+gHYVj4Ud4iIuVrhLBtIG/5K09fTcBgrAiv9dgvW6IQ5OlaTe88wqNDzn6mwwiaT11AWbbLDYuS+uQ79tg7hdKa6zRc1HcS8eTurAgI//MvGiZhgeidua+d8sTEHK+gncl/ytNiMjIyMj4y64AFF4gYa2ugzWAAAAAElFTkSuQmCC>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAYCAYAAAC8/X7cAAABw0lEQVR4Xu2VyytFURSHl+RVjBH+AQNDBgbCzIgyMzKiKMy8ysBEBgxMGHiEgZgamRAJRUhKKcbCACFvv9XeO8u6+9xL6XZwvvqy9m/v0133Wuccon9KD7yEb3BE5MU2e4EVIpf0kzmTrzeSTReZRjTHOvDguy7plFNsIxNq7SOHYq/7MbiBazIf4LyCQ/KQJZU+N5IGW8V6DM7Cdbgm8gE4LtanZD7TUQvbxDoP3sFJOCfyuLjmEyHP3Ih6Hrbbuhreir0nWGjrMvuX75lMW3Oz7v4oIXPe8SDqQFLINLajNzzwuVxYT2Y0ZO6Yhr1i7fthZKbrA7gFj2CG2AukhcyFNXrDA5/j5hdF1mBzB9fpts5We0wjXLU1j6G+9tuc0dcv5HP6LDd7Yuss+tjvgN1wisyj1MGP4T5bD8IF2GTXr2TuNceSqAPxNRXELizVIViGG7CZzAi4ceTx5IeCHCnmAh7CAngPO23OPwB/iW24YrOEcPObOvwtJJr/Kh2EjXOKPz787w818eaf57ZSh2HCvVn39QYoouAvFhqGyTRZp/JRm++pPDTMwEcyr3N+ZLkxYnn9TObR5l7/ERERERF/h3fKp3O4TWgNmwAAAABJRU5ErkJggg==>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAAAYCAYAAABKtPtEAAACgElEQVR4Xu2XS6hNURjHP0LIc3BTmDCQRHkMSCJlYOAxYGgmSuQRJhgg3QF5DuRROkUMRO7kptStW2TiVYjEQIiJ98z7/299X+fb3z7LuWXg3NP+1b/zrf/69tp7rb0e+4hU/BOXoN/QC+gidBm6rd4kl9eMEdHoT7CzY4PXGcp/YzT0K5qtwBToNLTceTtcTIZLGgBjt/5ucl4zjkLHo/k/GSzpjZyFRkELJXVyD/TV5ZED0E+NB0FvXB2ZAV2H1ktaIuekmMN2Te/Uuwd9gTZDNWiN+oQDdQv6DC1Q77mk66dDXdBr9QdAd7XeeOziLGzMGvfQ3xW8b9ANaB/0CFpXrJZX0FQpzhIfx7JdT4/7wlNJM4Q8gRZrTJgzHhqq8Wz1t0uaqbzW8ox47xI1ySfR5+yI3kiNZzn/pYsvSHHgfPtjpPH6j8/Q0cDLdewKdFXj/ZLuT/hS32qchQ3FGxnR5xuKHpkMTXRl5gzReA703tUdgU64MuFy6Q7eSUnT2Zgn9XtzFnx3dfTnu3iYxj3QBo2z2AAcjhUNOAj9CN44KQ+KLz+AVkqaaYT7B6/hZspOEe4RczU2lkEPXZn3naDxGam/cQ6G7SUkN0uyHJL6IJhOFTISfICYx6lMf63L4/r0b261pI7whCE8Ye5Aey1B8g96DboPfZC0dAzmb4F6pTxzlkIfoZuSb7cEd9/YuWeFjNYi1zFuits0PiblpdYnlkh9EFqRjZKebVqskHRcclktkuJRmGVVNJTz0roD0IyZ0cixAtoaTWWn9N8B6DPcqGwnjXCnbrQRthW2zuO/Mn5UxM/ftoTfzwOhT5IGgkcHf2sup6KioqKioqK9+AME3aMv3kUhogAAAABJRU5ErkJggg==>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAYCAYAAADzoH0MAAAAuElEQVR4XmNgGAWjADvwB+J1QJyILkEIeALxfyB2hPKnAvFZhDR+4MoA0SyHJAbin0bi4wUgxa/QxBSR2NJA/IMBoi4ZSRwMzKASvugSSGAyEhukNhOJzzAXKogLgLyFLL8Tjc8Qiy6ABGLQBYDgMxCfQRcEGaCBJvYOiN3RxNgZcFgmBcR/GCCSILwKVRoOQHKM6ILEgvdI7DtIbKLAdQZIygThbCDejSqNHxgxILwGw7koKkY4AAAR1Sif0tAZKwAAAABJRU5ErkJggg==>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAYAAAD3Va0xAAAA0klEQVR4XmNgGAWkgtlA/AmI/yPhVygqGBi+IMmBsDeqNCqAKcIGmoD4PLogNsDIADHkFroEEFwGYl90QVwgmwFiUDiSGBMQ/wNiLiQxguAlA6q3DIH4KRKfaIAcPtOg7GMIaeIBSOMFBojLtKB8XAGPE8DC5w+S2BKoWD6SGEHwmgG77SS7CpeGtwwQcUV0CWyAmQGi+DS6BBCoMkDk3qNLYAP9DBDFoegSUABzrSC6BAwsY4Dkr3dQ/JUBkvhgQIYB4hJQWnrMAFF7D0l+FIwCALDWPUOqr0VdAAAAAElFTkSuQmCC>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAI0AAAAYCAYAAADH9X5VAAAFlUlEQVR4Xu2ZV4hsRRCGSzHnLMbrVVAUEwYwofhgxoSIATHnhIpZMSfEBxNiulz0oqKiPhhREcSsmBAMIAbM2auICUN/dNdOTU13z5nZnZ1dmA+KnfNXdU9vn+o4IiNGjBgxHdnHCwNkOy9MU/bywpDZ3wuD5J5g23hxgFwf7BAvTjHOD7aFFw0fBFvOi32wphfGAXW97cUS1wabG+y/ZL8H+9Fp941Ft7NrsCe9mFhGYj1aB1bjRGnF/RXs3XZ3Gz8FW9eLUwTav0T6e5TzwZUSk0p5Q9r7qGbKDxltIrgu2C1erFFqxAyJei45cvEeXv4j0j32YYkx/O3G0tK9vmFwicR2HZH++tlk3qTn+EaibyHvCCwu0beR0VZP2kTTU50Ev+TFRC6hzg32ltNyUO709LfEp8F2lBizofOV+Fcmdy/VBNr/tRcNDDxGcw5Nmvm9I8FMfJJ5XkHqfdovc6ThMsUmiAbs5B2BhSWfNDzX1m3l82CbSYxf0flgk2QvSud31Lgm2J9eHDK0/yovGvDP58WEJo3183nP9JlZym4TWP576a+mLCkN62UJKQU+JNHnd/uleMvOwY5Mn4k/wPgUZhnIJWaNlaS3+HmCnRDs9mBrtLvGODrYA1Ie7SV2CbaHxPacKnGvt3JbRNxo1tqbS5pNpX3/s7f5XFuiDwv2RbA7JM5IvUK97MuqlF7Y9hJ1P6XSKbl4j13uiOeFWUhIBX9pw12iSRtAN6bHBHs+ff7W+JdK2vFG6wUS5S6JdZyWnklqy9VSb68mzVbJDkzPNmksuaRZIGkMVoWDzQ3muQnUcY4XPQRhP0s8mfyRnt8JtqyJUy6WzgbnsDF8/sg8LyatNZp/Ev96LXcjKDPTixnoOMsMiWU5hQCfF2y5++JRqffJs1L3a9KwvJFgDCCee0mav4O95zRmLuJ8EtcgnlmqiO5nDveOClToG5yDKVIh3pYhMZWXpVwfiVyCMnZUlZjlBYnLLeVJqCZ1dIO62JyXYBnWpThHbnnicGCThuVK8Umje0/dA1nQuRuy1NpCn/BOilBZ6YWVmC3dy/AiWA4UXr6WOTPYIsbnE0phH1I7IVFmdy86OPbavYDlO8l/bz9Qz71eNHwc7BMvGnJJw3JDXynceyk+aQ5OzzsYTcn1L6tFid+Cve5FS67Cbpwt3cu85p51Qw13W4dE/X6nNYFyTY7oZ3gh8ZTEOr7yjj4ovTDlCanPRN2O3GD73CfNxumZTbCn13dM7INetBDwoRe7wMjt1gjv1xtfO1qAmQJ9fac/JnHPUTqiAuVqfuUXLwS2DvaPtEaoHdG9sqV0/r8eDhO1GE0aZpccm0tsr8L1BfHMxgrPuctRdO7VgBXghWCHjnk7sfEdnCcxwC4jTal1wKXS6V8naf76P7c8ckxcS+IUmVujgQ26L1eCU9Fu5ln3M5pwT6fnm8YieoM9U7e2rC31GA4g+Lkn8ejM/rjR6Ec0e+usx35ukBXuv+xBYLbEgfqm0TzUwUzWBkewXyU2lJHPGlabOnNQsd2YwfIS62WGmCvxNyQbw3KgsJ8ghu9nv8PGWE8zSq2Tr5DY7qaQONSHveJ8wEhWv91vNYFTS5OLRurmks7Sy29P/DwBfNeXwT6TeANt+23VYN9Lqww/bXjo61W8mFhUYrmBwPLBNDcouDDzO34LL2oDLw4JOvksL2bgbuhyLw6BWlLcLPGCc2DUvny8MBpWC/aMd8iAR0ND7pRWG5q2RY/Fw+SiYLcF29bpysDbd5nUj5njgVvVVyVOtx5OOzl9MqFzuetgP3Or89VgEBznxUmE22/uz07xjsAFMkkz4XPS7Ng7UVwo/W3cJxr2JjcG29c7GsBJiRvxqQSnsfe9OEiO9cIA2c8L05SDvDBkTvbCiBEjRkwu/wNq15ch43JdhQAAAABJRU5ErkJggg==>

[image25]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAYCAYAAADH2bwQAAAAZklEQVR4XmNgGAW4ABsQewCxO7qEKhD/B+LjQFwCxLHIkhFQSXFkQWQAkrwJxCxIGA5MoQpARu9AwnBQDFXAhCyIDAwYIArY0SWQwR8gPorEB5kmg8QHA5C9IJNA+CAQM6JKD2sAAPoREsiqMI75AAAAAElFTkSuQmCC>