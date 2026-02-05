# **AlphaHoldem: 엔드투엔드 심층 강화학습을 통한 헤즈업 노리밋 텍사스 홀덤 인공지능의 구현 및 기술적 프레임워크 분석**

헤즈업 노리밋 텍사스 홀덤(Heads-Up No-Limit Texas Hold'em, HUNL)은 불완전 정보 게임(Imperfect Information Games, IIGs) 분야에서 인공지능의 성능을 측정하는 가장 정교하고 도전적인 벤치마크 중 하나로 간주된다.1 이 게임은 광대한 의사결정 공간과 전략적 복잡성으로 인해 전통적인 게임 이론 접근 방식으로는 해결하기 어려운 과제들을 제시해 왔다.1 기존의 HUNL 인공지능인 DeepStack이나 Libratus는 주로 후회 최소화 알고리즘(Counterfactual Regret Minimization, CFR)과 그 변형들에 의존해 왔으나, 이러한 방식은 훈련 및 반복 추론 과정에서 막대한 계산 비용을 발생시켜 연구의 접근성과 광범위한 적용을 제한하는 요소가 되었다.1

AlphaHoldem은 이러한 CFR 프레임워크의 한계를 극복하고, 저렴한 계산 및 저장 비용으로도 슈퍼휴먼(Superhuman) 수준의 성능을 발휘할 수 있는 고성능 HUNL AI를 개발하는 것을 목표로 설계되었다.1 이 모델은 엔드투엔드(End-to-End) 자기 대전(Self-Play) 강화학습 프레임워크를 도입하여 입력 상태 정보로부터 출력 행동을 단일 순전파(Feedforward) 과정만으로 직접 도출해냄으로써, 계산 집약적인 CFR 반복 추론 과정을 제거하였다.1 본 보고서는 AlphaHoldem의 핵심 아키텍처, 상태 표현 방식, 손실 함수, 그리고 이를 구현하기 위해 필요한 기술적 세부 사항들을 심층적으로 분석한다.

## **1\. HUNL AI의 역사적 맥락과 CFR의 한계**

포커 AI 연구의 이정표가 된 초기 시스템들은 대부분 CFR 알고리즘에 기반하고 있었다.4 CFR은 두 플레이어의 후회를 최소화하여 시간 평균 전략이 내쉬 균형(Nash Equilibrium)에 수렴하도록 유도하는 개념적으로 단순한 반복 알고리즘이다.4 2015년 헤즈업 리밋 홀덤(HULH)이 해결된 이후 연구의 초점은 노리밋 버전인 HUNL로 이동하였으며, DeepStack과 Libratus가 잇따라 인간 전문가를 꺾는 성과를 거두었다.3

그러나 DeepStack과 같은 모델들은 고품질의 예측을 보장하기 위해 실제 플레이 도중 1,000회 이상의 반복적인 추론 과정을 수행해야 하며, 이는 의사결정 속도를 늦추고 방대한 컴퓨팅 자원을 요구하게 만든다.4 이러한 제약은 소규모 연구 기관에서 고성능 포커 AI를 개발하는 것을 불가능하게 만들었으며, CFR 모델을 더 큰 규모의 불완전 정보 게임으로 확장하는 데 장애물이 되었다.1 AlphaHoldem은 이러한 배경 속에서 CFR 프레임워크가 아닌 강화학습 기반의 해결책을 제시하며 등장하였다.1

| 모델명 | 주요 알고리즘 | 특징 | 의사결정 속도 |
| :---- | :---- | :---- | :---- |
| DeepStack | CFR \+ Deep Learning | 재귀적 추론 및 서브게임 해결 | 수 초 단위 (상대적 저속) |
| Libratus | CFR \+ Blueprint Strategy | 평형 전략 사전 계산 및 실시간 최적화 | 고성능 서버 필요 |
| AlphaHoldem | End-to-End RL (PPO) | 단일 순전파, 경량 아키텍처 | 2.9 밀리초 (매우 빠름) |

1

## **2\. AlphaHoldem의 다차원 텐서 상태 표현**

AlphaHoldem의 핵심적인 기술적 기여 중 하나는 인간의 도메인 지식에 의존한 추상화를 배제하고, 게임의 상태 정보를 다차원 텐서(Multidimensional Tensor) 형태로 표현한 것이다.1 기존의 포커 AI들은 계산 복잡도를 낮추기 위해 유사한 가치를 지닌 카드나 베팅 크기를 그룹화하는 카드 추상화(Card Abstraction) 및 액션 추상화(Action Abstraction)를 수행하였으나, 이는 전략적 정밀도를 떨어뜨리는 원인이 되었다.4

AlphaHoldem은 원본 카드 정보와 베팅 이력을 인코딩하여 컨볼루션 신경망(CNN)이 학습하기에 적합한 다차원 구조로 재구성한다.1 이 방식은 현재와 과거의 포커 정보를 모두 포함하며, 정보 집합(Information Set) 내부의 숨겨진 정보를 추론하는 데 필요한 구조적 데이터를 제공한다.4

### **2.1 카드 정보 인코딩**

카드 정보는 추상화 없이 텐서에 직접 인코딩된다.1 이는 신경망이 숫자의 크기나 무늬의 배열을 스스로 학습하도록 유도하며, 도메인 지식 없이도 강력한 특징(Feature)을 추출할 수 있게 한다.1 각 단계(Pre-flop, Flop, Turn, River)에서의 공개 카드와 플레이어의 비공개 카드가 텐서의 특정 채널에 배치되어 계층적으로 처리된다.4

### **2.2 베팅 및 액션 이력 인코딩**

노리밋 게임의 특성상 베팅 크기는 제한이 없으나, AlphaHoldem은 이를 효과적으로 학습하기 위해 ![][image1]와 같은 이산적인 액션 공간을 정의하거나, 베팅 이력을 시퀀스 데이터로 변환하여 텐서에 포함시킨다.4 이 텐서 표현은 컨볼루션 네트워크를 통해 효율적인 특징 학습이 가능하게 하며, 벡터 기반의 상태 표현보다 전략적 우위를 점하는 것으로 입증되었다.4

| 데이터 범주 | 인코딩 방식 | 목적 |
| :---- | :---- | :---- |
| 비공개 카드 (Hole Cards) | 이진 또는 인덱스 기반 텐서 채널 | 플레이어 고유 정보 보존 |
| 공용 카드 (Community Cards) | 단계별 텐서 매핑 (최대 5장) | 공유 정보 및 보드 텍스처 인식 |
| 베팅 이력 (Betting History) | 액션 시퀀스 및 칩 수량의 다차원 배열 | 상대방 범위(Range) 추론 및 전략적 흐름 파악 |
| 칩 상태 (Stack/Pot Size) | 정규화된 스칼라 값의 텐서 통합 | 기대 가치(EV) 및 위험 관리 계산 기초 |

1

## **3\. Pseudo-Siamese 신경망 아키텍처**

AlphaHoldem은 입력 상태 정보로부터 액션을 직접 도출하기 위해 유사-샴 아키텍처(Pseudo-Siamese Architecture)를 채택하였다.1 이 구조는 카드 정보와 액션(베팅) 정보를 분리하여 처리한 후, 최종적으로 통합하여 의사결정을 내리는 방식을 취한다.4

### **3.1 ResNet 기반의 특징 추출**

모델의 특징 학습 부분에는 잔차 네트워크(Residual Network, ResNet) 구조가 활용된다.4 ResNet은 스킵 연결(Skip Connection)을 통해 깊은 층에서도 그래디언트 소실 문제를 완화하며, 복잡한 포커 데이터에서 고수준의 특징 계층을 학습할 수 있게 한다.11 AlphaHoldem의 신경망은 컨볼루션 층, 배치 정규화(Batch Normalization), ReLU 활성화 함수로 구성된 다수의 잔차 블록을 포함한다.11

### **3.2 모델 규모와 효율성**

AlphaHoldem 모델 전체의 크기는 100MB 미만으로 매우 경량화되어 있다.4 이러한 설계는 단일 GPU를 사용하는 환경에서도 2.9밀리초라는 초고속 의사결정을 가능하게 하며, 이는 DeepStack보다 1,000배 이상 빠른 속도이다.2 추론 시에는 단 한 번의 순전파만 수행하면 되므로, 실시간 멀티 플레이어 환경이나 모바일 기기 등 자원이 제한된 환경으로의 확장성도 뛰어난 것으로 평가받는다.1

## **4\. Trinal-Clip PPO 손실 함수**

포커와 같은 불완전 정보 게임은 상대방의 정책 불확실성(예: 블러핑)으로 인해 보상의 분산이 매우 크며, 이는 강화학습의 안정성을 해치는 주요 요인이 된다.1 AlphaHoldem은 이를 해결하기 위해 기존의 근사 정책 최적화(Proximal Policy Optimization, PPO)를 개선한 Trinal-Clip PPO 손실 함수를 제안하였다.1

### **4.1 정책 손실 함수 (![][image2])**

정책 손실 함수는 업데이트되는 정책이 이전 정책에서 너무 멀어지지 않도록 제한하는 클리핑 메커니즘을 포함한다.16 AlphaHoldem은 특히 이득 함수(![][image3])가 음수일 때, 즉 현재 행동이 좋지 않은 결과를 가져왔을 때의 분산을 관리하기 위해 추가적인 클리핑 하이퍼파라미터 ![][image4]을 도입하였다.4

정책 손실 수식은 다음과 같이 정의된다:

![][image5]  
여기서 $r\_t(\\theta)$는 새로운 정책과 이전 정책의 확률 비율이며, ![][image6]은 표준 PPO 클립 비율(일반적으로 0.2)이다.4 ![][image4]은 불완전 정보 환경에서의 변동성을 제어하여 학습 속도와 안정성을 동시에 향상시킨다.1

### **4.2 가치 손실 함수 및 보상 클리핑**

가치 네트워크 또한 큰 분산에 노출되므로, AlphaHoldem은 가치 손실 계산 시에도 클리핑을 적용한다.4 이를 위해 ![][image7]와 ![][image8]라는 두 개의 하이퍼파라미터가 사용되는데, 이들은 플레이어와 상대방이 베팅한 칩의 총량에 기반하여 동적으로 계산된다.4 이러한 동적 클리핑은 수동적인 튜닝 없이도 리플레이 데이터의 특성에 맞춰 가치 예측의 오차를 최소화하는 데 기여한다.4

| 하이퍼파라미터 | 역할 | 계산 방식 |
| :---- | :---- | :---- |
| **![][image6]** | 표준 정책 업데이트 제한 | 고정 상수 (예: 0.2) |
| ![][image4] | 부정적 이득 상황에서의 추가 정책 클리핑 | 학습 중 설정되는 하이퍼파라미터 |
| ![][image9] | 가치 네트워크 업데이트 안정화를 위한 클리핑 | 리플레이 내 칩 투입량에 따른 동적 계산 |

1

## **5\. K-Best Self-Play 알고리즘**

강화학습 기반 포커 AI의 고전적인 문제 중 하나는 자기 대전 과정에서 에이전트가 특정 전략에만 매몰되거나, 이전 버전의 자기 자신을 이기는 법만 학습하여 전략적 다양성을 잃는 현상이다.1 AlphaHoldem은 이를 방지하기 위해 단일 스레드 에이전트의 효율성을 유지하면서도 정책 다양성을 확보할 수 있는 K-Best Self-Play 알고리즘을 도입하였다.1

### **5.1 풀(Pool) 관리 및 선택 기준**

알고리즘은 훈련 과정에서 생성된 주요 에이전트들의 이력 버전을 풀에 유지한다.4 훈련 중인 메인 에이전트는 이 풀에서 선별된 상대들과 대전하며 학습 데이터를 생성한다.4

1. **ELO 점수 기반 평가:** 모든 이력 버전은 대전 결과를 통해 ELO 점수가 산출되며, 이는 상대적인 실력을 나타내는 척도로 사용된다.3  
2. **Top-K 생존자 선택:** 풀 내의 에이전트 중 ELO 점수가 가장 높은 상위 ![][image10]개의 버전을 'K-Best' 상대로 선정한다.4  
3. **학습 다각화:** 메인 에이전트는 이 ![][image10]개의 최고 성능 버전들과 경쟁함으로써, 과거에 유효했던 다양한 전략들을 모두 극복할 수 있는 강력한 최적 대응(Best Response) 전략을 학습하게 된다.1

### **5.2 NFSP와의 비교 및 우위**

K-Best Self-Play는 신경 가상 자기 대전(Neural Fictitious Self-Play, NFSP)의 개념을 차용하면서도, 모든 과거 정책의 평균을 사용하는 대신 최고 성능의 정책들만을 대상으로 함으로써 계산 효율성을 크게 높였다.1 이 방식은 에이전트가 국소 최적점(Local Minima)에 빠지는 것을 방지하고, 단순한 카운터 전략에 패배하지 않는 견고한 전략을 구축하도록 돕는다.4

## **6\. 모델 구현을 위한 기술적 요구사항 및 환경**

AlphaHoldem을 실제로 구현하고 훈련시키기 위해서는 분산 컴퓨팅 프레임워크와 정교한 포커 시뮬레이션 환경이 필요하다.21

### **6.1 하드웨어 및 소프트웨어 스택**

AlphaHoldem의 공식 연구 결과에 따르면, 고성능 HUNL AI를 구축하기 위해 다음과 같은 자원이 투입되었다.1

* **컴퓨팅 서버:** 8개의 GPU(NVIDIA TITAN V 등)와 64개의 CPU 코어를 갖춘 단일 서버.1  
* **훈련 기간:** 약 3일 간의 집중 학습을 통해 슈퍼휴먼 수준 도달.1  
* **프레임워크:** TensorFlow를 주 학습 프레임워크로 사용하며, 대규모 병렬 처리를 위해 Ray 라이브러리를 활용하는 것이 일반적이다.21

### **6.2 포커 환경 (RLCard 등)**

구현 시 가장 흔히 사용되는 환경은 rlcard의 nl-holdem 환경이다.21 그러나 기본 제공 환경은 50bb(Big Blind) 스택 설정이나 베팅 크기 제한 등 실제 ACPC(Annual Computer Poker Computer) 표준과 다를 수 있으므로, 전문가 수준의 모델을 위해서는 환경의 규칙을 수정하거나 확장할 필요가 있다.21

### **6.3 분산 훈련 구조**

분산 학습 시스템은 일반적으로 다음과 같은 구조로 운영된다 8:

1. **Actor:** 여러 CPU 코어에서 병렬로 실행되며, 현재 정책을 사용하여 환경과 상호작용하고 경험 샘플(State, Action, Reward, Next State)을 생성한다.8  
2. **Learner:** GPU를 활용하여 Actor들이 수집한 리플레이 버퍼에서 미니 배치를 샘플링하고, Trinal-Clip PPO 알고리즘을 통해 신경망 가중치를 업데이트한다.8  
3. **League Tracker:** 훈련 중인 모델의 체크포인트를 저장하고, 이들의 ELO 점수를 실시간으로 추적하여 K-Best 상대를 갱신한다.19

| 하이퍼파라미터 항목 | 권장 설정값 | 비고 |
| :---- | :---- | :---- |
| 학습률 (Learning Rate) | ![][image11] | Adam 옵티마이저 권장 8 |
| 배치 크기 (Batch Size) | ![][image12] | 메모리 및 GPU 성능에 따라 조절 8 |
| 할인 계수 (Gamma) | ![][image13] | 미래 보상에 대한 가치 부여 8 |
| GAE 람다 (![][image14]) | 0.95 | 장기적인 보상 예측의 분산 감소 8 |
| 클리핑 범위 (![][image6]) | 0.2 | PPO 표준 설정 4 |

## **7\. 실험 결과 및 성능 검증**

AlphaHoldem의 성능은 기존의 강력한 AI 에이전트들과 인간 전문가들과의 대전을 통해 검증되었다.1 성능 지표로는 1,000 핸드당 밀리-빅블라인드 이득을 나타내는 mbb/h가 사용되었다.1

### **7.1 기존 AI와의 비교 (Slumbot 및 DeepStack)**

AlphaHoldem은 10만 핸드 이상의 테스트에서 Slumbot과 DeepStack을 모두 압도하였다.2 Slumbot과의 대결에서는 111.56 mbb/h라는 큰 격차로 승리하였으며, DeepStack과의 경기에서도 16.91 mbb/h의 우위를 점하며 통계적으로 유의미한 성능 차이를 보였다.1 이는 AlphaHoldem의 엔드투엔드 학습이 온라인 서브게임 해결 방식보다 효율적이면서도 강력한 전략을 생성할 수 있음을 입증한다.1

### **7.2 인간 전문가와의 대전**

10,000 핸드에 걸친 인간 전문가들과의 테스트에서 AlphaHoldem은 평균 10.27 ± 65.13 mbb/h의 승률을 기록하며 슈퍼휴먼 성능을 달성하였다.1 특히 학습된 정책을 시각화하여 분석한 결과, AlphaHoldem은 인간이 가르쳐주지 않았음에도 불구하고 프리플랍(Pre-flop) 단계에서 인간 전문가들이 선호하는 베팅 빈도와 매우 유사한 전략을 구사하는 것으로 나타났다.1 이는 강화학습을 통해 실제 게임의 본질적인 전략적 구조를 성공적으로 파악했음을 시사한다.1

## **8\. 구현 시 고려해야 할 심층적 통찰**

동일한 모델을 성공적으로 구현하기 위해서는 단순한 아키텍처 복제 이상의 세부적인 전략이 필요하다.

### **8.1 상태 표현의 정밀도와 특징 학습**

AlphaHoldem이 카드 추상화를 배제했다는 점은 모델이 카드의 절대적 가치뿐만 아니라 보드 텍스처(예: 스트레이트나 플러시 가능성)를 학습해야 함을 의미한다.1 따라서 컨볼루션 필터의 크기와 스트라이드(Stride) 설정이 카드 간의 관계를 포착하기에 충분히 정교해야 한다.11 7x7 컨볼루션보다는 3x3 컨볼루션을 여러 층 쌓는 것이 더 복잡한 특징을 효과적으로 추출하는 데 유리할 수 있다.11

### **8.2 분산 환경에서의 통신 병목 현상**

Ray와 같은 프레임워크를 사용할 때 수십 개의 Actor가 생성한 대량의 리플레이 데이터를 중앙 Learner에게 전달하는 과정에서 병목 현상이 발생할 수 있다.21 이를 완화하기 위해 공유 메모리(Plasma Store)를 효율적으로 활용하고, 데이터 전송 시 직렬화(Serialization) 비용을 최소화하는 인코딩 방식을 채택해야 한다.21

### **8.3 수렴 지연 및 보상 설계의 난점**

비공식 구현 사례에 따르면, 10억 번 이상의 자기 대전 후에도 모델이 완전히 수렴하지 않고 지속적으로 성능이 개선되는 경향을 보였다.21 이는 HUNL의 복잡성으로 인해 내쉬 균형에 도달하는 것이 매우 어렵기 때문일 수 있다. 따라서 모델의 성능 변화를 단순히 승률로만 측정하기보다는, 특정 벤치마크 에이전트(예: Slumbot)와의 주기적인 평가를 통해 개선 여부를 판단하는 리그 트래커(League Tracker) 시스템이 필수적이다.19

### **8.4 가치 네트워크와 정책 네트워크의 균형**

가치 네트워크(Critic)는 정책 네트워크(Actor)의 업데이트 가이드 역할을 하므로, 초기에 가치 네트워크가 충분히 안정되지 않으면 정책 학습이 붕괴될 위험이 있다.8 AlphaHoldem의 Trinal-Clip PPO는 이러한 붕괴를 막는 안전장치 역할을 하지만, 초기에 가치 네트워크만을 대상으로 하는 사전 훈련(Pre-training) 기간을 두거나 학습률을 차등적으로 적용하는 등의 전략도 유효할 수 있다.24

## **9\. 결론 및 향후 전망**

AlphaHoldem은 기존의 복잡하고 비용이 많이 드는 CFR 프레임워크를 대체할 수 있는 강력하고 효율적인 강화학습 기반의 HUNL AI 솔루션을 제시하였다.1 다차원 텐서 표현, Pseudo-Siamese 아키텍처, Trinal-Clip PPO, 그리고 K-Best Self-Play로 이어지는 기술적 연쇄는 소규모 연구진도 슈퍼휴먼 수준의 포커 AI를 개발할 수 있는 길을 열어주었다.1

이 모델은 단일 GPU만으로도 극도의 의사결정 속도를 보장하며, 이는 실시간 서비스 제공이나 더 큰 규모의 다인용 게임(Multi-player Poker)으로의 확장에 핵심적인 토대가 될 것이다.1 향후 AlphaHoldem의 프레임워크는 포커뿐만 아니라 사이버 보안, 경매 시스템, 동적 가격 책정 등 불완전한 정보와 높은 변동성을 특징으로 하는 다양한 현실 세계의 의사결정 문제로 전이(Transfer)될 수 있는 잠재력을 지니고 있다.4 전문가 수준의 모델 구현을 희망하는 개발자라면 본 보고서에서 분석된 텐서 인코딩과 클리핑 메커니즘을 정밀하게 적용하고, 분산 학습 환경의 안정성을 확보하는 데 주력해야 할 것이다.1

#### **참고 자료**

1. AlphaHoldem: High-Performance Artificial Intelligence for Heads-Up No-Limit Poker via End-to-End Reinforcement Learning \- Liner, 2월 5, 2026에 액세스, [https://liner.com/review/alphaholdem-highperformance-artificial-intelligence-for-headsup-nolimit-poker-via-endtoend](https://liner.com/review/alphaholdem-highperformance-artificial-intelligence-for-headsup-nolimit-poker-via-endtoend)  
2. AlphaHoldem: High-Performance Artificial Intelligence for Heads-Up No-Limit Poker via End-to-End Reinforcement Learning, 2월 5, 2026에 액세스, [https://aaai.org/papers/04689-alphaholdem-high-performance-artificial-intelligence-for-heads-up-no-limit-poker-via-end-to-end-reinforcement-learning/](https://aaai.org/papers/04689-alphaholdem-high-performance-artificial-intelligence-for-heads-up-no-limit-poker-via-end-to-end-reinforcement-learning/)  
3. A Survey on Self-play Methods in Reinforcement Learning \- arXiv, 2월 5, 2026에 액세스, [https://arxiv.org/pdf/2408.01072](https://arxiv.org/pdf/2408.01072)  
4. AlphaHoldem: High-Performance Artificial Intelligence for Heads-Up ..., 2월 5, 2026에 액세스, [https://cdn.aaai.org/ojs/20394/20394-13-24407-1-2-20220628.pdf](https://cdn.aaai.org/ojs/20394/20394-13-24407-1-2-20220628.pdf)  
5. AlphaHoldem: High-Performance Artificial Intelligence for Heads-Up No-Limit Poker via End-to-End Reinforcement Learning, 2월 5, 2026에 액세스, [https://ojs.aaai.org/index.php/AAAI/article/view/20394](https://ojs.aaai.org/index.php/AAAI/article/view/20394)  
6. AlphaHoldem: High-Performance Artificial Intelligence for Heads-Up No-Limit Poker via End-to-End Reinforcement Learning | Semantic Scholar, 2월 5, 2026에 액세스, [https://www.semanticscholar.org/paper/AlphaHoldem%3A-High-Performance-Artificial-for-Poker-Zhao-Yan/4fcf18bda55414c5f31cc4be560bae92e8e4b7e9](https://www.semanticscholar.org/paper/AlphaHoldem%3A-High-Performance-Artificial-for-Poker-Zhao-Yan/4fcf18bda55414c5f31cc4be560bae92e8e4b7e9)  
7. Counterfactual Regret Minimization (CFR) \- Steven Gong, 2월 5, 2026에 액세스, [https://stevengong.co/notes/Counterfactual-Regret-Minimization](https://stevengong.co/notes/Counterfactual-Regret-Minimization)  
8. Algorithms for Games AI \- MDPI, 2월 5, 2026에 액세스, [https://mdpi-res.com/bookfiles/book/11350/Algorithms\_for\_Games\_AI.pdf?v=1755579679](https://mdpi-res.com/bookfiles/book/11350/Algorithms_for_Games_AI.pdf?v=1755579679)  
9. Deep Counterfactual Regret Minimization \- Proceedings of Machine Learning Research, 2월 5, 2026에 액세스, [https://proceedings.mlr.press/v97/brown19b.html](https://proceedings.mlr.press/v97/brown19b.html)  
10. RL-CFR: Improving Action Abstraction for Imperfect Information Extensive-Form Games with Reinforcement Learning \- GitHub, 2월 5, 2026에 액세스, [https://raw.githubusercontent.com/mlresearch/v235/main/assets/li24t/li24t.pdf](https://raw.githubusercontent.com/mlresearch/v235/main/assets/li24t/li24t.pdf)  
11. ResNets fully explained with implementation from scratch using PyTorch. \- Medium, 2월 5, 2026에 액세스, [https://medium.com/@YasinShafiei/residual-networks-resnets-with-implementation-from-scratch-713b7c11f612](https://medium.com/@YasinShafiei/residual-networks-resnets-with-implementation-from-scratch-713b7c11f612)  
12. ResNet in a Nutshell: The Breakthrough That Made Deep Learning Deeper | by Okan Yenigün | Artificial Intelligence in Plain English, 2월 5, 2026에 액세스, [https://ai.plainenglish.io/resnet-in-a-nutshell-the-breakthrough-that-made-deep-learning-deeper-8667692eaac5](https://ai.plainenglish.io/resnet-in-a-nutshell-the-breakthrough-that-made-deep-learning-deeper-8667692eaac5)  
13. ResNet: Enabling Deep Convolutional Neural Networks through Residual Learning \- arXiv, 2월 5, 2026에 액세스, [https://arxiv.org/html/2510.24036v1](https://arxiv.org/html/2510.24036v1)  
14. Detailed Explanation of Resnet CNN Model. | by TANISH SHARMA \- Medium, 2월 5, 2026에 액세스, [https://medium.com/@sharma.tanish096/detailed-explanation-of-residual-network-resnet50-cnn-model-106e0ab9fa9e](https://medium.com/@sharma.tanish096/detailed-explanation-of-residual-network-resnet50-cnn-model-106e0ab9fa9e)  
15. 8.6. Residual Networks (ResNet) and ResNeXt \- Dive into Deep Learning, 2월 5, 2026에 액세스, [https://d2l.ai/chapter\_convolutional-modern/resnet.html](https://d2l.ai/chapter_convolutional-modern/resnet.html)  
16. Proximal Policy Optimization (PPO) \- verl documentation \- Read the Docs, 2월 5, 2026에 액세스, [https://verl.readthedocs.io/en/latest/algo/ppo.html](https://verl.readthedocs.io/en/latest/algo/ppo.html)  
17. Proximal Policy Optimization — Spinning Up documentation \- OpenAI, 2월 5, 2026에 액세스, [https://spinningup.openai.com/en/latest/algorithms/ppo.html](https://spinningup.openai.com/en/latest/algorithms/ppo.html)  
18. PPO Algorithm. Proximal Policy Optimization (PPO) is… | by DhanushKumar | Artificial Intelligence in Plain English, 2월 5, 2026에 액세스, [https://ai.plainenglish.io/ppo-algorithm-3b33195de14a](https://ai.plainenglish.io/ppo-algorithm-3b33195de14a)  
19. A Survey on Self-play Methods in Reinforcement Learning \- NICS-EFC, 2월 5, 2026에 액세스, [https://nicsefc.ee.tsinghua.edu.cn/nics\_file/pdf/db43f779-dd0e-4f2e-a51c-1caa107e21eb.pdf](https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/db43f779-dd0e-4f2e-a51c-1caa107e21eb.pdf)  
20. Deep Reinforcement Learning from Self-Play in No-limit Texas Hold'em Poker, 2월 5, 2026에 액세스, [https://www.researchgate.net/publication/357245384\_Deep\_Reinforcement\_Learning\_from\_Self-Play\_in\_No-limit\_Texas\_Hold'em\_Poker](https://www.researchgate.net/publication/357245384_Deep_Reinforcement_Learning_from_Self-Play_in_No-limit_Texas_Hold'em_Poker)  
21. bupticybee/AlphaNLHoldem: An unoffical implementation of AlphaHoldem. 1v1 nl-holdem AI. \- GitHub, 2월 5, 2026에 액세스, [https://github.com/bupticybee/AlphaNLHoldem](https://github.com/bupticybee/AlphaNLHoldem)  
22. Scalable Implementation of Deep CFR and Single Deep CFR \- GitHub, 2월 5, 2026에 액세스, [https://github.com/EricSteinberger/Deep-CFR](https://github.com/EricSteinberger/Deep-CFR)  
23. A Deep Reinforcement Learning-Based Approach in Porker Game, 2월 5, 2026에 액세스, [http://www.csroc.org.tw/journal/JOC34-2/JOC3402-04.pdf](http://www.csroc.org.tw/journal/JOC34-2/JOC3402-04.pdf)  
24. Reinforcement learning with Takagi-Sugeno-Kang fuzzy systems \- OAE Publishing Inc., 2월 5, 2026에 액세스, [https://www.oaepublish.com/articles/ces.2023.11](https://www.oaepublish.com/articles/ces.2023.11)  
25. Mastering Proximal Policy Optimization (PPO) in Reinforcement Learning with Code | by Felix Verstraete | Medium, 2월 5, 2026에 액세스, [https://medium.com/@felix.verstraete/mastering-proximal-policy-optimization-ppo-in-reinforcement-learning-230bbdb7e5e7](https://medium.com/@felix.verstraete/mastering-proximal-policy-optimization-ppo-in-reinforcement-learning-230bbdb7e5e7)  
26. Optimal Policy of Multiplayer Poker via Actor-Critic Reinforcement Learning \- PMC \- NIH, 2월 5, 2026에 액세스, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9222241/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9222241/)  
27. Techniques and Paradigms in Modern Game AI Systems \- MDPI, 2월 5, 2026에 액세스, [https://www.mdpi.com/1999-4893/15/8/282](https://www.mdpi.com/1999-4893/15/8/282)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADcAAAAYCAYAAABeIWWlAAABrklEQVR4Xu2VvyuFURjHHwqDlFBksBkUZfEXWMjgVzIYDCYZZDAYhJTBpDCI/IzVQibJIruSom5hUCaxKL++T+d5r3Me56p7y/uG86lv7zmfc+55n3PfX0SBwJ+iHKnT8rfThjwj78iRGkuCJuSWTD3jaixneLFmLWNmm0wdESvIvdXPiQFyF02CGjI1dCrPrlu5rLih5De3SKaGauXZPdiiAVlFiqVfgkwic0i+OBte4BipRXaRenc4FlJk6ihVnl36jy9C9pE+kdPIuoxNiNNEC7QgedKecWb8PGuU+cqlaz6UY7S5qWiAzBXUm+sXV2U5vup6nmYrQzaRDTLF8jr8UliW33wHf4r4nF3KO5sbleO5LYUhj7vyOL5FtYuDPXLPey39L7WwOFGOH0w9kfunHqfnxUU7ckFmo/x+8NbCgp8h7cY8rsPjUsppZrNMrnAtO7boFWnTY7kyZEHa7CqlzQyKSwI+74HVbxTncOaR9v18Z/lXZFjahWTmtH4Oxwqfm19KEW9kPl8OL8i8cgVkfszR35JH8U9IhRqLkyUydVzKccQdDgQCgcA/5wMffH0o+N9wQAAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAYCAYAAACbU/80AAABQklEQVR4XmNgGAWjABXMA+JPQPwfCX8E4j5kRfQAMMsHBDAyQCw/iy5BL5DNAHGAF7oEvcBLhgEMfhAY0PgHAZDlJ9AF6QUIxb8TGp8bjU8xeM2AP/hBZQIywKeWLIAv/muA2BGJ7w/EF5H4FANmBojl2AyVZUB1GMyh6A62A+LnQLwZiJOAmAOIVwHxUiCeBmVXwFWjgX4GiGGBaOIzoOIX0MTRQ6qYAZF41wPxESC+DuWjOx4FLAbiX0D8F4j/MaD6DMT/A8TfgVgGpgEK0A1C58NAFBAfR+LjUkcS8APiS0h8HgbcBt8EYncom4kBtzqSwCEgDoeyy6E0usGwtAQSB1kMAiB91lA2RUAOiB8wQBIXDAQB8T0gvgzEHUjiIAc8AeK3QOyDJE4XEAvEp9AF6QXEgfgFENcCMTua3CiAAwDgmFNJr82WmgAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAYCAYAAAD6S912AAABBElEQVR4XmNgIAxMgPg/EJehS5ADLIH4G5R9Hoi7kOTIAmvQ+NPR+IMHSALxP3RBSgDIMFDEUAWkMkAMo5qBMMOoYuBeIFYA4hcMVDCQE4gfQNn7GKhgICxBg8BUBoiB6khiJAFnIO5B4hcwQAwMQRIjCaB7zxsq1ogmDgMs6ALIYC4Qf0LDnxkgBq5DUgcDIJ9MQBeEASYg/oguCAUgA++gCzJAEr0wuiAMvAFiVnRBKEBPi1lIYiCcgSQHdtlRqAQ2ALIE3UAQEGDAksfDGCDefAvEX4H4N6o0WANMHkSD5Iuhcn0MeMKPHPCXAU/4kQNgQRCPIkoBAOX1g0Csgy4xggAA0kBBjqHDSgoAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAWCAYAAADwza0nAAAAtUlEQVR4XmNgGNIgAYj/A/E+IPZDlcINQoE4CMp+wwAxgCgwBYht0QWJASoMJNiCDkAaz6MLEgM2MkA0u6BL4AI2DBAN+lCaKCcrMKAqhNlKEIAUdSDxraFiEkhiTEB8HYnPoMOAaboMmtg8IHZGE2OYhC4ABGVYxEAARawUXQAIHgPxcjQxEEBXBxbggLKFoXxsAEMc5PEXUIm3aHLIAEMjsYB+Gr8D8XsGSFZ7B8QHUaXpAQCYGS4xx6PEaAAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,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>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAWCAYAAAD9091gAAAAaElEQVR4XmNgGAXYgDsQXwPif0C8BVmCEYj/A/FiIA4AYh8gfgPEn2AKjsEYuMAzIOYDYnsgroeyUcA0ID7LALEGhJWQJa2AuBhZAB08RRdAB8uBuAJNLBKIryILnGdA2A/CBH01qAAAGnUVZ96Il4QAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAZCAYAAAA4/K6pAAAA8UlEQVR4XmNgGLZACF2AWMAIxP+B+CiUBmGSQC8DxBAQkADi3UhyRIFEIN6PLkgqADlbCV2QFPCOgQy/w0AMAyLw7qDJEQQfgbgRyv7BQKIrbgDxayR+GgPEAGEkMRAQQePDAUhxBxLfGioGikoQ4ADiD1CxVpgiGNCBSiADGTSxc0js30C8EonPMIkB04AyNDEQmwfKrkSTYyhFFwCCx0C8HE0MBrYC8U90QZABIH+CACjg0A1EBiA5bnRBJiB+wQCRfIsmhwxAUc2PLkgsuATELFD2amQJYgAoPJKAOAGIMxkgCY0kAPIaMl6LKj2yAQC+GzeCOfWV/wAAAABJRU5ErkJggg==>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAZCAYAAAA4/K6pAAAA+0lEQVR4XmNgGLZACF2AWMAIxP+B+CiUBmGSQC8DxBAQkADi3UhyRIFEIN6PLkgqADlbCV2QFPCOgQy/w0AMAyLw7qDJEQQfgbgRyv7BQKIrbgDxayR+GgPEAGEkMVDssCPxUQBIcQcS3xoqBopKEGAD4ldA/BmIJ8MUwYAOA6ZzZdDEQGwFJPYUhBQDwySoIDIoQxO7DsSaUDZIfD2SHEMpVBAZPAbi5WhiIMABxP/QBUEAZABIEgRAAYduIAiAXLUaiAvRJUCACYhfMEA0vkWTQweXgfgBuiAh8BCJ3cmA3YU4ASsDqoaTDJDoJAn0MUA0bmWAeHUUIAEAFGc3l1OJoqoAAAAASUVORK5CYII=>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACQAAAAWCAYAAACosj4+AAABo0lEQVR4Xu2WzSsFURjGXx9FUYqNZGEhNsofgJ2dsiA2LGTHnrKxU2In+QfkK0kWZMHNX0BRVnYU5SN2vsLz3nPO9Z53xtyZaVY3v3q6532eM2fOmTkzd4j+KQHGoG8oB/X7UWZcQK/QCtSkMo8haMC2H8hMLGveoSqoncz4n37sswz1aDNjEi2ylRIekIIbaFWbUfCEzrSZIW7RfMtisUfmgF4dZAiPX/ROdJPp1Gl/ix6QgkMyT1gfmfGP/PiXFvIn4K5SlpxAl6KOXDQH86Lusl6jrauhZ+vNuU4JCHtY+PGX3jDZV0CHCphm5Z2K9ge0Jeo43FLwHFfC4wVznzxLInBMK4/btbY9o7I4cP+NEG9N1E+uMUXBE1xTcADHPvSmTTCrDQGPvxjiVSivAId82ZgGW/8FZzXK27X+gfIdk+T/RfCi1kXtKOzPcuiOzKCPhTjIC1SnTVAP3VP0QkbJ5KwFlaXiHKq07W0ZCHjCaRiENrUZBe+ncTKfJhNkPh00bdCINmNyTObrIjbuUjvt+HGeL20khCcUtq9SU6aNkuAHdJtpPML774cAAAAASUVORK5CYII=>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAYAAAD3Va0xAAAA0klEQVR4XmNgGAWkgtlA/AmI/yPhVygqGBi+IMmBsDeqNCqAKcIGmoD4PLogNsDIADHkFroEEFwGYl90QVwgmwFiUDiSGBMQ/wNiLiQxguAlA6q3DIH4KRKfaIAcPtOg7GMIaeIBSOMFBojLtKB8XAGPE8DC5w+S2BKoWD6SGEHwmgG77SS7CpeGtwwQcUV0CWyAmQGi+DS6BBCoMkDk3qNLYAP9DBDFoegSUABzrSC6BAwsY4Dkr3dQ/JUBkvhgQIYB4hJQWnrMAFF7D0l+FIwCALDWPUOqr0VdAAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHUAAAAUCAYAAACzrHJDAAACvElEQVR4Xu2YS6hOURTHl1fymHi7ySMDZSJJiaQwUXeCMJHEQB4RrshrwsTAQIqJMnK7ZKKUAWViyIQQIgPvSCbKvXHZ/7PX+vY66+59bm5KX61frb6zfuucc++393f22XsTOY7jOI7zrxkX4naI3yHuhxhWLzeyMsQzitd2m5rmOsVznoYYa2rC9hAfQvwMccbULIdC7LLSicyg2NhjOJ/E+fDWGWUOhuhX+U6K12pGsZvF+QjOO1pnRNDp71V+JcRXlYOeEH0Ur0fsrpcd4XuIa8Y9CPHDuBxo2PkZp5+yeyHeqhycpYGdj3xkxq0xTvBObQCNs8m4Y+ybWEv5c3qp7nF8UeVgGXvhnMkFuJdWMt6pBVZQbJzlxm9lP9F4zV3Kd8RrSl6G2hOpXDGb/XrOZUi1wOU88E4tsJ9i4ywyfiP7JcZrvlG+wZ9Q8gv5GO9ezRT2RzgvdV7JA+/UAqcoNs4C49ex32y8ptTgDyn5VXy8L5UrJrC/xHnpXiUP4PdYOQh6kvXI1DRTKY4ybckOil8QT5RmA/vVxmswU801+GNKfh4fY0TQTGZ/mvNS55U8gN9rZQOfQ8xR+TaK98jN8i9Y0U7IO3Wp8VvYY7lTovROfUXJY72L46OpXDGTvYwEWJfm7jVYp9oRoImTVlAaMRYrh6XXc5W3HaMpfqmhzH6PU/6cv5n9ylr1DucWOL0O1qBmR4ASGDGaNlTeUfoB4f9ve/BFzht3i70Gk6dpxuEcO0OGu6lydIp9fx2m+v2nm1yA67KSQe2AlU4k91Qil+UGkGHUnveR4hJGkM7BLpKA5ZK9LvdDQudfVXknDbxOkHcyNjGcAth++8WfaKzcsHaD4n6r5UuIT5Se7rn1coU8mdi5wgz0cr3cArUXFHehcP74ernaSsSEBztUb/gTfxvXOY7jOM7/4g//yM/lyt71RAAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEkAAAAUCAYAAADBYxD1AAABuUlEQVR4Xu2XvytGURjHn0hJYVEGZZMBg9GglCKLScpqIL8pWVhtRlIGg0UhMSD+AYOMBouIwY/BoJCI5+uc897nPO7JdnN1PvXtnuf7PKfOee577r0vUSQSifxtpllD2hQcsT5Zj6x+lXPUs57I1J2yiv10PllnvZHZFDTspwsgV2LHfTa+T9Lf1LKORVxOpq5SeLkn1KQD1o7y9sjUdwsPsWaUda7NPBNq0juZXI/wGqx3JzzEzSIGE6xL5eWaUJNqWGvKa6PkueNwR3ZReRUizj2hJqVxSKa+UXjV1nP6YNWJ/L8AGxvRZgp4Y6H2RCeYFvIbteungxSRP2/bT3v0aiNLsLgxbabwQv4xc8yxrux4kJINrxYqwqCuTMQLrFcRS860kSVY6Lg2FVjghjYp+XVpnindl6A5rdpkmsjMxTF2dLFWRJw5WNCkNgVbrHnlXdvrkhhrfmvSgDYEOIau0dCtn84eLGJKm5YZ+vk1jjuM5gDMCzUj5OeOKjKbwbNA007JndTqFHWIO0QM9lnLyssdm6wH1g2Z44IrPhDxV8WhGyOF4+AoJfPah39hr7MiH4lEIiG+AMJzeQ8UxIT4AAAAAElFTkSuQmCC>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFMAAAAUCAYAAADx7wHUAAACF0lEQVR4Xu2XzUtWQRTGjwZp5KJ0ISa08A8IM0L8IEFbtYnS2oi4c63QKmihf0EELQTBhWiImxZpUC2klrlwFYILKYQi3bhIRPDjPJwZPffhfX1NRPE6P3i4nGfO3PfOeWfmzhVJJBKJxEm4rvqk2lN9V5Vlm4/kvmpVrO8ragN3VV9UTapyVYPqteq9T8oL9WKFuBbimhBj4KWYFMuNjKnWXAweiuV4/c1k5Ih/qmnyFlRb5DG3xQrzhHx43S7uUM2KFXpEVeXacgcG/5y8l8E/irdiObfIh7fh4nbVsItzywOxwWPAnv7gV5PvWRHLuUF+XMqRNrkkxRwUGzheDp5nwW8m3zMuxWemL2aLHO6t71Sbqq+uPTdgD8Mg75CPfRB+L/me+KJ6Sj4XE3/UTxcDtH8jrxi/5fCe69TGdLJxlgyIPWQj+T3B7yKf+SDZwv0KsfcK8UdK54DPqkcubhXrd9N5Eaym/znSnTpxz8RS9PQFH8emUjxWLYkVFsep4xRzXiynlnwGxWTib/hVc1W14+JzoULswU7yNi8G+k1RzPfC0QteJfnMPTYc+LiI94auZJvPBzzIG/Lmgu/BMuKZhJyPLsZ2wf0Qz5C3HfzcUWgWIvYvFuxFhWYY4gkX74p9KnoWxY5HkTqxfv3OyxU4smDPwRUDxZGJwbf0C/JGxfKXw3Uo23zAD7H2OCPxiZlIJBIXiX1XA4fEjBycrAAAAABJRU5ErkJggg==>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAWCAYAAAD5Jg1dAAAAhklEQVR4XmNgGDTAAl0AF0gF4v9AvB1dAht4yQBRTBCYMEAUhqJLYAMghb/QBbGBIwxEWi/EAFFYhy6BDYAUEjT1OhDfZYAoZEGTg4MHQLwSiJkZIAqXochCASgMbyHxsVr/AYi/o4kVMkAUSsEEPkMFsAGQ+CUQQwbKAbkJG9jDgNuQoQkACMggwxLqat0AAAAASUVORK5CYII=>