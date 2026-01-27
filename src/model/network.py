import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaHoldemNetwork(nn.Module):
    """
    AlphaHoldem의 두뇌: Actor-Critic 아키텍처
    - Input: 54차원 상태 벡터 (Day 3에서 정의함)
    - Actor Output: 6개 행동에 대한 확률 (Logits)
    - Critic Output: 현재 상태의 가치 (Value)
    """
    def __init__(self, input_dim=54, action_dim=5, hidden_dim=256):
        super(AlphaHoldemNetwork, self).__init__()
        
        # 1. Feature Extractor (공통 정보를 처리하는 앞단)
        # 포커의 복잡한 상관관계를 이해하기 위해 3층 MLP 구조 사용
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 2. Actor Head (행동 결정)
        # 6개의 행동 중 무엇을 선택할지 점수(Logit)를 냄
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        
        # 3. Critic Head (가치 판단)
        # 현재 상태가 얼마나 좋은지 숫자 하나(Scalar)로 냄
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        순전파 (Forward Pass)
        state -> features -> (action_logits, value)
        """
        # 공통 특징 추출
        features = self.feature_layer(state)
        
        # Actor: 행동 확률 계산을 위한 Logit 값 (Softmax 전 단계)
        action_logits = self.actor_head(features)
        
        # Critic: 상태 가치 (Value)
        value = self.critic_head(features)
        
        return action_logits, value

    def get_action(self, state, deterministic=False):
        """
        실제 게임에서 행동을 선택할 때 사용
        """
        action_logits, _ = self.forward(state)
        probs = F.softmax(action_logits, dim=-1)
        
        if deterministic:
            # 테스트 모드: 가장 확률 높은 행동 선택
            action = torch.argmax(probs, dim=-1)
        else:
            # 학습 모드: 확률 분포에 따라 샘플링 (탐험)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
        return action.item(), probs