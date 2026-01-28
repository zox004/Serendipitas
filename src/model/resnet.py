import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical # [추가됨] 확률 분포 계산용
from src.config import AlphaHoldemConfig as cfg

class ResidualBlock(nn.Module):
    """
    ResNet의 핵심 건물 블록 (층)
    입력 -> [Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm] + 입력 -> ReLU
    """
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x  # 입력을 기억해둠 (지름길)
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        out += residual  # 기억해둔 원본을 더함 (핵심!)
        out = F.relu(out)
        return out

class AlphaHoldemResNet(nn.Module):
    def __init__(self):
        super(AlphaHoldemResNet, self).__init__()
        
        # 1. 입력층 (Input -> Hidden)
        self.input_layer = nn.Sequential(
            nn.Linear(cfg.INPUT_DIM, cfg.HIDDEN_DIM),
            nn.ReLU()
        )
        
        # 2. ResNet 블록 쌓기 (몸통)
        blocks = []
        for _ in range(cfg.NUM_RES_BLOCKS):
            blocks.append(ResidualBlock(cfg.HIDDEN_DIM))
        self.res_blocks = nn.Sequential(*blocks)
        
        # 3-1. Actor Head (행동 결정: 확률 출력)
        self.actor_head = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM, cfg.ACTION_DIM),
            nn.Softmax(dim=-1)
        )
        
        # 3-2. Critic Head (가치 판단: 점수 출력)
        self.critic_head = nn.Linear(cfg.HIDDEN_DIM, 1)

    def forward(self, x):
        # x shape: (batch, input_dim)
        
        # 입력층 통과
        out = self.input_layer(x)
        
        # ResNet은 배치 단위(여러 개) 처리가 기본이라 차원 문제 방지
        if out.dim() == 1: 
             out = out.unsqueeze(0)
             
        # ResNet 블록 통과
        out = self.res_blocks(out)
        
        probs = self.actor_head(out)
        value = self.critic_head(out)
        
        return probs, value

    def get_action(self, x, deterministic=False):
            """
            상태(x)를 받아서 실제로 할 행동(action)을 결정하는 함수
            """
            # [수정됨] BatchNorm 에러 방지 코드
            # 1. 현재 모드 저장 (Train 모드인지 확인)
            is_training = self.training
            
            # 2. 평가 모드로 전환 (Batch Size가 1이어도 에러 안 남)
            self.eval()
            
            # 3. 그라디언트 계산 끄기 (속도 향상 및 메모리 절약)
            with torch.no_grad():
                # 1. 신경망을 통과시켜 확률(probs)을 얻음
                probs, _ = self.forward(x)
                
                # 2. 확률 분포 생성
                m = Categorical(probs)
                
                # 3. 행동 선택
                if deterministic:
                    action = torch.argmax(probs, dim=1)
                else:
                    action = m.sample()
            
            # 4. 원래 모드로 복구 (나중에 학습을 위해)
            if is_training:
                self.train()
                
            return action.item(), probs