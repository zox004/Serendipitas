import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from src.config import AlphaHoldemConfig as cfg

class ConvBlock(nn.Module):
    """
    기본 컨볼루션 블록: Conv2d -> BatchNorm -> ReLU
    Padding=1을 사용하여 입력의 H, W 크기를 유지합니다.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """
    ResNet의 핵심 블록: 입력을 출력에 더해주는 Skip Connection 포함
    구조: Input -> [Conv -> BN -> ReLU -> Conv -> BN] + Input -> ReLU
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip Connection (핵심)
        out = F.relu(out)
        return out

class AlphaHoldemResNet(nn.Module):
    def __init__(self):
        super(AlphaHoldemResNet, self).__init__()
        
        # ==========================================
        # Branch 1: Card Processing (CNN)
        # Input: (Batch, 6, 4, 13)
        # ==========================================
        self.card_conv = ConvBlock(cfg.CARD_CHANNELS, 64)
        self.card_res = nn.Sequential(*[
            ResidualBlock(64) for _ in range(cfg.NUM_RES_BLOCKS)
        ])
        # Flatten Dimension: 64 * 4 * 13 = 3328
        self.card_flat_dim = 64 * cfg.CARD_HEIGHT * cfg.CARD_WIDTH
        
        # ==========================================
        # Branch 2: Betting History Processing (CNN)
        # Input: (Batch, 24, 4, 5)
        # ==========================================
        self.hist_conv = ConvBlock(cfg.HIST_CHANNELS, 64)
        self.hist_res = nn.Sequential(*[
            ResidualBlock(64) for _ in range(cfg.NUM_RES_BLOCKS)
        ])
        # Flatten Dimension: 64 * 4 * 5 = 1280
        self.hist_flat_dim = 64 * cfg.HIST_HEIGHT * cfg.HIST_WIDTH
        
        # ==========================================
        # Merge & Heads
        # ==========================================
        total_dim = self.card_flat_dim + self.hist_flat_dim # 3328 + 1280 = 4608
        
        self.fc_merge = nn.Linear(total_dim, cfg.HIDDEN_DIM)
        
        # Actor Head (행동 확률)
        self.actor_head = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM, cfg.ACTION_DIM),
            nn.Softmax(dim=-1)
        )
        
        # Critic Head (가치 평가)
        self.critic_head = nn.Linear(cfg.HIDDEN_DIM, 1)

        # 가중치 초기화 (학습 안정성 향상)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: (card_tensor, hist_tensor) 튜플
        """
        card_x, hist_x = x
        
        # 배치 차원 확인 및 추가 (Single inference 대응)
        if card_x.dim() == 3: card_x = card_x.unsqueeze(0)
        if hist_x.dim() == 3: hist_x = hist_x.unsqueeze(0)
        
        # 1. Card Branch Forward
        c = self.card_conv(card_x)
        c = self.card_res(c)
        c = c.view(c.size(0), -1) # Flatten (Batch, 3328)
        
        # 2. History Branch Forward
        h = self.hist_conv(hist_x)
        h = self.hist_res(h)
        h = h.view(h.size(0), -1) # Flatten (Batch, 1280)
        
        # 3. Merge & Heads
        combined = torch.cat([c, h], dim=1)        # (Batch, 4608)
        features = F.relu(self.fc_merge(combined)) # (Batch, 256)
        
        probs = self.actor_head(features)
        value = self.critic_head(features)
        
        return probs, value

    def get_action(self, x, deterministic=False):
        """
        외부에서 호출하는 행동 결정 함수
        """
        is_training = self.training
        
        # 추론 모드 전환 (BatchNorm 고정 등)
        self.eval()
        
        with torch.no_grad():
            probs, _ = self.forward(x)
            m = Categorical(probs)
            
            if deterministic:
                action = torch.argmax(probs, dim=1)
            else:
                action = m.sample()
        
        # 학습 중이었다면 다시 학습 모드로 복구
        if is_training:
            self.train()
            
        return action.item(), probs