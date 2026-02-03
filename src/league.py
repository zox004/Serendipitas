# src/league.py (Day 15: Pure Self-Play Mode)
import os
import glob
import random
import torch
from src.config import AlphaHoldemConfig as cfg
from src.model.resnet import AlphaHoldemResNet

class LeagueManager:
    def __init__(self):
        # Day 15: 이제 외부 선수는 영입하지 않습니다.
        self.opponents = {} 
        print("🔒 리그가 폐쇄되었습니다. 오직 'Self-Play'만 진행합니다.")

    def refresh_pool(self):
        # 아무것도 하지 않음 (과거 모델 로드 X)
        pass

    def get_opponent(self):
        """
        무조건 None을 반환하여 Self-Play를 강제합니다.
        """
        return None