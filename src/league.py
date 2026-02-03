# src/league.py
import os
import glob
import random
import torch
from src.config import AlphaHoldemConfig as cfg
from src.model.resnet import AlphaHoldemResNet

class LeagueManager:
    def __init__(self):
        self.opponents = {}  # {파일경로: 모델객체} 딕셔너리
        self.refresh_pool()
        
    def refresh_pool(self):
        """
        checkpoints 폴더를 스캔해서 모든 모델을 로드합니다.
        """
        if not os.path.exists(cfg.CHECKPOINT_DIR):
            os.makedirs(cfg.CHECKPOINT_DIR)
            
        files = glob.glob(os.path.join(cfg.CHECKPOINT_DIR, "*.pth"))
        
        for f in files:
            try:
                model = AlphaHoldemResNet().to(cfg.DEVICE)
                model.load_state_dict(torch.load(f, map_location=cfg.DEVICE))
                model.eval() 
                self.opponents[f] = model
                print(f"🥊 리그 선수 등록 완료: {os.path.basename(f)}")
            except Exception as e:
                print(f"⚠️ 모델 로드 실패 (파일: {f}): {e}")

    def get_opponent(self):
        """
        이번 판에 싸울 상대를 결정합니다.
        과거 모델 중에서 랜덤 선택 (latest 포함).
        Returns:
            (opponent, opponent_info): (상대 모델, 상대 정보 문자열)
        """
        if len(self.opponents) == 0:
            # 모델이 없으면 None 반환 (에러 방지)
            return None, "No-Opponent"
        
        # 과거 모델 중 랜덤 선택
        filename = random.choice(list(self.opponents.keys()))
        model = self.opponents[filename]
        model_name = os.path.basename(filename)
        return model, f"Past-Model({model_name})"