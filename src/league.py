# src/league.py (전체 코드 수정)
import os
import glob
import random
import torch
from src.config import AlphaHoldemConfig as cfg
from src.model.resnet import AlphaHoldemResNet

class LeagueManager:
    def __init__(self):
        self.opponents = {} # 로드된 모델들을 저장할 딕셔너리
        self.refresh_pool()
        
    def refresh_pool(self):
        """
        checkpoints 폴더를 스캔해서 새로운 모델이 있으면 로드합니다.
        """
        if not os.path.exists(cfg.CHECKPOINT_DIR):
            os.makedirs(cfg.CHECKPOINT_DIR)
            
        files = glob.glob(os.path.join(cfg.CHECKPOINT_DIR, "*.pth"))
        
        for f in files:
            # best나 latest가 아닌 '과거 기록(ep)' 파일들만 영입
            if f not in self.opponents and "best" not in f and "latest" not in f:
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
        [수정된 비율]
        - 40%: 랜덤 봇 (무지성 뻥카 참교육 담당)
        - 10%: 과거의 나 (다양한 전략 경험)
        - 50%: Self-Play (최신 전략 연구)
        """
        rand = random.random()
        
        # [수정됨] 랜덤 봇 비중을 20% -> 40%로 대폭 상향
        if rand < 0.4:
            return "random"
        
        # 과거 모델 비중은 30% -> 10%로 축소 (0.4 ~ 0.5 구간)
        elif rand < 0.5 and len(self.opponents) > 0:
            filename = random.choice(list(self.opponents.keys()))
            return self.opponents[filename]
        
        # 나머지 50%는 자가 대전
        else:
            return None