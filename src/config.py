# 파일명: src/config.py
import torch

class AlphaHoldemConfig:
    # --- 1. 환경 및 데이터 설정 ---
    NUM_PLAYERS = 2
    CHIPS_FOR_EACH = 100  # 1인당 칩 개수
    # 전체 칩 총량 (Max Pot) = 100 * 2 = 200
    MAX_CHIPS = CHIPS_FOR_EACH * NUM_PLAYERS 
    
    # Input: 내패(52) + 보드(52) + 칩정보(3) = 107
    INPUT_DIM = 107
    # Output: Fold, Check/Call, Min, Pot, All-in (5개)
    ACTION_DIM = 5
    
    # --- 2. 학습(Training) 하이퍼파라미터 ---
    LR = 0.0002           # 학습률
    GAMMA = 0.99          # 할인율 (미래 보상의 가치)
    K_EPOCHS = 4          # PPO 업데이트 반복 횟수
    EPS_CLIP = 0.2        # PPO 클리핑 비율 (20%)
    BATCH_SIZE = 256      # 학습 배치 사이즈
    
    NUM_EPISODES = 10000  # 총 학습 에피소드
    EVAL_INTERVAL = 100   # 평가 주기
    
    # 보상 스케일링 (100칩 -> 보상 1.0)
    REWARD_SCALE = 100.0
    
    # --- 3. 시스템 설정 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    LOG_DIR = "runs/AlphaHoldem_Day10"
    MODEL_PATH = "alpha_holdem_day10.pth"