# 파일명: src/config.py
import torch
import os
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
    # --- [New] 모델 구조 설정 ---
    HIDDEN_DIM = 256      # 뉴런 개수 (방 크기)
    NUM_RES_BLOCKS = 3    # ResNet 블록 개수 (아파트 층수)
    
    LR = 0.0002           # 학습률
    GAMMA = 0.99          # 할인율 (미래 보상의 가치)
    K_EPOCHS = 4          # PPO 업데이트 반복 횟수
    EPS_CLIP = 0.2        # PPO 클리핑 비율 (20%)
    BATCH_SIZE = 256      # 학습 배치 사이즈
    
    NUM_EPISODES = 50000  # 총 학습 에피소드
    EVAL_INTERVAL = 100   # 평가 주기
    
    # 보상 스케일링 (100칩 -> 보상 1.0)
    REWARD_SCALE = 100.0
    
    # --- 3. 시스템 설정 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    LOG_DIR = "runs/AlphaHoldem_Day15"
        
    # [New] 체크포인트 설정
    CHECKPOINT_DIR = "checkpoints"  # 저장할 폴더 이름
    HISTORY_INTERVAL = 1000         # 몇 판마다 박제할 것인가? (1000판 추천)
    
    # 폴더가 없으면 자동으로 만듭니다
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        
    MODEL_PATH = os.path.join(CHECKPOINT_DIR, "alpha_holdem_latest.pth")