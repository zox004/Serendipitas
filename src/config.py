# 파일명: src/config.py
import torch
import os

class AlphaHoldemConfig:
    # --- 1. 환경 및 데이터 설정 ---
    NUM_PLAYERS = 2
    CHIPS_FOR_EACH = 100
    MAX_CHIPS = CHIPS_FOR_EACH * NUM_PLAYERS 
    
    # [Card Tensor] (6, 4, 13)
    # Ch: MyHand, Flop, Turn, River, AllPublic, Bias
    CARD_CHANNELS = 6
    CARD_HEIGHT = 4
    CARD_WIDTH = 13
    
    # [Betting History Tensor] (24, 4, 5)
    # 24: Bit Encoding Depth (Amount)
    # 4: Rounds (Pre, Flop, Turn, River)
    # 5: Actions (Fold, Check/Call, Raise Half, Raise Pot, All-in)
    HIST_CHANNELS = 24
    HIST_HEIGHT = 4
    HIST_WIDTH = 5
    
    ACTION_DIM = 5
    
    # --- 2. 학습 하이퍼파라미터 ---
    HIDDEN_DIM = 256
    NUM_RES_BLOCKS = 3    
    
    LR = 0.0002           
    GAMMA = 0.99          
    K_EPOCHS = 4          
    EPS_CLIP = 0.2        
    BATCH_SIZE = 128      
    
    NUM_EPISODES = 50000  
    EVAL_INTERVAL = 100   
    REWARD_SCALE = 100.0
    
    # --- 3. 시스템 설정 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    LOG_DIR = "runs/AlphaHoldem_v2.0-Siamese"
    
    CHECKPOINT_DIR = "checkpoints"
    
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        
    MODEL_PATH = os.path.join(CHECKPOINT_DIR, "alpha_holdem_siamese.pth")