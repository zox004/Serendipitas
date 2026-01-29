# 파일명: src/utils.py
import torch
import os
import glob
from src.config import AlphaHoldemConfig as cfg

def save_checkpoint(agent, episode, win_rate):
    """
    현재 에이전트의 뇌(모델)를 파일로 저장합니다.
    파일명 예시: checkpoints/model_ep1500_win72.pth
    """
    filename = f"model_ep{episode}_win{int(win_rate)}.pth"
    path = os.path.join(cfg.CHECKPOINT_DIR, filename)
    
    torch.save(agent.policy.state_dict(), path)
    print(f"💾 Checkpoint saved: {path}")

def load_checkpoint(agent, filename):
    """
    특정 파일(과거의 나)을 불러와서 에이전트에게 덮어씌웁니다.
    """
    path = os.path.join(cfg.CHECKPOINT_DIR, filename)
    if os.path.exists(path):
        agent.policy.load_state_dict(torch.load(path, map_location=cfg.DEVICE))
        agent.policy.eval() # 불러온 모델은 보통 '상대방'용이므로 평가 모드로 설정
        print(f"📂 Loaded model from {path}")
        return True
    else:
        print(f"❌ File not found: {path}")
        return False

def get_all_checkpoints():
    """
    checkpoints 폴더에 있는 모든 .pth 파일 목록을 가져옵니다.
    """
    files = glob.glob(os.path.join(cfg.CHECKPOINT_DIR, "*.pth"))
    return sorted(files)