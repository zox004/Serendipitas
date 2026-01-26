# 파일명: tests/test_day3.py
import sys
import os

# 프로젝트 루트 경로 추가 (모듈 import를 위해)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rlcard
from src.env.wrappers import AlphaHoldemWrapper

def test_environment():
    print(">>> AlphaHoldem 환경 테스트 시작...")

    # 1. RLCard 원본 환경 생성
    raw_env = rlcard.make('no-limit-holdem', config={'seed': 42})
    
    # 2. Wrapper 적용
    env = AlphaHoldemWrapper(raw_env)
    print("✅ Wrapper 적용 완료")

    # 3. Reset 테스트
    state_tensor, player_id = env.reset()
    print(f"✅ Reset 완료. Player ID: {player_id}")
    print(f"   - Observation Shape: {state_tensor.shape} (기대값: [1, 54])")
    print(f"   - Data Type: {state_tensor.dtype}")

    # 4. Step 테스트 (Random Action)
    print("\n>>> 임의의 행동(Check/Call) 실행 중...")
    # Action 1: Check/Call
    next_state, next_player_id = env.step(1) 
    
    if next_state is not None:
        print(f"✅ Step 실행 성공. Next Player: {next_player_id}")
        print(f"   - Next State Shape: {next_state.shape}")
    else:
        print("✅ 게임 종료 (Game Over)")

    print("\n>>> 모든 테스트 통과! Day 3 완료.")

if __name__ == "__main__":
    test_environment()