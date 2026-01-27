import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_selfplay import run_training

def test_selfplay_loop():
    print(">>> AlphaHoldem Self-Play 통합 테스트 시작...")

    try:
        run_training(num_episodes=100, print_interval=10)
        print("\n✅ 테스트 통과! Self-Play 루프가 정상 작동합니다.")
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        raise e

if __name__ == "__main__":
    test_selfplay_loop()