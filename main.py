import time
import rlcard
from rlcard.agents import RandomAgent

def benchmark(num_hands=10000):
    # 1. 환경 생성 (Heads-Up No-Limit Hold'em)
    env = rlcard.make('no-limit-holdem', config={'seed': 42})

    # 2. 랜덤 에이전트 설정 (단순 속도 측정용)
    agents = [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players)]
    env.set_agents(agents)

    print(f"Starting benchmark for {num_hands} hands...")
    start_time = time.time()

    for _ in range(num_hands):
        trajectories, payoffs = env.run(is_training=False)

    end_time = time.time()
    duration = end_time - start_time
    fps = num_hands / duration

    print(f"-" * 30)
    print(f"Total Time: {duration:.4f} seconds")
    print(f"Speed: {fps:.2f} hands/second")
    print(f"-" * 30)

    # 목표 달성 여부 확인
    if fps > 1000:
        print("✅ Status: PASS (충분한 속도입니다)")
    else:
        print("⚠️ Status: WARNING (최적화가 필요할 수 있습니다)")

if __name__ == '__main__':
    benchmark()