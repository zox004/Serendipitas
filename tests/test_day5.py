import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.ppo_agent import PPOAgent

def test_ppo_agent():
    print(">>> PPO Agent 학습 테스트 시작...")
    
    # 1. 에이전트 생성
    agent = PPOAgent()
    print("✅ 에이전트 생성 완료")
    
    # 2. 가상의 데이터(Transition) 만들기
    # 상황: 3번 정도 게임을 진행했다고 가정
    print(">>> 더미 데이터 주입 중...")
    for _ in range(32): # 배치 사이즈 32
        s = torch.randn(1, 54)       # 현재 상태
        a = 1                        # 행동 (Check/Call)
        r = 1.0                      # 보상 (이김)
        ns = torch.randn(1, 54)      # 다음 상태
        done = False                 # 게임 안 끝남
        prob_a = 0.2                 # 그때 그 행동을 할 확률
        
        agent.put_data((s, a, r, ns, done, prob_a))
        
    # 3. 학습 실행 (Train Net)
    print(">>> 학습 시작 (Gradient Update)...")
    initial_weight = agent.policy.actor_head.weight.data.clone() # 학습 전 가중치 복사
    
    loss = agent.train_net()
    
    final_weight = agent.policy.actor_head.weight.data # 학습 후 가중치
    
    print(f"✅ 학습 완료. Loss 값: {loss:.6f}")
    
    # 4. 가중치 변화 확인
    if not torch.equal(initial_weight, final_weight):
        print("✅ 신경망 가중치가 변경되었습니다. (학습 정상 작동)")
    else:
        print("❌ 경고: 가중치가 변하지 않았습니다. (학습 실패)")

if __name__ == "__main__":
    test_ppo_agent()