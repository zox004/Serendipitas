import sys
import os
import torch

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.network import AlphaHoldemNetwork

def test_network():
    print(">>> AlphaHoldem 신경망(Brain) 테스트 시작...")

    # 1. 네트워크 생성
    input_dim = 54
    action_dim = 6
    model = AlphaHoldemNetwork(input_dim, action_dim)
    print(f"✅ 모델 생성 완료 (Input: {input_dim}, Action: {action_dim})")

    # 2. 가상의 입력 데이터 생성 (Day 3 Wrapper가 주는 것과 동일한 형태)
    # Batch Size=1, Feature=54
    dummy_input = torch.randn(1, input_dim)
    print(f"✅ 더미 입력 생성: {dummy_input.shape}")

    # 3. 순전파 (Forward) 테스트
    logits, value = model(dummy_input)
    
    print("\n[출력 결과 확인]")
    print(f"1. Actor Output (Logits): {logits.shape} (기대값: [1, 6])")
    print(f"   값 예시: {logits.detach().numpy()}")
    print(f"2. Critic Output (Value): {value.shape} (기대값: [1, 1])")
    print(f"   값 예시: {value.item():.4f}")

    # 4. 행동 선택 테스트
    action, probs = model.get_action(dummy_input)
    print(f"\n[행동 선택 테스트]")
    print(f"   선택된 행동(Index): {action}")
    print(f"   행동 확률 분포: {probs.detach().numpy()}")

    if logits.shape == (1, 6) and value.shape == (1, 1):
        print("\n✅ 테스트 통과! 뇌 구조가 정상입니다.")
    else:
        print("\n❌ 테스트 실패: 출력 크기가 맞지 않습니다.")

if __name__ == "__main__":
    test_network()