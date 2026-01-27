import os
import torch
import rlcard
from rlcard.agents import RandomAgent
from src.env.wrappers import AlphaHoldemWrapper
from src.agent.ppo_agent import PPOAgent

def run_training(num_episodes=1000, print_interval=100):
    # 1. 환경 및 에이전트 준비
    # RLCard의 기본 설정을 가져오되, 우리가 만든 Wrapper로 감쌉니다.
    raw_env = rlcard.make('no-limit-holdem', config={'seed': 42})
    env = AlphaHoldemWrapper(raw_env)

    # 우리의 주인공 AlphaHoldem (입력 54, 행동 5)
    agent = PPOAgent(input_dim=54, action_dim=5, lr=0.0002, K_epochs=4)

    # 기록용 변수
    total_rewards = 0
    wins = 0

    print(f"🚀 학습 시작! (총 {num_episodes} 에피소드)")

    for episode in range(1, num_episodes + 1):
        # 게임 시작 (환경 초기화)
        state, player_id = env.reset()
        
        # 이번 판의 데이터를 임시 저장할 리스트 (Player 0, 1 별도 관리)
        # 구조: [state, action, prob]
        episode_memory = {0: [], 1: []}
        
        done = False
        
        # --- [Game Loop] 게임이 끝날 때까지 진행 ---
        while not done:
            # 1. 현재 플레이어의 행동 결정
            # (Self-Play: 두 플레이어 모두 같은 Agent 사용)
            action, probs = agent.policy.get_action(state)
            
            # [수정 포인트] 전체 확률 텐서(probs) 대신, 
            # 실제로 선택한 행동의 확률값(Scalar Float)만 추출하여 저장해야 함
            prob_a = probs[0][action].item() # .item()으로 순수 float 변환
            
            # 2. 임시 메모리에 '상태, 행동, 확률' 저장 (보상은 아직 모름)
            episode_memory[player_id].append({
                's': state,
                'a': action,
                'prob': prob_a
            })

            # 3. 환경에 행동 적용
            next_state, next_player_id = env.step(action)
            
            # 4. 상태 업데이트
            # (주의: next_state가 None이면 게임 끝)
            if next_state is None:
                done = True
            else:
                state = next_state
                player_id = next_player_id

        # --- [Game Over] 게임 종료 후 보상 처리 ---
        # RLCard에서 최종 승패 보상 가져오기 (예: [1.0, -1.0])
        payoffs = env.env.get_payoffs() 
        
        # 각 플레이어의 기억을 되살려 학습 데이터(Transition) 생성
        for pid in [0, 1]:
            reward = payoffs[pid]
            memory = episode_memory[pid]
            
            for i, step_data in enumerate(memory):
                s = step_data['s']
                a = step_data['a']
                prob = step_data['prob']
                
                # 다음 상태(Next State) 정의
                # 포커는 내 턴 -> 상대 턴 -> 내 턴 이므로, 
                # 바로 다음 데이터가 나의 Next State가 됨 (단, 마지막 턴은 종료 상태)
                if i < len(memory) - 1:
                    ns = memory[i+1]['s']
                    d = False
                else:
                    ns = s # 마지막 상태는 큰 의미 없음 (done=True라 무시됨)
                    d = True
                
                # PPO 에이전트에 데이터 주입
                # (중요: PPO는 Step별 보상보다, 게임 종료 보상을 주로 사용)
                # 여기서는 단순화를 위해 마지막 스텝에만 큰 보상을 주고 나머진 0 처리 가능하지만,
                # 우선 모든 스텝에 최종 보상을 할당 (Monte Carlo 방식)
                agent.put_data((s, a, reward, ns, d, prob))

        # 기록 업데이트 (Player 0 기준)
        total_rewards += payoffs[0]
        if payoffs[0] > 0: wins += 1

        # --- [Training] 일정 데이터가 모이면 학습 수행 ---
        # 에피소드마다 바로 학습하거나, 배치를 모아서 할 수 있음.
        # 여기서는 32 에피소드마다 학습 진행
        if len(agent.data) >= 200: # 약 3~4게임 분량의 턴 데이터
            loss = agent.train_net()

        # --- [Logging] 진행 상황 출력 ---
        if episode % print_interval == 0:
            avg_reward = total_rewards / print_interval
            win_rate = wins / print_interval * 100
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Win Rate (P0) = {win_rate:.1f}%")
            total_rewards = 0
            wins = 0

    # 학습 완료 후 모델 저장
    save_path = "alpha_holdem_day6.pth"
    torch.save(agent.policy.state_dict(), save_path)
    print(f"✅ 학습 종료! 모델 저장 완료: {save_path}")

if __name__ == "__main__":
    run_training()