import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import rlcard
from rlcard.agents import RandomAgent
from src.env.wrappers import AlphaHoldemWrapper
from src.agent.ppo_agent import PPOAgent


def evaluate(agent, env, num_games=20):
    """
    현재 AI(Player 0) vs 랜덤 봇(Player 1) 평가전
    """
    wins = 0
    total_rewards = 0
    
    for _ in range(num_games):
        state, player_id = env.reset()
        done = False
        
        while not done:
            # Player 0: 우리 AI (PPO)
            if player_id == 0:
                # Deterministic=True: 시험 칠 때는 모험하지 않고 최선의 수만 둠
                action, _ = agent.policy.get_action(state, deterministic=True)
            
            # Player 1: 랜덤 봇 (Random)
            else:
                # Wrapper가 처리해주므로 랜덤하게 정수(0~4)만 뽑으면 됨
                # 단, 합법적인 액션 중에서 골라야 함
                # RLCard의 raw state에서 legal_actions 가져오기
                raw_state = env.env.get_state(player_id)
                legal_actions = list(raw_state['legal_actions'].keys())
                # 랜덤 선택 (Wrapper가 0~4 매핑 처리하므로 이 인덱스가 중요)
                # 하지만 우리는 Wrapper의 step(action_idx)를 부르므로
                # Wrapper의 decode 로직에 맞춰야 함.
                # 여기서는 간단히: 랜덤 봇은 그냥 아무거나 던지고 Wrapper가 처리하게 둠
                # (더 정확히는 legal_actions 중 하나를 랜덤 선택)
                action = np.random.choice(legal_actions)

            next_state, next_player_id = env.step(action)
            
            if next_state is None: # 게임 종료
                done = True
                payoffs = env.env.get_payoffs()
                total_rewards += payoffs[0]
                if payoffs[0] > 0:
                    wins += 1
            else:
                state = next_state
                player_id = next_player_id
                
    return wins / num_games * 100, total_rewards / num_games

def run_training(num_episodes=5000, eval_interval=100):
    # 1. TensorBoard 설정
    writer = SummaryWriter("runs/AlphaHoldem_Day7")
    
    # 2. 환경 및 에이전트 설정
    raw_env = rlcard.make('no-limit-holdem', config={'seed': 42})
    env = AlphaHoldemWrapper(raw_env)
    
    # 액션 5개로 통일된 설정
    agent = PPOAgent(input_dim=54, action_dim=5, lr=0.0002, K_epochs=4, eps_clip=0.2)
    
    print(f"🚀 학습 시작! (총 {num_episodes} 에피소드, 텐서보드로 모니터링 중...)")

    # 학습 루프
    for episode in range(1, num_episodes + 1):
        state, player_id = env.reset()
        episode_memory = {0: [], 1: []}
        done = False
        
        while not done:
            # Self-Play: 항상 AI가 행동 결정
            action, probs = agent.policy.get_action(state)
            
            # [Day 6 수정사항 반영] 확률값(Scalar)만 저장
            action_prob = probs[0][action].item()
            
            episode_memory[player_id].append({
                's': state, 'a': action, 'prob': action_prob
            })

            next_state, next_player_id = env.step(action)
            
            if next_state is None:
                done = True
            else:
                state = next_state
                player_id = next_player_id

        # 게임 종료 및 데이터 저장
        payoffs = env.env.get_payoffs()
        for pid in [0, 1]:
            reward = payoffs[pid]
            memory = episode_memory[pid]
            for i, step_data in enumerate(memory):
                s, a, prob = step_data['s'], step_data['a'], step_data['prob']
                ns = memory[i+1]['s'] if i < len(memory)-1 else s
                d = False if i < len(memory)-1 else True
                agent.put_data((s, a, reward, ns, d, prob))

        # --- [Training] 학습 및 Loss 기록 ---
        if len(agent.data) >= 256: # 배치 사이즈가 차면 학습
            loss = agent.train_net()
            writer.add_scalar("Training/Loss", loss, episode)

        # --- [Evaluation] 주기적 평가 ---
        if episode % eval_interval == 0:
            # 랜덤 봇과 50판 대결
            win_rate, avg_reward = evaluate(agent, env, num_games=200)
            
            print(f"Episode {episode}: Eval WinRate = {win_rate:.1f}% | AvgReward = {avg_reward:.2f}")
            
            # 텐서보드에 기록
            writer.add_scalar("Evaluation/WinRate_vs_Random", win_rate, episode)
            writer.add_scalar("Evaluation/AvgReward_vs_Random", avg_reward, episode)
            
            # 모델 저장
            torch.save(agent.policy.state_dict(), "alpha_holdem_day7.pth")

    writer.close()
    print("✅ 학습 종료!")

if __name__ == "__main__":
    run_training()