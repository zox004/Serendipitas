import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import rlcard
from src.env.wrappers import AlphaHoldemWrapper
from src.agent.ppo_agent import PPOAgent

# [수정됨] 평가(Evaluate) 때는 실제 칩 개수를 그대로 봅니다 (직관성을 위해)
def evaluate(agent, env, num_games=20):
    wins = 0
    total_rewards = 0
    
    for _ in range(num_games):
        state, player_id = env.reset()
        done = False
        
        while not done:
            if player_id == 0:
                action, _ = agent.policy.get_action(state, deterministic=True)
            else:
                raw_state = env.env.get_state(player_id)
                if isinstance(raw_state['legal_actions'], dict):
                    legal_actions = list(raw_state['legal_actions'].keys())
                else:
                    legal_actions = raw_state['legal_actions']
                action = np.random.choice(legal_actions)

            next_state, next_player_id = env.step(action)
            
            if next_state is None:
                done = True
                payoffs = env.env.get_payoffs()
                total_rewards += payoffs[0] # 여기는 나누기 안 함 (실제 칩 확인)
                if payoffs[0] > 0:
                    wins += 1
            else:
                state = next_state
                player_id = next_player_id
                
    return wins / num_games * 100, total_rewards / num_games

def run_training(num_episodes=5000, eval_interval=100):
    writer = SummaryWriter("runs/AlphaHoldem_Day9") # 로그 폴더 변경
    
    # RLCard 기본 칩 설정 (chips_for_each=100)
    reward_scale = 100.0 
    
    raw_env = rlcard.make('no-limit-holdem', config={'seed': 42})
    env = AlphaHoldemWrapper(raw_env)
    
    # Input=107, Action=5
    agent = PPOAgent(input_dim=107, action_dim=5, lr=0.0002, K_epochs=4, eps_clip=0.2)
    
    print(f"🚀 학습 시작! (Reward Scale: 1/{reward_scale} 적용)")

    for episode in range(1, num_episodes + 1):
        state, player_id = env.reset()
        episode_memory = {0: [], 1: []}
        done = False
        
        while not done:
            action, probs = agent.policy.get_action(state)
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

        # --- [핵심 수정 구간] 보상 정규화 적용 ---
        payoffs = env.env.get_payoffs()
        
        for pid in [0, 1]:
            # 실제 칩(+100)을 스케일링(+1.0)해서 학습 데이터에 넣음
            raw_reward = payoffs[pid]
            normalized_reward = raw_reward / reward_scale 
            
            memory = episode_memory[pid]
            for i, step_data in enumerate(memory):
                s, a, prob = step_data['s'], step_data['a'], step_data['prob']
                ns = memory[i+1]['s'] if i < len(memory)-1 else s
                d = False if i < len(memory)-1 else True
                
                # 정규화된 보상(normalized_reward) 주입
                agent.put_data((s, a, normalized_reward, ns, d, prob))

        if len(agent.data) >= 256:
            loss = agent.train_net()
            writer.add_scalar("Training/Loss", loss, episode)

        if episode % eval_interval == 0:
            # 평가는 기존처럼 (사람이 보기 편하게)
            win_rate, avg_reward = evaluate(agent, env, num_games=200)
            
            print(f"Episode {episode}: Eval WinRate = {win_rate:.1f}% | AvgReward = {avg_reward:.2f} chips")
            
            writer.add_scalar("Evaluation/WinRate_vs_Random", win_rate, episode)
            writer.add_scalar("Evaluation/AvgReward_vs_Random", avg_reward, episode)
            
            torch.save(agent.policy.state_dict(), "alpha_holdem_day9.pth")

    writer.close()
    print("✅ 학습 종료!")

if __name__ == "__main__":
    run_training()