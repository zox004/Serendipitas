import os
import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import rlcard

from src.config import AlphaHoldemConfig as cfg
from src.env.wrappers import AlphaHoldemWrapper
from src.agent.ppo_agent import PPOAgent
from src.league import LeagueManager
from src.utils import save_checkpoint

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
                total_rewards += payoffs[0]
                if payoffs[0] > 0: wins += 1
            else:
                state = next_state
                player_id = next_player_id
    return wins / num_games * 100, total_rewards / num_games

def run_training():
    writer = SummaryWriter(cfg.LOG_DIR)
    
    raw_env = rlcard.make('no-limit-holdem', config={'seed': cfg.SEED})
    env = AlphaHoldemWrapper(raw_env)
    
    agent = PPOAgent(
        input_dim=cfg.INPUT_DIM, 
        action_dim=cfg.ACTION_DIM, 
        lr=cfg.LR, 
        K_epochs=cfg.K_EPOCHS, 
        eps_clip=cfg.EPS_CLIP
    )
    
    league = LeagueManager()
    
    best_win_rate = -1.0
    print(f"🚀 리그 트레이닝 시작! (Opponents: {len(league.opponents)} models)")

    for episode in range(1, cfg.NUM_EPISODES + 1):
        # --- [1] 이번 판의 상대 결정 ---
        opponent = league.get_opponent()
        
        state, player_id = env.reset()
        episode_memory = {0: [], 1: []} 
        done = False
        train_player_id = random.choice([0, 1]) 
        
        while not done:
            # --- [2] 행동 결정 ---
            is_training_turn = False 

            if opponent is None:
                action, probs = agent.policy.get_action(state)
                is_training_turn = True
            else:
                if player_id == train_player_id:
                    action, probs = agent.policy.get_action(state)
                    is_training_turn = True
                else:
                    if opponent == "random":
                        raw_state = env.env.get_state(player_id)
                        if isinstance(raw_state['legal_actions'], dict):
                            legal_actions = list(raw_state['legal_actions'].keys())
                        else:
                            legal_actions = raw_state['legal_actions']
                        action = random.choice(legal_actions)
                        probs = None 
                    else:
                        action, _ = opponent.get_action(state, deterministic=True)
                        probs = None

            # --- [3] 데이터 저장 ---
            if is_training_turn:
                action_prob = probs[0][action].item()
                episode_memory[player_id].append({
                    's': state, 'a': action, 'prob': action_prob
                })

            # --- [4] 환경 진행 ---
            next_state, next_player_id = env.step(action)
            
            if next_state is None:
                done = True
            else:
                state = next_state
                player_id = next_player_id

        # --- [5] 보상 계산 및 PPO 데이터 주입 ---
        payoffs = env.env.get_payoffs()
        players_to_train = [0, 1] if opponent is None else [train_player_id]
        
        for pid in players_to_train:
            normalized_reward = payoffs[pid] / cfg.REWARD_SCALE
            memory = episode_memory[pid]
            for i, step_data in enumerate(memory):
                s, a, prob = step_data['s'], step_data['a'], step_data['prob']
                ns = memory[i+1]['s'] if i < len(memory)-1 else s
                d = False if i < len(memory)-1 else True
                agent.put_data((s, a, normalized_reward, ns, d, prob))

        # --- [6] 학습 수행 ---
        if len(agent.data) >= cfg.BATCH_SIZE:
            loss = agent.train_net()
            writer.add_scalar("Training/Loss", loss, episode)
            # [삭제됨] 여기에 있던 refresh_pool은 실행이 보장되지 않아 제거함

        # --- [7] 평가 및 저장, 그리고 리그 갱신 ---
        if episode % cfg.EVAL_INTERVAL == 0:
            win_rate, avg_reward = evaluate(agent, env, num_games=200)
            
            opp_name = "Self-Play"
            if opponent == "random": opp_name = "Random"
            elif opponent is not None: opp_name = "Past-Model"
            
            print(f"Ep {episode}: WR={win_rate:.1f}% | R={avg_reward:.2f} | Opp={opp_name}")
            writer.add_scalar("Evaluation/WinRate_vs_Random", win_rate, episode)
            writer.add_scalar("Evaluation/AvgReward_vs_Random", avg_reward, episode)
            
            torch.save(agent.policy.state_dict(), cfg.MODEL_PATH)
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_path = os.path.join(cfg.CHECKPOINT_DIR, "alpha_holdem_best.pth")
                torch.save(agent.policy.state_dict(), best_path)
                print(f"🏆 최고 승률 갱신! ({best_win_rate:.1f}%)")

            if episode % cfg.HISTORY_INTERVAL == 0:
                save_checkpoint(agent, episode, win_rate)
                
                # [위치 이동 완료]
                # 체크포인트 저장 직후에 풀을 갱신해야 가장 확실하게 로드됩니다.
                print("🔄 리그 선수 명단 갱신 중...")
                league.refresh_pool()

    writer.close()
    print("✅ 리그 트레이닝 종료!")

if __name__ == "__main__":
    run_training()