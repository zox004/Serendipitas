import torch
import numpy as np
import rlcard
from rlcard.utils import print_card

from src.config import AlphaHoldemConfig as cfg
from src.env.wrappers import AlphaHoldemWrapper
from src.model.resnet import AlphaHoldemResNet
import os

def format_cards(cards):
    """['SA', 'H10'] -> [♠A, ♥T] 변환"""
    if not cards: return "[]"
    suits = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}
    ranks = {'T': '10', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A'}
    formatted = []
    for card in cards:
        s, r = card[0], card[1]
        formatted.append(f"{suits.get(s, s)}{ranks.get(r, r)}")
    return str(formatted)

def get_human_action(legal_actions):
    """
    사용자로부터 행동을 입력받는 함수
    [수정됨] RLCard 표준 Action ID에 맞춰 매핑 수정
    0: Fold
    1: Call/Check
    2: Raise Half
    3: Raise Pot
    4: All-in
    """
    action_map = {
        0: "Fold (포기)",
        1: "Call/Check (따라가기)",
        2: "Raise (Half-Pot)",
        3: "Raise (Pot)",
        4: "All-in"
    }
    
    print("\n[Your Turn] 가능한 행동:")
    valid_inputs = []
    
    # legal_actions에 있는 행동만 보여줌
    sorted_actions = sorted(legal_actions)
    for action_id in sorted_actions:
        action_name = action_map.get(action_id, f"Action {action_id}")
        print(f"  [{action_id}] {action_name}")
        valid_inputs.append(str(action_id))
        
    while True:
        user_input = input(">> 행동 번호를 입력하세요: ")
        if user_input in valid_inputs:
            return int(user_input)
        print("⚠️ 가능한 행동 번호만 입력해주세요.")

def run_game():
    print("\n" + "="*40)
    print(" 🃏 AlphaHoldem: Human vs AI Match 🃏")
    print("="*40)

    # 1. 환경 및 AI 설정
    raw_env = rlcard.make('no-limit-holdem', config={'seed': 42})
    env = AlphaHoldemWrapper(raw_env)
    
    # 2. 모델 로드
    agent = AlphaHoldemResNet().to(cfg.DEVICE)
    model_path = os.path.join(cfg.CHECKPOINT_DIR, "alpha_holdem_siamese.pth")
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        return

    print(f"🤖 AI 모델 로드 중... ({model_path})")
    try:
        agent.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
    except Exception as e:
        print(f"❌ 모델 로드 에러: {e}")
        return

    agent.eval() 
    print("✅ 준비 완료! 게임을 시작합니다.\n")

    while True:
        state, player_id = env.reset()
        done = False
        
        human_seat = np.random.choice([0, 1])
        ai_seat = 1 - human_seat
        
        print("-" * 40)
        print(f"🎮 New Game! (Human: Player {human_seat}, AI: Player {ai_seat})")
        
        while not done:
            current_player = player_id
            
            # --- [Human Turn] ---
            if current_player == human_seat:
                raw_state = env.env.get_state(player_id)
                # 데이터 구조 호환성 처리
                info = raw_state['raw_obs'] if 'raw_obs' in raw_state else raw_state
                
                public_cards = info.get('public_cards', [])
                hand_cards = info.get('hand', [])
                pot = info.get('pot', 0)
                my_chips = info.get('my_chips', 0)
                all_chips = info.get('all_chips', [0, 0])
                opp_chips = all_chips[ai_seat]
                
                print(f"\n--- [My Turn] ---")
                print(f"💰 Pot: {pot} (Me: {my_chips} vs AI: {opp_chips})")
                print(f"🃏 Board: {format_cards(public_cards)}")
                print(f"✋ My Hand: {format_cards(hand_cards)}")
                
                if isinstance(raw_state['legal_actions'], dict):
                    legal_actions = list(raw_state['legal_actions'].keys())
                else:
                    legal_actions = raw_state['legal_actions']
                
                action = get_human_action(legal_actions)
                
            # --- [AI Turn] ---
            else:
                print(f"\n🤖 AI Thinking...", end=" ")
                action, _ = agent.get_action(state, deterministic=True)
                
                # AI 행동 해석 (수정됨)
                action_names = ["Fold", "Call/Check", "Raise Half", "Raise Pot", "All-in"]
                action_str = action_names[action] if action < len(action_names) else str(action)
                print(f"-> AI chose: '{action_str}', '{action}'")

            # 환경 진행
            state, player_id = env.step(action)
            if state is None:
                done = True

        # --- [게임 종료] ---
        payoffs = env.env.get_payoffs()
        human_reward = payoffs[human_seat]
        
        final_state_ai = env.env.get_state(ai_seat)
        final_state_human = env.env.get_state(human_seat)
        
        ai_info = final_state_ai.get('raw_obs', final_state_ai)
        human_info = final_state_human.get('raw_obs', final_state_human)
        
        print("\n🏁 Game Over")
        print(f"🤖 AI Hand: {format_cards(ai_info.get('hand', []))}")
        print(f"🧑 My Hand: {format_cards(human_info.get('hand', []))}")
        print(f"🃏 Board : {format_cards(ai_info.get('public_cards', []))}")
        
        if human_reward > 0:
            print(f"\n🎉 You Win! (+{human_reward})")
        elif human_reward < 0:
            print(f"\n💀 You Lose... ({human_reward})")
        else:
            print(f"\n🤝 Draw!")

        if input("\n한 판 더 하시겠습니까? (Enter: Yes / q: Quit): ").lower() == 'q':
            break

if __name__ == "__main__":
    run_game()