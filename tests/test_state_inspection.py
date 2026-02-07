# tests/test_state_inspection.py
"""env에서 나오는 state (card_tensor, hist_tensor) 직접 확인용 테스트"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import rlcard
from src.config import AlphaHoldemConfig as cfg
from src.env.wrappers import AlphaHoldemWrapper
from src.env.encoder import AlphaHoldemEncoder


def test_state_shape_and_values():
    """reset/step 후 state의 shape, dtype, 값 범위 출력"""
    raw_env = rlcard.make("no-limit-holdem", config={"seed": 42})
    env = AlphaHoldemWrapper(raw_env)
    encoder = AlphaHoldemEncoder()

    # 1) reset 후 state
    state, player_id = env.reset()
    card_tensor, hist_tensor = state

    print("\n=== After env.reset() ===")
    print(f"player_id: {player_id}")
    print(f"card_tensor shape: {card_tensor.shape}, dtype: {card_tensor.dtype}")
    print(f"hist_tensor shape: {hist_tensor.shape}, dtype: {hist_tensor.dtype}")
    print(f"card_tensor min/max: {card_tensor.min().item():.2f} / {card_tensor.max().item():.2f}")
    print(f"hist_tensor min/max: {hist_tensor.min().item():.2f} / {hist_tensor.max().item():.2f}")

    # 2) 카드 채널별 1 개수 (어디에 카드가 찍혀 있는지)
    for ch in range(card_tensor.shape[0]):
        n = (card_tensor[ch] > 0.5).sum().item()
        print(f"  card channel {ch} non-zero cells: {n}")

    # --- 실제 텐서 출력 (reset 직후) ---
    print("\n--- card_tensor (채널별 4x13, 행=무늬 열=숫자) ---")
    torch.set_printoptions(threshold=5000, edgeitems=4, linewidth=120)
    for ch in range(card_tensor.shape[0]):
        print(f"  [channel {ch}]\n{card_tensor[ch]}")
    print("\n--- hist_tensor (24, 4, 5) ---")
    print(hist_tensor)

    # 3) raw_obs 한 번 더 찍어보기 (encoder 입력)
    raw_state = env.env.get_state(player_id)
    raw_obs = raw_state.get("raw_obs", raw_state)
    print(f"\nraw_obs keys: {list(raw_obs.keys())}")
    print(f"  hand: {raw_obs.get('hand', [])}")
    print(f"  public_cards: {raw_obs.get('public_cards', [])}")
    print(f"  action_history (first 5): {raw_obs.get('action_history', [])[:5]}")

    # 4) step 한 번 진행 후 state
    legal = raw_state.get("legal_actions")
    if isinstance(legal, dict):
        legal = list(legal.keys())
    action = legal[0] if legal else 0
    next_state, next_pid = env.step(action)

    if next_state is not None:
        card_next, hist_next = next_state
        print("\n=== After env.step(action) ===")
        print(f"next_player_id: {next_pid}")
        print(f"card_tensor shape: {card_next.shape}")
        print(f"hist_tensor shape: {hist_next.shape}")
        print(f"card_tensor min/max: {card_next.min().item():.2f} / {card_next.max().item():.2f}")
        print(f"hist_tensor min/max: {hist_next.min().item():.2f} / {hist_next.max().item():.2f}")
        print("\n--- next card_tensor (채널별) ---")
        for ch in range(card_next.shape[0]):
            print(f"  [ch {ch}]\n{card_next[ch]}")
        print("\n--- next hist_tensor ---")
        print(hist_next)
    else:
        print("\n=== After env.step(action): game ended (next_state is None) ===")

    # 5) encoder에 직접 raw_state 넣어서 동일한지 확인
    again_card, again_hist = encoder.encode(raw_state)
    print("\n=== encoder.encode(raw_state) direct call ===")
    print(f"card shape: {again_card.shape}, equal to state[0]: {torch.allclose(card_tensor, again_card)}")
    print(f"hist shape: {again_hist.shape}, equal to state[1]: {torch.allclose(hist_tensor, again_hist)}")


def _summarize_state(card_tensor, hist_tensor):
    """state 요약 문자열 (한 줄)"""
    card_counts = [(card_tensor[ch] > 0.5).sum().item() for ch in range(card_tensor.shape[0])]
    hist_ones = (hist_tensor > 0.5).sum().item()
    return f"card_ch_counts={card_counts} | hist_ones={hist_ones}"


def test_state_evolution_full_game():
    """한 게임을 끝까지 진행하며 매 턴 state 변화 출력 (랜덤 행동)"""
    import random
    torch.set_printoptions(threshold=5000, edgeitems=4, linewidth=120)

    raw_env = rlcard.make("no-limit-holdem", config={"seed": 42})
    env = AlphaHoldemWrapper(raw_env)

    state, player_id = env.reset()
    step_idx = 0
    history = []  # (step_idx, player_id, action, state_tuple) 기록

    print("\n" + "=" * 60)
    print("  게임 시작 ~ 끝까지 state 변화")
    print("=" * 60)

    while True:
        raw_state = env.env.get_state(player_id)
        legal = raw_state.get("legal_actions")
        if isinstance(legal, dict):
            legal = list(legal.keys())
        action = 1

        card_tensor, hist_tensor = state
        summary = _summarize_state(card_tensor, hist_tensor)
        history.append((step_idx, player_id, action, state))

        print(f"\n--- Step {step_idx} (Player {player_id}, action={action}) ---")
        print(f"  요약: {summary}")
        print(f"  raw_obs: hand={raw_state.get('raw_obs', {}).get('hand', [])}, "
              f"public_cards={raw_state.get('raw_obs', {}).get('public_cards', [])}")

        next_state, next_pid = env.step(action)
        step_idx += 1

        if next_state is None:
            payoffs = env.env.get_payoffs()
            print("\n" + "=" * 60)
            print(f"  게임 종료 (총 {step_idx} 스텝) | payoffs = {payoffs}")
            print("=" * 60)
            break

        state = next_state
        player_id = next_pid

    # 처음 2스텝, 마지막 2스텝의 실제 텐서 출력
    print("\n" + "=" * 60)
    print("  [처음 2스텝] 실제 텐서")
    print("=" * 60)
    for step_idx, pid, act, (card, hist) in history[:2]:
        print(f"\n>>> Step {step_idx} (Player {pid}, action={act})")
        print("  card_tensor (채널별):")
        for ch in range(card.shape[0]):
            print(f"    [ch{ch}]\n{card[ch]}")
        print("  hist_tensor:")
        print(hist)

    if len(history) > 2:
        print("\n" + "=" * 60)
        print("  [마지막 2스텝] 실제 텐서")
        print("=" * 60)
        for step_idx, pid, act, (card, hist) in history[-2:]:
            print(f"\n>>> Step {step_idx} (Player {pid}, action={act})")
            print("  card_tensor (채널별):")
            for ch in range(card.shape[0]):
                print(f"    [ch{ch}]\n{card[ch]}")
            print("  hist_tensor:")
            print(hist)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        test_state_evolution_full_game()
    else:
        test_state_shape_and_values()