# 파일명: src/env/encoder.py
import numpy as np
import torch
from src.config import AlphaHoldemConfig as cfg

class AlphaHoldemEncoder:
    def __init__(self):
        self.max_chips = float(cfg.MAX_CHIPS)
        self.suits = ['S', 'H', 'D', 'C']
        self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        
        # RLCard Action String -> Index 매핑
        # 주의: rlcard의 raw_obs['action_history']는 구체적인 액션 문자열을 줍니다.
        # 여기서는 단순화를 위해 액션의 종류를 추정하여 매핑합니다.
        self.action_map = {
            'fold': 0, 
            'check': 1, 'call': 1, 
            'raise_half_pot': 2, 
            'raise_pot': 3, 
            'all_in': 4
        }

    def encode(self, raw_state):
        """
        Returns:
            card_tensor: (6, 4, 13)
            hist_tensor: (24, 4, 5)
        """
        raw_obs = raw_state['raw_obs']
        
        # --- 1. Card Tensor (6, 4, 13) ---
        card_np = np.zeros((cfg.CARD_CHANNELS, cfg.CARD_HEIGHT, cfg.CARD_WIDTH), dtype=np.float32)
        
        my_hand = raw_obs['hand']
        public_cards = raw_obs.get('public_cards', [])
        
        self._fill_plane(card_np[0], my_hand)           # Ch 0: My Hand
        
        if len(public_cards) >= 3:
            self._fill_plane(card_np[1], public_cards[:3]) # Ch 1: Flop
        if len(public_cards) >= 4:
            self._fill_plane(card_np[2], [public_cards[3]]) # Ch 2: Turn
        if len(public_cards) >= 5:
            self._fill_plane(card_np[3], [public_cards[4]]) # Ch 3: River
            
        self._fill_plane(card_np[4], public_cards)      # Ch 4: All Public
        all_my_cards = list(my_hand) + list(public_cards)  # Ch 5: My hand + all public (full visible set)
        self._fill_plane(card_np[5], all_my_cards)
        
        card_tensor = torch.from_numpy(card_np).float()

        # --- 2. Betting History Tensor (24, 4, 5) ---
        # Ch: 24 = 4 rounds × 6 (한 라운드당 최대 베팅 횟수), H: Rounds (4), W: Actions (5)
        # 채널 0–5: 라운드0(preflop), 6–11: 라운드1(flop), 12–17: 라운드2(turn), 18–23: 라운드3(river)
        hist_np = np.zeros((cfg.HIST_CHANNELS, cfg.HIST_HEIGHT, cfg.HIST_WIDTH), dtype=np.float32)
        history = raw_state.get('action_record', [])
        max_slots_per_round = 6
        # 라운드별로 슬롯 개수 카운트 (실제 라운드에만 할당하기 위함)
        slot_count_per_round = [0, 0, 0, 0]

        for i, rec in enumerate(history):
            if i >= cfg.HIST_CHANNELS:
                break
            if len(rec) >= 3:
                player_id, action, round_idx = rec[0], rec[1], int(rec[2])
                round_idx = max(0, min(3, round_idx))
            else:
                player_id, action = rec[0], rec[1]
                round_idx = i // max_slots_per_round

            act_idx = int(action.value) if hasattr(action, 'value') else int(action)
            act_idx = max(0, min(4, act_idx))

            slot_in_round = slot_count_per_round[round_idx]
            if slot_in_round >= max_slots_per_round:
                continue
            slot_count_per_round[round_idx] += 1
            ch_idx = round_idx * max_slots_per_round + slot_in_round
            hist_np[ch_idx, round_idx, act_idx] = 1.0

        hist_tensor = torch.from_numpy(hist_np).float()
        
        return card_tensor, hist_tensor

    def _parse_action(self, action_str):
        # rlcard action string to 0~4 index
        action_str = action_str.lower()
        if 'fold' in action_str: return 0
        if 'check' in action_str or 'call' in action_str: return 1
        if 'half' in action_str: return 2
        if 'pot' in action_str: return 3
        if 'all' in action_str: return 4
        return 1 # Default

    def _fill_plane(self, plane, cards):
        for card in cards:
            r, c = self._get_card_idx(card)
            if r is not None:
                plane[r][c] = 1.0

    def _get_card_idx(self, card_str):
        if not card_str: return None, None
        suit = card_str[0]
        rank = card_str[1]
        try:
            r = self.suits.index(suit)
            c = self.ranks.index(rank)
            return r, c
        except ValueError:
            return None, None