# 파일명: src/env/encoder.py 수정
import numpy as np
import torch
from src.config import AlphaHoldemConfig # Config 임포트

class AlphaHoldemEncoder:
    def __init__(self):
        # Config에서 값 가져오기
        self.state_shape = [AlphaHoldemConfig.INPUT_DIM]
        self.max_chips = float(AlphaHoldemConfig.MAX_CHIPS)
        
        self.suits = ['S', 'H', 'D', 'C']
        self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

    def encode(self, raw_state):
        raw_obs = raw_state['raw_obs']
        
        my_hand_vec = self._cards_to_array(raw_obs['hand'])
        board_vec = self._cards_to_array(raw_obs['public_cards'])
        
        # Config의 max_chips 사용
        my_stack = raw_obs['my_chips'] / self.max_chips
        
        opp_chips = raw_obs['all_chips'][1] if raw_obs['all_chips'][0] == raw_obs['my_chips'] else raw_obs['all_chips'][0]
        opp_stack = opp_chips / self.max_chips
        
        pot = raw_obs['pot'] / self.max_chips
        
        chips_vec = np.array([my_stack, opp_stack, pot])
        feature = np.concatenate([my_hand_vec, board_vec, chips_vec])
        
        return torch.from_numpy(feature).float().unsqueeze(0)

    def _cards_to_array(self, card_list):
        matrix = np.zeros(52)
        for card in card_list:
            idx = self._get_card_id(card)
            if idx is not None:
                matrix[idx] = 1.0
        return matrix

    def _get_card_id(self, card_str):
        if not card_str: return None
        suit = card_str[0]
        rank = card_str[1]
        try:
            suit_idx = self.suits.index(suit)
            rank_idx = self.ranks.index(rank)
            return suit_idx * 13 + rank_idx
        except ValueError:
            return None