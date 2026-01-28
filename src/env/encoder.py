import numpy as np
import torch

class AlphaHoldemEncoder:
    """
    Raw Observation -> Structured Normalized Tensor 변환기
    """
    def __init__(self):
        # 입력 차원: 내패(52) + 보드(52) + 칩정보(3) = 107
        self.state_shape = [107]
        # RLCard 표준 No-Limit Hold'em 칩 총량
        self.max_chips = 200.0 
        
        # 카드 변환을 위한 기준 정의 (RLCard 표준 순서)
        self.suits = ['S', 'H', 'D', 'C'] # Spades, Hearts, Diamonds, Clubs
        self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

    def encode(self, raw_state):
        """
        RLCard의 raw_state 딕셔너리를 받아서 107차원 텐서로 변환
        """
        raw_obs = raw_state['raw_obs']
        
        # 1. 내 카드 인코딩 (52 dim)
        my_hand_vec = self._cards_to_array(raw_obs['hand'])
        
        # 2. 보드 카드 인코딩 (52 dim)
        board_vec = self._cards_to_array(raw_obs['public_cards'])
        
        # 3. 칩 정보 정규화 (3 dim)
        my_stack = raw_obs['my_chips'] / self.max_chips
        
        # 상대방 칩 찾기 logic
        opp_chips = raw_obs['all_chips'][1] if raw_obs['all_chips'][0] == raw_obs['my_chips'] else raw_obs['all_chips'][0]
        opp_stack = opp_chips / self.max_chips
        
        pot = raw_obs['pot'] / self.max_chips
        
        chips_vec = np.array([my_stack, opp_stack, pot])
        
        # 4. 합치기
        feature = np.concatenate([my_hand_vec, board_vec, chips_vec])
        
        return torch.from_numpy(feature).float().unsqueeze(0)

    def _cards_to_array(self, card_list):
        """
        카드 문자열 리스트(['SA', 'HT'])를 One-hot 벡터(52)로 변환
        """
        matrix = np.zeros(52)
        for card in card_list:
            # 자체 구현한 함수 사용
            idx = self._get_card_id(card)
            if idx is not None:
                matrix[idx] = 1.0
        return matrix

    def _get_card_id(self, card_str):
        """
        Card String (e.g., 'SA') -> ID (0~51)
        Rule: Suit Index * 13 + Rank Index
        """
        if not card_str: return None
        
        suit = card_str[0]
        rank = card_str[1]
        
        try:
            suit_idx = self.suits.index(suit)
            rank_idx = self.ranks.index(rank)
            return suit_idx * 13 + rank_idx
        except ValueError:
            # 예외 처리: 알 수 없는 카드 형식이 들어오면 무시
            return None