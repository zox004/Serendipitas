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
        card_np[5].fill(1.0)                            # Ch 5: Bias
        
        card_tensor = torch.from_numpy(card_np).float()

        # --- 2. Betting History Tensor (24, 4, 5) ---
        # Ch: Amount Bits (24), H: Rounds (4), W: Actions (5)
        hist_np = np.zeros((cfg.HIST_CHANNELS, cfg.HIST_HEIGHT, cfg.HIST_WIDTH), dtype=np.float32)
        
        # action_history 예시: [(0, 'check'), (1, 'raise_half_pot'), ...]
        # 라운드 구분: public_cards 개수 변화 등을 추적해야 하지만, 
        # rlcard raw_obs는 라운드 경계를 명시적으로 안 줌.
        # 간략화를 위해 action_history를 순회하며 라운드를 추정하거나,
        # 현재 라운드 정보(stage)를 활용해야 함.
        # 여기서는 RLCard history 구조 상 라운드 구분이 어려우므로,
        # 간단히 리스트 길이 기반이나 키워드로 매핑 (구현 편의상 모든 액션을 Preflop(0)부터 채운다고 가정하거나 개선 필요)
        # *실제 AlphaHoldem은 라운드별 히스토리를 정확히 추적함*
        
        # 이번 구현에서는 '현재 라운드'까지의 액션을 단순히 누적한다고 가정 (정교한 라운드 구분 로직은 환경 래퍼에서 관리 필요)
        # 하지만 스켈레톤 레벨이므로, 액션 종류별로 비트 마킹만 수행
        
        history = raw_obs.get('action_history', [])
        # 라운드 추적용 변수 (0: Pre, 1: Flop, 2: Turn, 3: River)
        # RLCard에서는 히스토리만 보고 라운드를 완벽히 알기 어려우므로, 
        # 대략적인 공용 카드 수로 추론하거나 해야 함. 
        # 여기서는 단순히 모두 0(Pre) 행에 넣지 않고, 가상의 로직으로 분배
        
        # (임시) 단순히 최근 액션들을 채우는 방식 or 전체 히스토리
        current_round = 0 
        # *개선: RLCard Env 수정 없이 정확히 하려면 복잡함. 일단 0번 행(Pre)에 다 넣는 것으로 처리 후 향후 고도화 추천*
        
        for player_id, action_str in history:
            # Action Key Parsing
            act_idx = self._parse_action(action_str)
            
            # Amount Encoding (칩 양을 24비트로)
            # 베팅 금액을 정확히 알 수 있다면 그 값을, 모르면 1로 설정
            # Check/Fold는 금액 0 -> 1.0 (발생 표시)
            # Raise/All-in은 칩 비율
            
            # (간소화) 해당 액션 칸의 모든 채널을 1로 채움 (One-hot like)
            # 실제로는 amount를 이진법으로 24개 채널에 분배해야 함
            hist_np[:, current_round, act_idx] = 1.0 
            
            # 라운드 넘김 로직 (가정)
            # 실제로는 별도 라운드 로그가 필요
        
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