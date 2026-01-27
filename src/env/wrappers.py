import numpy as np
import torch

class AlphaHoldemWrapper:
    """
    AlphaHoldem 전용 환경 래퍼 (RLCard Default 5-Actions 적용)
    - Action Space: 5 (Fold, Check/Call, Half-Pot, Pot, All-in)
    - Observation Space: Tensor [1, 54]
    """
    def __init__(self, env):
        self.env = env
        # RLCard No-Limit Hold'em 기본 행동 개수 (5개)
        self.num_actions = 5  
        # 입력 데이터 차원 (기본 54)
        self.state_shape = [54] 

    def reset(self):
        state, player_id = self.env.reset()
        return self._process_state(state), player_id

    def step(self, action_idx):
        """
        AI의 행동(0~4)을 받아서 환경에 적용
        """
        # 현재 상태에서 가능한 행동 목록 조회 (예: [0, 1, 4])
        raw_state = self.env.get_state(self.env.get_player_id())
        legal_actions = list(raw_state['legal_actions'].keys()) 
        # *주의: RLCard 버전에 따라 legal_actions가 dict인 경우 keys() 추출 필요
        # 만약 list라면 그대로 사용. 안전을 위해 list로 변환.
        if isinstance(raw_state['legal_actions'], dict):
             legal_actions = list(raw_state['legal_actions'].keys())
        else:
             legal_actions = raw_state['legal_actions']

        # 1. 행동 유효성 검사 및 보정
        real_action = self._decode_action(action_idx, legal_actions)
        
        # 2. 환경에 적용
        next_state, next_player_id = self.env.step(real_action)
        
        # 3. 게임 종료 처리
        if self.env.is_over():
            return None, next_player_id
            
        return self._process_state(next_state), next_player_id

    def _process_state(self, state):
        """
        Numpy Observation -> PyTorch Tensor 변환
        """
        obs = state['obs']
        # 차원 추가: [54] -> [1, 54] (Batch Dimension)
        tensor_obs = torch.from_numpy(obs).float().unsqueeze(0)
        return tensor_obs

    def _decode_action(self, action_idx, legal_actions):
        """
        AI가 선택한 행동이 유효한지 확인하고, 불가능하면 안전한 행동으로 대체
        RLCard Action ID:
          0: Fold
          1: Check/Call
          2: Raise Half-Pot
          3: Raise Pot
          4: All-in
        """
        # 1. AI가 고른 행동이 규칙상 가능하면 그대로 진행
        if action_idx in legal_actions:
            return action_idx
            
        # 2. 불가능한 행동(예: 돈이 없는데 레이즈)이라면?
        # -> Check/Call(1) 시도
        if 1 in legal_actions:
            return 1
            
        # 3. 그것도 안 되면(매우 드문 경우) -> Fold(0)
        return 0