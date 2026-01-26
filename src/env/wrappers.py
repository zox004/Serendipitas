# 파일명: src/env/wrappers.py
import numpy as np
import torch
from rlcard.envs import Env

class AlphaHoldemWrapper:
    """
    AlphaHoldem 전용 환경 래퍼
    - Observation: Dictionary -> PyTorch Tensor 변환
    - Action: Discrete(6) -> 실제 베팅 금액 변환
    """
    def __init__(self, env):
        self.env = env
        self.num_actions = 6  # Fold, Check/Call, Min-Raise, Half-Pot, Pot, All-in
        # 입력 차원: RLCard 기본 54 dim + 추가 정보(필요시 확장 가능)
        # 현재는 RLCard 기본 obs(54)를 그대로 사용
        self.state_shape = [54] 

    def reset(self):
        state, player_id = self.env.reset()
        return self._process_state(state), player_id

    def step(self, action_idx):
        """
        AI의 행동(0~5)을 실제 환경에 적용
        """
        raw_state = self.env.get_state(self.env.get_player_id())
        legal_actions = raw_state['legal_actions']
        
        # 1. AI의 선택(Index)을 실제 포커 액션(String/Int)으로 변환
        real_action = self._decode_action(action_idx, legal_actions, raw_state)
        
        # 2. 환경에 적용
        next_state, next_player_id = self.env.step(real_action)
        
        # 3. 다음 상태 반환 (게임 종료 시 next_state 처리 주의)
        if self.env.is_over():
            return None, next_player_id
            
        return self._process_state(next_state), next_player_id

    def _process_state(self, state):
        """
        RLCard 상태 딕셔너리 -> PyTorch Tensor 변환
        """
        # 'obs'는 RLCard가 제공하는 카드+칩 정보 벡터 (크기 54)
        obs = state['obs']
        
        # numpy array -> torch tensor (배치 차원 추가: 1 x 54)
        tensor_obs = torch.from_numpy(obs).float().unsqueeze(0)
        return tensor_obs

    def _decode_action(self, action_idx, legal_actions, state):
        """
        Discrete Action(0~5)을 실제 베팅 금액으로 매핑
        """
        # action_idx:
        # 0: Fold
        # 1: Check / Call
        # 2: Min-Raise
        # 3: Half-Pot Raise
        # 4: Pot Raise
        # 5: All-in
        
        # RLCard의 legal_actions에는 허용된 액션 ID들이 들어있음
        # RLCard HUNL Action ID 매핑: 0:Call, 1:Raise, 2:Fold, 3:Check
        
        # 1. Fold
        if action_idx == 0:
            return 2 # RLCard Fold ID
            
        # 2. Check / Call
        if action_idx == 1:
            if 3 in legal_actions: return 3 # Check
            return 0 # Call

        # 3. Raise 계열 (Min, Half, Pot, All-in)
        # 레이즈가 불가능한 상황(이미 올인이거나 칩 부족)이면 Call로 대체
        if 1 not in legal_actions:
            return 0 # Call (Fallback)

        # 현재 팟 정보 가져오기 (RLCard raw_obs 활용 필요)
        # 간단한 구현을 위해 여기서는 팟 사이즈 추정이 필요하나, 
        # RLCard wrapper 내부 로직상 정확한 금액 계산을 위해서는 
        # env.game.state 등의 접근이 필요함.
        # Day 3 단계에서는 'Min-Raise'와 'All-in' 위주로 매핑하고 
        # Half/Pot은 Min-Raise로 임시 매핑 후 Day 4에서 고도화 추천.
        
        if action_idx == 2: # Min-Raise
            return 1 # RLCard는 raise 시 금액을 지정해야 함 (env.step(action, amount))
            # *주의: RLCard의 step은 (action_id) 혹은 (action_id, amount)를 받음
            # 이 Wrapper는 단순화를 위해 기본 Raise 동작을 수행하도록 설정
            
        if action_idx == 5: # All-in
            # All-in 로직은 별도 구현 필요, 여기서는 단순히 Raise로 처리
            return 1 

        # 기본 리턴 (안전장치)
        return 0