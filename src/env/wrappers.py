# 파일명: src/env/wrappers.py 수정
import numpy as np
import torch
from src.env.encoder import AlphaHoldemEncoder  # 인코더 임포트

class AlphaHoldemWrapper:
    def __init__(self, env):
        self.env = env
        self.num_actions = 5
        
        # 인코더 장착
        self.encoder = AlphaHoldemEncoder()
        self.state_shape = self.encoder.state_shape # [107]

    def reset(self):
        state, player_id = self.env.reset()
        # 인코더를 통해 변환
        return self._process_state(state), player_id

    def step(self, action_idx):
        raw_state = self.env.get_state(self.env.get_player_id())
        
        if isinstance(raw_state['legal_actions'], dict):
             legal_actions = list(raw_state['legal_actions'].keys())
        else:
             legal_actions = raw_state['legal_actions']

        real_action = self._decode_action(action_idx, legal_actions)
        next_state, next_player_id = self.env.step(real_action)
        
        if self.env.is_over():
            return None, next_player_id
            
        return self._process_state(next_state), next_player_id

    def _process_state(self, state):
        # 기존: state['obs'] 사용
        # 변경: RLCard 내부의 raw_obs를 가져와서 인코더에 넘김
        
        # 주의: step()에서 받은 state는 이미 가공된 것일 수 있음.
        # 확실한 Raw Data를 얻기 위해 env에서 직접 조회
        player_id = self.env.get_player_id()
        raw_state = self.env.get_state(player_id)
        
        return self.encoder.encode(raw_state)

    def _decode_action(self, action_idx, legal_actions):
        if action_idx in legal_actions: return action_idx
        if 1 in legal_actions: return 1
        return 0