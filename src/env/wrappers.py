# 파일명: src/env/wrappers.py
import numpy as np
import torch
from src.env.encoder import AlphaHoldemEncoder

class AlphaHoldemWrapper:
    def __init__(self, env):
        self.env = env
        self.num_actions = 5
        self.encoder = AlphaHoldemEncoder()

    def reset(self):
        state, player_id = self.env.reset()
        # Returns tuple (card, hist), player_id
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
        player_id = self.env.get_player_id()
        raw_state = self.env.get_state(player_id)
        # returns (card_tensor, hist_tensor)
        return self.encoder.encode(raw_state)

    def _decode_action(self, action_idx, legal_actions):
        if action_idx in legal_actions: return action_idx
        if 1 in legal_actions: return 1
        return 0