# 파일명: src/env/wrappers.py
import numpy as np
import torch
from src.env.encoder import AlphaHoldemEncoder

# RLCard no-limit hold'em Stage enum 값 → hist 라운드 인덱스 (0=preflop, 1=flop, 2=turn, 3=river)
def _stage_to_round_idx(stage):
    if stage is None:
        return 0
    try:
        v = stage.value
    except AttributeError:
        try:
            v = int(stage)
        except (TypeError, ValueError):
            return 0
    return min(3, max(0, v))

class AlphaHoldemWrapper:
    def __init__(self, env):
        self.env = env
        self.num_actions = 5
        self.encoder = AlphaHoldemEncoder()
        # (player_id, action, round_idx) 리스트 — 실제 라운드별 채널 할당용
        self._action_record_with_round = []

    def reset(self):
        self._action_record_with_round = []
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
        round_idx = _stage_to_round_idx(raw_state.get('raw_obs', {}).get('stage'))
        self._action_record_with_round.append((self.env.get_player_id(), real_action, round_idx))

        next_state, next_player_id = self.env.step(real_action)

        if self.env.is_over():
            return None, next_player_id

        # encoder가 실제 라운드별 채널에 넣을 수 있도록 action_record를 교체
        next_state['action_record'] = list(self._action_record_with_round)
        return self._process_state(next_state), next_player_id

    def _process_state(self, state):
        player_id = self.env.get_player_id()
        raw_state = self.env.get_state(player_id)
        raw_state['action_record'] = list(self._action_record_with_round)
        # returns (card_tensor, hist_tensor)
        return self.encoder.encode(raw_state)

    def _decode_action(self, action_idx, legal_actions):
        if action_idx in legal_actions: return action_idx
        if 1 in legal_actions: return 1
        return 0