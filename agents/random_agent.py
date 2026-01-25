import numpy as np

class TexasRandomAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.use_raw = False

    def step(self, state):
        """딕셔너리 형태의 legal_actions에서 키값(액션 ID)만 추출하여 선택"""
        # 만약 dict 형태라면 keys를 리스트로 변환, 리스트라면 그대로 사용
        legal_actions_list = list(state['legal_actions'].keys())
        return np.random.choice(legal_actions_list)

    def eval_step(self, state):
        return self.step(state), {}