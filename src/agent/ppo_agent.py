import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.model.network import AlphaHoldemNetwork

class PPOAgent:
    def __init__(self, input_dim=54, action_dim=6, lr=0.0003, gamma=0.99, K_epochs=3, eps_clip=0.2):
        self.gamma = gamma          # 할인율 (미래 보상 중요도)
        self.eps_clip = eps_clip    # 급발진 방지 규제 (20%)
        self.K_epochs = K_epochs    # 데이터 재사용 횟수
        
        # 데이터 저장소 (Buffer)
        self.data = []
        
        # 신경망 및 최적화 도구 (Day 4의 그 뇌!)
        self.policy = AlphaHoldemNetwork(input_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 손실 함수 (MSE for Critic)
        self.mse_loss = nn.MSELoss()

    def put_data(self, transition):
        """
        게임을 하면서 모은 데이터를 저장
        (state, action, reward, next_state, done, prob_a)
        """
        self.data.append(transition)

    def make_batch(self):
        """
        저장된 데이터를 텐서로 변환
        """
        s_lst, a_lst, r_lst, ns_lst, done_lst, prob_a_lst = [], [], [], [], [], []
        
        for transition in self.data:
            s, a, r, ns, done, prob_a = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            ns_lst.append(ns)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            prob_a_lst.append([prob_a])
            
        # 리스트 -> 텐서 변환 (Batch 처리)
        # s_lst는 이미 텐서 리스트일 수 있으므로 cat 또는 stack 사용
        s = torch.cat(s_lst) 
        a = torch.tensor(a_lst)
        r = torch.tensor(r_lst, dtype=torch.float)
        ns = torch.cat(ns_lst)
        done = torch.tensor(done_lst, dtype=torch.float)
        prob_a = torch.tensor(prob_a_lst, dtype=torch.float)
        
        self.data = [] # 버퍼 비우기
        return s, a, r, ns, done, prob_a

    def train_net(self):
        """
        PPO 학습의 핵심 로직
        """
        s, a, r, ns, done, prob_a = self.make_batch()

        # 1. 반복 학습 (K_epochs 만큼 데이터를 우려먹음)
        for _ in range(self.K_epochs):
            # 현재 정책으로 다시 계산
            logits, curr_val = self.policy(s)
            _, next_val = self.policy(ns) # 다음 상태 가치
            
            # 2. TD Target 계산 (정답지 만들기)
            # Bellman 방정식: 보상 + 할인된 미래 가치
            td_target = r + self.gamma * next_val * done
            
            # Advantage 계산 (얼마나 이득인가?)
            delta = td_target - curr_val
            advantage = delta.detach() # 그래디언트 흐름 끊기
            
            # 3. Ratio 계산 (새 정책 / 옛날 정책)
            # a(행동)에 해당하는 확률만 뽑아냄
            probs = torch.softmax(logits, dim=-1)
            prob_a_curr = probs.gather(1, a)
            ratio = torch.exp(torch.log(prob_a_curr) - torch.log(prob_a)) # exp(ln(a)-ln(b)) = a/b

            # 4. PPO Loss (Clipped Surrogate Objective)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            
            # Actor Loss: Advantage를 최대화하되, 급격한 변화는 무시(min)
            actor_loss = -torch.min(surr1, surr2) 
            
            # Critic Loss: 예측값(curr_val)이 정답(td_target)에 가까워지도록
            critic_loss = self.mse_loss(curr_val, td_target.detach())
            
            # 최종 Loss 합산
            loss = actor_loss + 0.5 * critic_loss
            
            # 5. 역전파 및 가중치 업데이트
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        return loss.mean().item()