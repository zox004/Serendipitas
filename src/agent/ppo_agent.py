# src/agent/ppo_agent.py 수정

import torch
import torch.optim as optim
from src.config import AlphaHoldemConfig as cfg
# [변경] 기존 network 대신 resnet 임포트
from src.model.resnet import AlphaHoldemResNet 

class PPOAgent:
    def __init__(self, input_dim, action_dim, lr, K_epochs, eps_clip):
        self.gamma = cfg.GAMMA
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.data = []
        
        # [변경] ResNet으로 객체 생성 (인자 필요 없음, Config에서 가져옴)
        self.policy = AlphaHoldemResNet().to(cfg.DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = AlphaHoldemResNet().to(cfg.DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.mse_loss = torch.nn.MSELoss()
    
    # ... (나머지 메서드: put_data, make_batch, train_net 등은 수정 불필요) ...
    # 기존 코드 그대로 두시면 됩니다.
    
    # (참고) 혹시 train_net 등 메서드 구현이 없으실까봐 
    # put_data, make_batch, train_net 메서드는 기존 Day 5~6 코드 유지입니다.
    
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, ns_lst, d_lst, prob_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, ns, d, prob = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            ns_lst.append(ns)
            d_lst.append([d])
            prob_lst.append([prob])
            
        # [수정 핵심] 
        # s와 ns는 이미 Tensor 객체들이 담긴 리스트입니다.
        # 따라서 torch.cat()을 사용해 차원을 이어 붙여야 합니다. (Batch 생성)
        s = torch.cat(s_lst, dim=0).to(cfg.DEVICE)
        ns = torch.cat(ns_lst, dim=0).to(cfg.DEVICE)
        
        # a, r, d, prob는 일반 숫자(Scalar)들이 담긴 리스트입니다.
        # 따라서 torch.tensor()를 사용해 Tensor로 변환합니다.
        a = torch.tensor(a_lst).to(cfg.DEVICE)
        r = torch.tensor(r_lst, dtype=torch.float).to(cfg.DEVICE)
        d = torch.tensor(d_lst, dtype=torch.float).to(cfg.DEVICE)
        prob = torch.tensor(prob_lst, dtype=torch.float).to(cfg.DEVICE)
        
        return s, a.long(), r, ns, d, prob

    def train_net(self):
        s, a, r, ns, d, prob_old = self.make_batch()
        loss_sum = 0
        
        for _ in range(self.K_epochs):
            # ResNet Forward
            pi, v = self.policy(s)
            
            # --- PPO 알고리즘 (기존 동일) ---
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_old))
            
            advantage = r - v.detach()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(v, r)
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            loss_sum += loss.mean().item()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.data = []
        return loss_sum / self.K_epochs