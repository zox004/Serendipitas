# 파일명: src/agent/ppo_agent.py
import torch
import torch.optim as optim
from src.config import AlphaHoldemConfig as cfg
from src.model.resnet import AlphaHoldemResNet

class PPOAgent:
    def __init__(self, input_dim, action_dim, lr, K_epochs, eps_clip):
        # input_dim은 더 이상 사용되지 않음 (Config에서 로드)
        self.gamma = cfg.GAMMA
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.data = []
        
        self.policy = AlphaHoldemResNet().to(cfg.DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = AlphaHoldemResNet().to(cfg.DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.mse_loss = torch.nn.MSELoss()
        
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, ns_lst, d_lst, prob_lst = [], [], [], [], [], []
        
        for transition in self.data:
            s, a, r, ns, d, prob = transition
            s_lst.append(s)    
            ns_lst.append(ns)  
            
            a_lst.append([a])
            r_lst.append([r])
            d_lst.append([d])
            prob_lst.append([prob])

        # --- [수정] State Tuple Batching ---
        # torch.cat -> torch.stack (새로운 배치 차원 생성)
        
        c_list = [item[0] for item in s_lst]
        h_list = [item[1] for item in s_lst]
        
        # [변경] cat 대신 stack 사용
        s_card = torch.stack(c_list, dim=0).to(cfg.DEVICE) 
        s_hist = torch.stack(h_list, dim=0).to(cfg.DEVICE)
        
        # --- [수정] Next State Tuple Batching ---
        nc_list = [item[0] for item in ns_lst]
        nh_list = [item[1] for item in ns_lst]
        
        # [변경] cat 대신 stack 사용
        ns_card = torch.stack(nc_list, dim=0).to(cfg.DEVICE)
        ns_hist = torch.stack(nh_list, dim=0).to(cfg.DEVICE)

        # 나머지는 동일 (Scalar 값들은 tensor 변환이므로 그대로 유지)
        a = torch.tensor(a_lst).to(cfg.DEVICE)
        r = torch.tensor(r_lst, dtype=torch.float).to(cfg.DEVICE)
        d = torch.tensor(d_lst, dtype=torch.float).to(cfg.DEVICE)
        prob = torch.tensor(prob_lst, dtype=torch.float).to(cfg.DEVICE)
        
        return (s_card, s_hist), a.long(), r, (ns_card, ns_hist), d, prob

    def train_net(self):
        s, a, r, ns, d, prob_old = self.make_batch()
        loss_sum = 0
        
        for _ in range(self.K_epochs):
            # Forward에 튜플 s 전달
            pi, v = self.policy(s)
            
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