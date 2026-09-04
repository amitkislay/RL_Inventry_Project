from pathlib import Path
import numpy as np
import torch
from torch import nn
MODEL_KIND='REINFORCE_Baseline'
class Model(nn.Module):
    def __init__(self,hidden_dim=256):
        super().__init__(); self.backbone=nn.Sequential(nn.Linear(38,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,hidden_dim),nn.Tanh())
        self.actor_heads=nn.ModuleList([nn.Linear(hidden_dim,11) for _ in range(3)]); self.critic=nn.Linear(hidden_dim,1)
    def forward(self,x):
        f=self.backbone(x); return torch.stack([h(f) for h in self.actor_heads],dim=1),self.critic(f).squeeze(-1)
def _state(o):
    return np.concatenate([np.asarray(o['inventory'],dtype=np.float32)/1000.0,np.asarray(o['arrival_pipeline'],dtype=np.float32).reshape(-1)/400.0,np.asarray(o['demand_history'],dtype=np.float32).reshape(-1)/100.0,np.asarray(o['day'],dtype=np.float32).reshape(-1)/50.0,np.asarray(o['capacity_utilisation'],dtype=np.float32).reshape(-1)]).astype(np.float32)
_payload=torch.load(Path(__file__).resolve().parent/'model.pt',map_location='cpu',weights_only=True)
_model=Model(int(_payload.get('hidden_dim',256))); _model.load_state_dict(_payload['state_dict']); _model.eval()
def run_policy(observation):
    with torch.no_grad(): logits,_=_model(torch.as_tensor(_state(observation)).unsqueeze(0))
    return (logits.argmax(dim=2).squeeze(0).numpy().astype(int)*10).tolist()
