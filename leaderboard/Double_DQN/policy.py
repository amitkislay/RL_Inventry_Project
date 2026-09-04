from pathlib import Path
import torch
from torch import nn
import numpy as np

STATE_DIM=38
NUM_PRODUCTS=3
ACTIONS_PER_PRODUCT=11
MODEL_KIND='dqn'

class Model(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        activation = nn.Tanh if MODEL_KIND == "ppo" else nn.ReLU
        layers=[nn.Linear(STATE_DIM, hidden_dim)]
        if MODEL_KIND != "ppo": layers.append(nn.LayerNorm(hidden_dim))
        layers.extend([activation(), nn.Linear(hidden_dim, hidden_dim), activation()])
        self.backbone=nn.Sequential(*layers)
        self.heads=nn.ModuleList([nn.Linear(hidden_dim,ACTIONS_PER_PRODUCT) for _ in range(NUM_PRODUCTS)])
        if MODEL_KIND == "ppo": self.critic=nn.Linear(hidden_dim,1)
    def forward(self,x):
        f=self.backbone(x)
        return torch.stack([h(f) for h in self.heads],dim=1)

def _state(o):
    x=np.concatenate([
        np.asarray(o["inventory"],dtype=np.float32)/1000.0,
        np.asarray(o["arrival_pipeline"],dtype=np.float32).reshape(-1)/400.0,
        np.asarray(o["demand_history"],dtype=np.float32).reshape(-1)/100.0,
        np.asarray(o["day"],dtype=np.float32).reshape(-1)/50.0,
        np.asarray(o["capacity_utilisation"],dtype=np.float32).reshape(-1),
    ]).astype(np.float32)
    return x

_payload=torch.load(Path(__file__).resolve().parent/"model.pt",map_location="cpu",weights_only=True)
_model=Model(hidden_dim=int(_payload.get("hidden_dim",256)))
_state_dict=_payload["state_dict"]
if MODEL_KIND == "ppo":
    _state_dict={k.replace("actor_heads","heads"):v for k,v in _state_dict.items()}
_model.load_state_dict(_state_dict)
_model.eval()

def run_policy(observation):
    x=torch.as_tensor(_state(observation)).unsqueeze(0)
    with torch.no_grad():
        action=_model(x).argmax(dim=2).squeeze(0).numpy()
    return (action.astype(int)*10).tolist()
