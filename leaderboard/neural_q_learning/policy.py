from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

STATE_SIZE = 38
ACTION_VALUES = np.arange(0, 101, 10, dtype=np.int64)

class MultiHeadQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(STATE_SIZE, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.head = nn.Linear(128, 33)

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features).view(-1, 3, 11)


def _state(observation):
    inventory = np.asarray(observation["inventory"], dtype=np.float32) / 1000.0
    pipeline = np.asarray(observation["arrival_pipeline"], dtype=np.float32).reshape(-1) / 1000.0
    demand = np.asarray(observation["demand_history"], dtype=np.float32).reshape(-1) / 200.0
    day = np.asarray(observation["day"], dtype=np.float32).reshape(-1) / 50.0
    capacity = np.asarray(observation["capacity_utilisation"], dtype=np.float32).reshape(-1)
    return np.concatenate([inventory, pipeline, demand, day, capacity])

MODEL = MultiHeadQNetwork()
MODEL.load_state_dict(torch.load(Path(__file__).resolve().parent / "neural_q_model.pt", map_location="cpu"))
MODEL.eval()

def run_policy(observation):
    state = torch.as_tensor(_state(observation), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = MODEL(state)[0]
        indices = q_values.argmax(dim=1).cpu().numpy()

    return [int(ACTION_VALUES[int(index)]) for index in indices]
