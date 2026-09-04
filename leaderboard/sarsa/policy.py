from pathlib import Path
import pickle
import numpy as np

TABULAR_ACTIONS = np.asarray([
    [0, 0, 0], [20, 20, 20], [30, 20, 30], [30, 30, 30],
    [40, 30, 40], [40, 40, 40], [50, 40, 50], [50, 50, 50],
    [60, 50, 60], [60, 60, 60], [70, 60, 70], [80, 70, 80],
    [100, 100, 100],
], dtype=np.int64)

MODEL_FILE = Path(__file__).resolve().parent / "sarsa_table.pkl"
with MODEL_FILE.open("rb") as model_file:
    SARSA_TABLE = pickle.load(model_file)

def _recent_mean_demand(observation):
    history = np.asarray(observation["demand_history"], dtype=np.float64)
    observed_rows = history[np.any(history > 0, axis=1)]
    if len(observed_rows) == 0:
        return np.asarray([33.0, 25.0, 31.5], dtype=np.float64)
    return observed_rows.mean(axis=0)

def _discretize_ratio(value):
    if value < 0.75: return 0
    if value < 1.25: return 1
    if value < 1.75: return 2
    if value < 2.50: return 3
    return 4

def _tabular_state(observation):
    inventory = np.asarray(observation["inventory"], dtype=np.float64)
    pipeline = np.asarray(observation["arrival_pipeline"], dtype=np.float64)
    demand_mean = _recent_mean_demand(observation)
    inventory_position = inventory + pipeline.sum(axis=1)
    reference_lead_times = np.asarray([3.0, 2.0, 1.0], dtype=np.float64)
    expected = np.maximum(demand_mean * reference_lead_times, 1.0)
    coverage_bins = tuple(
        _discretize_ratio(value) for value in inventory_position / expected
    )
    capacity = float(np.asarray(observation["capacity_utilisation"]).reshape(-1)[0])
    if capacity < 0.40: capacity_bin = 0
    elif capacity < 0.70: capacity_bin = 1
    elif capacity < 0.90: capacity_bin = 2
    else: capacity_bin = 3
    day = int(np.asarray(observation["day"]).reshape(-1)[0])
    return coverage_bins + (capacity_bin, min(day // 10, 4))

def _fallback(observation):
    inventory = np.asarray(observation["inventory"], dtype=np.float64)
    pipeline = np.asarray(observation["arrival_pipeline"], dtype=np.float64)
    position = inventory + pipeline.sum(axis=1)
    required = np.maximum(np.asarray([110.0, 100.0, 120.0]) - position, 0.0)
    quantities = np.clip(np.ceil(required / 10.0) * 10.0, 0.0, 100.0)
    return quantities.astype(int).tolist()

def run_policy(observation):
    state = _tabular_state(observation)
    if state not in SARSA_TABLE:
        return _fallback(observation)
    action_index = int(np.argmax(np.asarray(SARSA_TABLE[state], dtype=np.float64)))
    return TABULAR_ACTIONS[action_index].astype(int).tolist()
