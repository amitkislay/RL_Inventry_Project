from pathlib import Path
import pickle
import numpy as np

TABULAR_ACTION_LEVELS = np.asarray([0, 20, 40, 60, 80, 100], dtype=np.int64)
TABULAR_ACTIONS = np.asarray([
    [product_1, product_2, product_3]
    for product_1 in TABULAR_ACTION_LEVELS
    for product_2 in TABULAR_ACTION_LEVELS
    for product_3 in TABULAR_ACTION_LEVELS
], dtype=np.int64)


def _find_model_file():
    policy_dir = Path(__file__).resolve().parent
    candidates = [
        policy_dir / "q_learning_table.pkl",
        policy_dir.parent / "models" / "q_learning_table.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("q_learning_table.pkl was not found")


with _find_model_file().open("rb") as model_file:
    Q_TABLE = pickle.load(model_file)


def _recent_mean_demand(observation):
    history = np.asarray(observation["demand_history"], dtype=np.float64)
    observed_rows = history[np.any(history > 0, axis=1)]
    if len(observed_rows) == 0:
        return np.asarray([33.0, 25.0, 31.5], dtype=np.float64)
    return observed_rows.mean(axis=0)


def _discretize_ratio(value):
    if value < 0.75:
        return 0
    if value < 1.25:
        return 1
    if value < 1.75:
        return 2
    if value < 2.50:
        return 3
    return 4


def _tabular_state(observation):
    inventory = np.asarray(observation["inventory"], dtype=np.float64)
    pipeline = np.asarray(observation["arrival_pipeline"], dtype=np.float64)
    demand_mean = _recent_mean_demand(observation)
    inventory_position = inventory + pipeline.sum(axis=1)
    lead_times = np.asarray([3.0, 2.0, 1.0], dtype=np.float64)
    expected_demand = np.maximum(demand_mean * lead_times, 1.0)
    coverage_bins = tuple(
        _discretize_ratio(value)
        for value in inventory_position / expected_demand
    )

    capacity = float(
        np.asarray(observation["capacity_utilisation"]).reshape(-1)[0]
    )
    if capacity < 0.40:
        capacity_bin = 0
    elif capacity < 0.70:
        capacity_bin = 1
    elif capacity < 0.90:
        capacity_bin = 2
    else:
        capacity_bin = 3

    day = int(np.asarray(observation["day"]).reshape(-1)[0])
    day_bin = min(day // 10, 4)
    return coverage_bins + (capacity_bin, day_bin)


def _fallback_policy(observation):
    inventory = np.asarray(observation["inventory"], dtype=np.float64)
    pipeline = np.asarray(observation["arrival_pipeline"], dtype=np.float64)
    inventory_position = inventory + pipeline.sum(axis=1)
    targets = np.asarray([110.0, 100.0, 120.0], dtype=np.float64)
    required = np.maximum(targets - inventory_position, 0.0)
    quantities = np.ceil(required / 10.0) * 10.0
    return np.clip(quantities, 0.0, 100.0).astype(int).tolist()


def run_policy(observation):
    """Return deterministic learned order quantities for three products."""
    state = _tabular_state(observation)
    if state not in Q_TABLE:
        return _fallback_policy(observation)
    q_values = np.asarray(Q_TABLE[state], dtype=np.float64)
    best_action_index = int(np.argmax(q_values))
    return TABULAR_ACTIONS[best_action_index].astype(int).tolist()
