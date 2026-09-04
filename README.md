# Reinforcement Learning for Industrial Inventory Control

## IITM Web M.Tech Course Project

This repository contains my implementation of the Reinforcement Learning Course Project for industrial inventory control. The objective is to learn daily replenishment decisions for a warehouse handling three products while minimizing holding, stockout, ordering, and discarding costs.

The project uses the official Gymnasium-compatible environment supplied with the course. The environment logic, observation structure, action mapping, transition rules, and cost function are not modified.

## Project Information

- **Student roll number:** `DA25M543`
- **Assigned variant:** `V024`
- **Configuration fingerprint:** `db750dbfac3be81f`
- **Project version:** `IITM-6002W-RL-Inventory-2026-v1`
- **Python version used:** Python 3.14
- **GitHub repository:** <https://github.com/amitkislay/RL_Inventry_Project>

The assigned configuration is generated programmatically from the roll number:

```python
from industrial_inventory_env import (
    generate_student_config,
    public_config_summary,
)

ROLL_NUMBER = "DA25M543"
student_config = generate_student_config(ROLL_NUMBER)
config_summary = public_config_summary(student_config)
```

The generated public configuration summary is:

```text
roll_number: DA25M543
variant_id: V024
config_fingerprint: db750dbfac3be81f
demand_multiplier_profile: [1.1, 1.0, 0.9]
initial_inventory_profile: [100, 110, 90]
lead_time_delay_profile: [0.02, 0.05, 0.08]
```

The variant was not selected manually. It is produced deterministically by the official configuration generator for roll number `DA25M543`.

---

## Problem Statement

The agent controls the daily replenishment quantities for three products over a 50-day episode. Each product has different demand behavior, storage volume, stockout cost, ordering cost, discarding cost, and supplier lead time.

The policy must balance four cost components:

1. **Holding cost:** Cost of carrying remaining inventory.
2. **Stockout cost:** Cost of unfulfilled product demand.
3. **Ordering cost:** Fixed cost for placing a non-zero order.
4. **Discarding cost:** Cost of discarding stock when warehouse capacity is exceeded.

The total daily cost is:

```text
daily_cost = holding_cost
           + stockout_cost
           + ordering_cost
           + discarding_cost
```

The official reward is:

```text
reward = -daily_cost / 100
```

Maximizing the official return is therefore equivalent to minimizing total inventory cost.

---

## Environment

The project uses the official `IndustrialInventoryEnv` environment with the Gymnasium API:

```python
observation, info = env.reset(seed=seed)

next_observation, reward, terminated, truncated, info = env.step(action)
```

An episode ends after 50 simulated days. The environment has no natural terminal state, so episode completion normally occurs when:

```python
truncated is True
```

### Observation Space

The policy receives a dictionary containing:

| Observation | Shape | Description |
|---|---:|---|
| `inventory` | `(3,)` | Current on-hand inventory for each product |
| `arrival_pipeline` | `(3, 4)` | Outstanding orders separated by expected arrival day |
| `demand_history` | `(7, 3)` | Demand observed during the previous seven days |
| `day` | `(1,)` | Current day index |
| `capacity_utilisation` | `(1,)` | Current warehouse capacity utilization |

The complete flattened observation contains 38 values.

### Action Space

The environment internally accepts one discrete action index for each product:

```text
[0, 1, 2, ..., 10]
```

Each index maps to an actual order quantity:

```text
[0, 10, 20, ..., 100]
```

For example:

```text
Internal action: [4, 2, 0]
Order quantity:  [40, 20, 0]
```

The submitted `run_policy(observation)` function must return actual quantities, not internal action indices.

---

## Demand Scenarios and Domain Randomization

Training and local validation cover the declared scenario families:

- Stationary demand
- Seasonal demand
- Gradual demand trend
- Temporary demand shock
- Mixed scenarios

Domain randomization is enabled during training. Episode-level randomization includes:

- Product-specific demand multipliers
- Initial stock levels
- Supplier delay probabilities
- Demand realizations
- Delay events

This reduces dependence on one favorable seed or one fixed demand pattern.

---

## Implemented Reinforcement Learning Techniques

The project notebook contains the following distinct techniques.

### 1. Tabular Q-Learning

An off-policy temporal-difference method using a discretized representation of inventory coverage, incoming stock, warehouse utilization, and episode progress.

Main implementation features:

- Epsilon-greedy exploration
- Deterministic greedy inference
- Q-table persistence using Pickle
- Multi-scenario domain-randomized training
- Local held-out validation
- Standalone policy export

Some experiments use a compact joint-action set. Later experiments investigate a larger structured action set to allow more independent product decisions.

### 2. Tabular SARSA

An on-policy temporal-difference implementation using the action selected by the current behavior policy in the update target.

The final notebook restores the original Tabular SARSA configuration because a later experimental optimization did not pass the required leaderboard checks:

```text
Episodes: 3000
Alpha: 0.10
Gamma: 0.99
Epsilon start: 1.00
Epsilon end: 0.05
Epsilon decay: 0.9985
Joint actions: 13
```

This restored version is reported as the original SARSA experiment, not as an optimized variant.

### 3. TD(lambda) with Eligibility Traces

An on-policy SARSA(lambda) implementation using replacing eligibility traces. The method propagates temporal-difference errors to recently visited state-action pairs.

Main implementation features:

- Replacing eligibility traces
- Configurable trace-decay parameter
- Epsilon-greedy behavior policy
- Multiple candidate configurations
- Separate tuning and validation seeds
- Deterministic final inference

### 4. Neural Network Based Q-Learning

A neural action-value approximation method using the full normalized observation. A shared encoder produces action values for the three product decisions.

Main implementation features:

- Full 38-value normalized state
- Neural Q-function approximation
- Experience replay
- Target network
- Gradient clipping
- Deterministic greedy inference

### 5. Neural Network Based SARSA

A neural on-policy action-value method. The improved experimental version uses Expected SARSA with replay memory and a target network for greater stability.

Main implementation features:

- Expected epsilon-greedy next-state value
- Replay buffer
- Target network
- Reward scaling
- AdamW optimizer
- Huber loss
- Gradient clipping
- Multiple training candidates
- Tuning-based candidate selection

### 6. REINFORCE with a Learned Baseline

A Monte Carlo policy-gradient implementation with a learned state-value baseline to reduce gradient variance.

Main implementation features:

- Categorical action distributions
- Discounted episode returns
- Learned value baseline
- Advantage normalization
- Entropy regularization
- Gradient clipping
- Deterministic final action selection

### 7. Advantage Actor-Critic, A2C

A synchronous actor-critic implementation in which the actor learns the ordering policy and the critic estimates state value.

Main implementation features:

- Shared neural feature encoder
- Actor and critic outputs
- Advantage-based policy updates
- Value-function loss
- Entropy regularization
- Deterministic inference

### 8. Asynchronous Advantage Actor-Critic, A3C

An actor-critic implementation based on worker environments and shared-policy updates. The notebook uses a Jupyter-safe worker-update structure to avoid autograd corruption caused by concurrent updates to one computation graph.

Main implementation features:

- Independent worker environments
- Local model snapshots
- Shared model updates
- Actor and critic losses
- Gradient clipping
- Deterministic final inference

### 9. Proximal Policy Optimization, PPO

An actor-critic policy-gradient method using a clipped probability-ratio objective.

Main implementation features:

- Generalized advantage estimation
- Clipped policy objective
- Multiple update epochs
- Value-function loss
- Entropy regularization
- Gradient clipping
- Deterministic action selection

### 10. Deep Q-Network, DQN

A replay-based neural Q-learning implementation using normalized observations and a target network.

Main implementation features:

- Experience replay
- Separate target network
- Epsilon-greedy exploration
- Huber loss
- Layer normalization
- Gradient clipping
- Model checkpoint saving
- Deterministic greedy inference

### 11. Double DQN

A Double DQN implementation that separates next-action selection from target-network evaluation to reduce Q-value overestimation.

Main implementation features:

- Online network for action selection
- Target network for value evaluation
- Experience replay
- Target-network synchronization
- Deterministic inference

---

## Notebook

The main project notebook is:

```text
RL_Inventory_Project_DA25M543.ipynb
```

The notebook includes:

- Environment and package checks
- Assigned-variant generation
- Observation and action inspection
- Deterministic demonstration policy
- Common evaluation functions
- Training pipelines
- Learning curves
- Model persistence
- Held-out validation
- Comparison of all completed policies
- Standalone `policy.py` generation
- Official interface validation
- Leaderboard ZIP creation
- Submission manifest generation

The final notebook does not require a separate `policies/` directory. Frozen policy files are generated directly inside technique-specific directories under `leaderboard/`.

---

## Repository Structure

A typical project directory after executing the notebook is:

```text
RL_Inventry_Project/
├── industrial_inventory_env/
├── leaderboard/
│   ├── q_learning/
│   │   ├── policy.py
│   │   └── q_learning_table.pkl
│   ├── sarsa/
│   │   ├── policy.py
│   │   └── sarsa_table.pkl
│   ├── dqn/
│   │   ├── policy.py
│   │   └── model.pt
│   ├── double_dqn/
│   │   ├── policy.py
│   │   └── model.pt
│   └── ...
├── models/
├── results/
├── checkpoints/
├── RL_Inventory_Project_DA25M543.ipynb
├── policy_validation_tests.py
├── submission_template.py
├── requirements.txt
└── README.md
```

The exact set of generated folders depends on which technique sections have been executed.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/amitkislay/RL_Inventry_Project.git
cd RL_Inventry_Project
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

```cmd
py -3.14 -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Confirm the environment

```bash
python --version
python -c "import torch, gymnasium, numpy, pandas; print('Environment ready')"
```

The project was developed using Python 3.14 with CPU-based PyTorch. Training times depend on the selected technique, episode count, and available hardware.

---

## Running the Project

Open the notebook in VS Code or Jupyter:

```text
RL_Inventory_Project_DA25M543.ipynb
```

Select the project virtual environment as the kernel, restart the kernel, and run the notebook sequentially.

Recommended execution order:

1. Setup and dependency checks
2. Assigned-variant generation
3. Environment inspection
4. Demonstration baseline
5. Common evaluation framework
6. Tabular techniques
7. Value-based neural techniques
8. Policy-gradient and actor-critic techniques
9. All-policy comparison
10. Model and policy export
11. Policy validation
12. Leaderboard ZIP creation
13. Final ZIP audit and submission manifest

Long-running training sections should be executed one technique at a time. Saving the notebook after each completed technique is recommended.

---

## Local Validation

Every final policy is evaluated using common validation seeds and multiple scenario families. The evaluation records:

- Total episode cost
- Holding cost
- Stockout cost
- Ordering cost
- Discarding cost
- Service level
- Stockout days
- Episode length
- Inference time

A standard validation call is:

```python
validation_results = evaluate_policy(
    policy=my_policy,
    student_config=student_config,
    seeds=VALIDATION_SEEDS,
    scenario_modes=VALIDATION_SCENARIOS,
    policy_name="Technique name",
)
```

The fixed local validation set contains 25 episodes when five seeds and five scenario modes are used.

The notebook generates:

```text
results/all_policy_comparison.csv
results/submission_manifest.csv
```

Additional training and validation CSV files are created for individual techniques.

---

## Policy Submission Interface

Every exported policy implements:

```python
def run_policy(observation):
    """Return order quantities for Products 1, 2, and 3."""
    return [q1, q2, q3]
```

The output must satisfy all of the following:

- Exactly three values
- Integer quantities
- Each quantity between 0 and 100
- Each quantity a multiple of 10
- Deterministic inference
- No environment reset or step calls
- No training during inference
- No internet access
- No loading files outside the submitted package

A policy is checked using:

```bash
python policy_validation_tests.py leaderboard/<technique>/policy.py
```

The validation script checks import behavior, deterministic inference, action validity, observation mutation, prohibited environment calls, and basic runtime behavior.

---

## Leaderboard Packaging

Each technique is packaged separately. A typical ZIP contains:

```text
policy.py
model_artifact
```

The files must appear at the ZIP root, not inside an additional enclosing directory.

Examples:

```text
DA25M543_V024_DQN.zip
├── policy.py
└── model.pt
```

```text
DA25M543_V024_Tabular_SARSA.zip
├── policy.py
└── sarsa_table.pkl
```

Before submission, the notebook checks:

- ZIP existence
- Root-level `policy.py`
- Matching model artifact
- No `__pycache__` directory
- No `.pyc` files
- Valid ZIP structure

---

## Public Leaderboard Results

The following public leaderboard costs were observed during project development. Lower cost is better.

| Technique | Public average cost, 20 episodes |
|---|---:|
| Neural Network based SARSA | 101,762.62 |
| DQN | 103,568.25 |
| Double DQN | 110,632.25 |
| Neural Network based Q-Learning | 116,460.12 |
| TD(lambda) with Eligibility Traces | 228,082.12 |
| Tabular SARSA | 234,995.38 |
| Tabular Q-Learning | 241,705.12 |
| PPO | 421,798.00 |
| A3C | 508,361.62 |
| REINFORCE with or without a baseline | 588,684.25 |
| A2C | 893,847.12 |

These values are historical public leaderboard observations, not guaranteed outputs of a fresh training run. Re-executing training can produce different models and costs. Final reporting should use the frozen policies actually selected for submission.

### Main observations

- Neural Network based SARSA, DQN, Double DQN, and Neural Network based Q-Learning produced the strongest public results.
- DQN-family methods benefited from replay memory and target networks.
- Tabular methods were easier to interpret but were limited by state discretization and joint-action representation.
- Policy-gradient methods were more sensitive to reward scale, trajectory batch size, critic quality, and deterministic action collapse.
- TD(lambda) improved credit assignment but still depended heavily on state quality, action coverage, learning rate, and trace decay.

---

## Reproducibility

To reproduce an experiment:

1. Use roll number `DA25M543` with the official configuration generator.
2. Confirm variant `V024` and fingerprint `db750dbfac3be81f`.
3. Use the supplied environment package without modification.
4. Run the technique training cell with the recorded random seed and hyperparameters.
5. Save the trained model immediately after training.
6. Evaluate using the common local validation seeds and scenarios.
7. Export the frozen model and deterministic policy.
8. Run `policy_validation_tests.py` on the exported file.
9. Create a ZIP containing only the policy and required model artifact.
10. Record the submission identifier and resulting leaderboard cost.

Training exploration may be stochastic. Submitted inference must remain deterministic.

---

## Generated Artifacts

### Results

The `results/` directory contains items such as:

```text
all_policy_comparison.csv
submission_manifest.csv
q_learning_training_results.csv
q_learning_validation_episode_results.csv
sarsa_training_results.csv
sarsa_validation_results.csv
dqn_training_results.csv
dqn_validation_results.csv
```

The exact filenames depend on the experiments executed.

### Models

The `models/` directory contains trained artifacts such as:

```text
q_learning_table.pkl
sarsa_table.pkl
dqn_model.pt
double_dqn_model.pt
ppo_model.pt
```

### Leaderboard packages

The `leaderboard/` directory contains technique-specific export folders and final ZIP files.

---

## Git Workflow

Initialize the local repository if needed:

```bash
git init
git branch -M main
git remote add origin https://github.com/amitkislay/RL_Inventry_Project.git
```

If the `origin` remote already exists:

```bash
git remote -v
git remote set-url origin https://github.com/amitkislay/RL_Inventry_Project.git
```

Add and commit the project:

```bash
git add README.md RL_Inventory_Project_DA25M543.ipynb requirements.txt

git commit -m "Add final RL inventory project notebook and documentation"
```

Push to GitHub:

```bash
git push -u origin main
```

Large generated model files and local training artifacts should normally be excluded unless they are required for project reproduction and permitted by the course submission rules.

---

## Suggested `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]

# Virtual environments
.venv/
venv/

# Jupyter
.ipynb_checkpoints/

# Local editor settings
.vscode/
.idea/

# Temporary files
*.tmp
*.log

# Generated caches
checkpoints/

# Operating system files
.DS_Store
Thumbs.db
```

Do not ignore required final policies or model artifacts if the repository is intended to reproduce the submitted leaderboard packages.

---

## Limitations

- Public leaderboard performance does not guarantee equivalent private leaderboard performance.
- Tabular policies generalize only through the selected state discretization.
- A restricted joint-action list can prevent product-specific order combinations.
- Larger action spaces require substantially more exploration.
- Factorized neural heads may not fully represent interactions caused by shared capacity.
- Policy-gradient techniques require careful return scaling, critic training, and rollout design.
- Model selection using too few seeds can overfit local validation.
- A fresh training run may not reproduce the exact public leaderboard cost unless the entire environment, package versions, seeds, training order, and checkpoint selection are identical.

---

## Academic Integrity

This repository is associated with roll number `DA25M543` and assigned variant `V024`. The configuration, notebook, trained models, and submitted policies must not be reused as another student's work.

The project follows these principles:

- Use only the assigned configuration.
- Do not alter the official environment or evaluator.
- Do not hard-code hidden evaluation cases.
- Declare every submitted technique correctly.
- Do not present hyperparameter variations as distinct techniques.
- Keep training evidence and exported policies reproducible.

---

## Final Submission Checklist

- [ ] Roll number is `DA25M543`.
- [ ] Assigned variant is `V024`.
- [ ] Configuration fingerprint is `db750dbfac3be81f`.
- [ ] The final notebook runs with the selected Python 3.14 environment.
- [ ] The official environment package is unchanged.
- [ ] Training and validation results are saved.
- [ ] The all-policy comparison table is generated.
- [ ] Submitted techniques are genuinely distinct.
- [ ] Every submitted `policy.py` returns valid quantities.
- [ ] Every submitted policy is deterministic.
- [ ] Every policy passes `policy_validation_tests.py`.
- [ ] Every ZIP contains `policy.py` at its root.
- [ ] Every ZIP contains its matching model artifact.
- [ ] ZIP files do not include cache files.
- [ ] The final submission manifest maps techniques to artifacts.
- [ ] The report and repository contain no unsupported performance claims.

---

## Author

**Amit Kumar**  
IITM Web M.Tech Program  
Roll number: `DA25M543`
