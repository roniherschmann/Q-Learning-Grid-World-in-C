# Q-learning Grid World (C)

Author: Roni Herschmann

A single-file Q-learning agent that learns to navigate a 2D grid world with walls to reach a goal. Built in pure C for speed and clarity—great for showcasing systems skills and AI fundamentals.

## Highlights

- Pure C (C11), single file—no external dependencies.
- Q-learning with epsilon-greedy exploration (with decay).
- Deterministic grid world with obstacles.
- CLI interface: train, render, save/load Q-table, play greedy policy.
- Binary serialization of the Q-table.

## Build

```
gcc -O2 -Wall -Wextra -std=c11 qgrid.c -o qgrid
```

Works on macOS/Linux with gcc or clang.

## Quick Start

Train from scratch and save the learned Q-table:

```
./qgrid --train 10000 --save qtable.bin
```

Load a Q-table, render each step, and play 5 greedy episodes:

```
./qgrid --load qtable.bin --render --play 5
```

Train while occasionally rendering progress (every 1000 episodes):

```
./qgrid --train 10000 --render-every 1000
```

Change grid size (≤ 10×10):

```
./qgrid --size 5 5 --train 8000 --save q.bin
```

Set seed for reproducibility:

```
./qgrid --seed 42 --train 5000
```

## What This Is

A compact implementation of model-free reinforcement learning (RL) using Q-learning. The agent learns a value for each state–action pair and follows the greedy policy (highest-Q action) at play time.

- **Grid cells**:
  - S = start
  - G = goal
  - # = wall
  - A = agent (current state)
  - . = empty

- **Rewards**:
  - Step: -1 per move (encourages shortest path)
  - Goal: +10 when reaching G

- Episode ends on reaching G or hitting a step limit.

## Core Concepts

- **Q-learning update (off-policy TD control)**:  
  For state s, action a, reward r, next state s':  
  Q(s,a) ← Q(s,a) + α (r + γ max_{a’} Q(s’,a’) - Q(s,a))  
  - α (alpha): learning rate  
  - γ (gamma): discount factor

- **Epsilon-greedy exploration with decay**:  
  - Choose a random action with probability ε, otherwise greedy (argmax_a Q(s,a)).  
  - ε decays exponentially from ε_start to ε_min to transition from exploration to exploitation.

- **Value-based policy**:  
  - At play time, the agent takes argmax_a Q(s,a)—no exploration.

## CLI Options

```
--train N          Train for N episodes
--play N           Play greedy policy for N episodes
--render           Render grid each step during play
--render-every N   Render training every N episodes
--save PATH        Save Q-table to PATH
--load PATH        Load Q-table from PATH
--size W H         Grid size (max 10x10)
--alpha A          Learning rate (default 0.1)
--gamma G          Discount factor (default 0.99)
--eps-start E      Starting epsilon (default 1.0)
--eps-min E        Minimum epsilon (default 0.05)
--eps-decay D      Epsilon decay rate (default 0.0025)
--seed S           RNG seed (default: time-based)
--help             Show usage
```

## Expected Behavior

- With step -1 and goal +10, the agent converges to a shortest path.
- On a 5×5 with the included walls, a typical shortest path is 8 steps: Return ≈ +3 (7 × -1 + 10).

Example play (greedy, with rendering):

```
Return: 3.00 | Steps: 8
```

## Q-table Format

- Stored as a binary blob:  
  - int w; int h;  
  - float q[w*h*4];  // 4 actions: up, right, down, left

- Loaded/saved with --load / --save.

## Customize

- **Environment**: Tweak obstacles in env_init (walls), rewards, and step limit.
- **Hyperparameters**: Pass --alpha, --gamma, --eps-* flags.
- **Grid size**: --size W H up to 10×10.

## Demo Scripts (optional)

Add simple shell scripts for reproducibility:

**train.sh**
```
#!/usr/bin/env bash
set -e
gcc -O2 -Wall -Wextra -std=c11 qgrid.c -o qgrid
./qgrid --seed 42 --size 5 5 --train 10000 --save qtable.bin
```

**play.sh**
```
#!/usr/bin/env bash
set -e
./qgrid --load qtable.bin --render --play 5
```

Make executable:

```
chmod +x train.sh play.sh
```

## Roadmap / Extensions

- Stochastic dynamics (e.g., wind/slip probability).
- Multiple terminal states (pits with negative reward).
- Policy printer (ASCII arrows) and state-value heatmap.
- Parallel training over seeds; aggregate Q-tables.
- CSV logging for learning curves (episode return, steps).

## Repo Structure

```
qgrid.c       # single-file implementation
README.md     # this file
qtable.bin    # (optional) saved Q-table artifact
```

## Troubleshooting

- “Nothing to do.” — You didn’t pass --train or --play.
- Mismatch sizes on load — Use --size W H to match the Q-table dimensions or retrain.
- No movement — Remember walls block movement; agent “bumps” and stays in place.
