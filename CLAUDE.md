# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A from-scratch reinforcement learning study project implementing classic RL algorithms with NumPy only — no RL frameworks. Each topic lives in its own top-level directory (`bandits/`, `MDP/`). The code favors heavy explanatory comments over abstraction; preserve that teaching style when editing.

## Environment & commands

- Python 3.13, NumPy is the only dependency (no requirements file — install with `pip install numpy`).
- Run a module's demo directly, from inside its directory so the sibling imports resolve:
  ```
  cd bandits && python env.py
  ```
- There is no test suite, linter, or build step. To sanity-check an agent, run it against the env for many steps and confirm it converges on the highest-probability arm (see how `run_bandit` reports "% optimal action").

## Architecture (bandits/)

The multi-armed bandit setup separates three concerns, connected by a shared **agent interface**:

- `env.py` — `BanditEnv`: each arm pays reward 1 with a fixed per-arm probability. Also the entry point: `run_bandit(agent, ...)` drives the act/observe/update loop, and `run_gradient_bandit` / `run_epsilon_greedy` are thin wrappers that build the respective agent. `run_bandit` adapts its policy printout by duck-typing on the agent (`H` → gradient preferences+softmax, `Q` → value estimates).
- `gradient_bandits.py` — `GradientBandit`: softmax policy over learned preferences `H`, updated by gradient ascent with a running-average reward baseline.
- `elipson_greedy.py` — `EpsilonGreedy`: incremental sample-average value estimates `Q`, ε-probability random exploration vs. greedy exploitation. (Note the misspelled filename.)

**Agent contract:** every agent exposes `act()` (returns an arm index) and `update(action, reward)`. Any new bandit agent that follows this contract drops straight into `run_bandit`. If it stores its policy under a new attribute (not `H` or `Q`), extend the printout branch in `run_bandit`.
