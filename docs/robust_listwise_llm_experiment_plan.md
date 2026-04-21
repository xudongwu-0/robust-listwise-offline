# Robust Listwise LLM Experiments — Environment, Implementation Plan, and Agent Brief

## 1. Goal

Move from the current fixed-list simulation to a real LLM experiment using a small instruction model such as **Qwen2.5-0.5B-Instruct**.

The objective is **not** to immediately build the final full system.
The objective is to establish a clean experimental ladder:

1. verify the training stack works,
2. reproduce a standard preference-training baseline,
3. implement the robust listwise objective on top of that stack,
4. run moderate-scale experiments and compare nominal / robust / pairwise / regularized baselines.

---

## 2. Recommended stack

Use the following stack as the main engineering base:

- `transformers`
- `trl`
- `datasets`
- `accelerate`
- `peft`
- `bitsandbytes`
- `wandb`
- `scipy`
- `numpy`
- `pandas`
- `matplotlib`

Use **TRL** as the training framework, not as a complete implementation of robust listwise by default.
The expected workflow is:

- use TRL for the training skeleton,
- use PEFT/LoRA for parameter-efficient finetuning,
- use bitsandbytes 4-bit loading if memory becomes tight,
- customize the trainer / loss / collator for robust listwise training.

---

## 3. Recommended base model

Start with:

- `Qwen/Qwen2.5-0.5B-Instruct`

Why this model:
- small enough to iterate quickly,
- instruction-tuned already,
- realistic enough to test post-training behavior,
- much closer to the final LLM setting than the current log-linear simulation.

Do **not** start from a larger model until the pipeline is stable.

---

## 4. Recommended datasets

### Stage A: pairwise baseline sanity check
Use:
- `HuggingFaceH4/ultrafeedback_binarized`

Why:
- simplest path to verify TRL + Qwen + DPOTrainer works,
- standard pairwise preference format,
- fast to debug.

### Stage B: main fixed-list robust listwise experiment
Use one of:
- `openbmb/UltraFeedback` (4 completions per prompt)
- `berkeley-nest/Nectar` (7-wise ranked responses)

Recommended order:
1. start with `openbmb/UltraFeedback`,
2. move to `Nectar` after the pipeline is stable.

Reason:
- UltraFeedback is easier to wire up first,
- Nectar is better for stronger listwise evidence later.

---

## 5. Environment setup

### 5.1 Create project folder

```bash
cd ~/work
mkdir -p robust_listwise_llm
cd robust_listwise_llm

mkdir -p docs
mkdir -p data/raw
mkdir -p data/processed
mkdir -p src/data
mkdir -p src/trainers
mkdir -p src/losses
mkdir -p src/models
mkdir -p src/scripts
mkdir -p outputs/logs
mkdir -p outputs/plots
mkdir -p outputs/checkpoints
touch README.md
touch requirements.txt
```

### 5.2 Create conda environment

```bash
conda create -n robust_listwise_llm python=3.10 -y
conda activate robust_listwise_llm
```

### 5.3 Install core libraries

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets trl accelerate peft bitsandbytes wandb scipy numpy pandas matplotlib sentencepiece
```

### 5.4 Save dependency snapshot

```bash
pip freeze > requirements.txt
```

### 5.5 Reopen later

```bash
cd ~/work/robust_listwise_llm
conda activate robust_listwise_llm
```

If `conda activate` fails:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate robust_listwise_llm
```

---

## 6. Memory strategy for a 4090

Default recommendation:

- load Qwen2.5-0.5B-Instruct with 4-bit quantization,
- use LoRA adapters,
- use bf16 if supported,
- keep batch size small and rely on gradient accumulation.

This is the safest first setup for a single 4090.

---

## 7. Project roadmap

## Stage 0 — Confirm the training stack works
Goal:
- verify the environment can load Qwen,
- verify TRL can train a standard baseline,
- verify logging/checkpointing works.

Task:
- run a small **pairwise DPO baseline** using `ultrafeedback_binarized`
- train for a very small number of steps
- confirm loss decreases and checkpoints save correctly

Deliverable:
- one working baseline script
- one tiny checkpoint
- one short note that the stack is functional

---

## Stage 1 — Reproduce a standard preference baseline
Goal:
- establish a pairwise baseline before custom listwise loss

Task:
- use Qwen2.5-0.5B-Instruct
- use `HuggingFaceH4/ultrafeedback_binarized`
- use TRL DPOTrainer
- train with LoRA
- log train loss / eval loss / generation samples

Deliverable:
- pairwise baseline result
- exact config file
- small report

This stage is important because if this does not work, your custom listwise trainer will be impossible to debug.

---

## Stage 2 — Build fixed-list listwise training
Goal:
- move from pairwise DPO to full fixed-list ranking training

Task:
- switch to original `openbmb/UltraFeedback`
- group by prompt
- keep the 4 responses as one list
- derive clean listwise ranking from overall scores
- build a collator that returns:
  - prompt
  - list of K responses
  - ranking label
  - optional noisy ranking label

At this stage, do **not** add robustness yet.
First implement:
- nominal listwise PL training on Qwen log-prob-based scores

Deliverable:
- one nominal listwise trainer
- one dataset wrapper for fixed K-way lists
- one evaluation script

---

## Stage 3 — Add robust listwise objective
Goal:
- implement the actual method

Task:
- compute per-candidate score:
  - ideally from log-prob / reference-log-prob style score
- compute nominal PL loss
- compute worst-case ranking from current candidate scores
- compute robust loss:
  - `(1-rho) * nominal + rho * worst_case`
- compare:
  - nominal listwise
  - robust listwise
  - pairwise baseline
  - nominal + weight decay regularized baseline

Deliverable:
- custom robust listwise trainer
- clean experiment config
- side-by-side comparison table

---

## 8. How to map your simulation idea to real LLM training

In simulation, you trained:

- frozen feature extractor,
- linear score model,
- fixed candidate lists,
- noisy ranking labels.

In the real LLM setting, replace the score source:

### Simulation
```text
g_theta(x, y) = theta^T phi(x, y)
```

### LLM version
Use one scalar score per candidate response, for example:
```text
g_theta(x, y) = beta * [ log pi_theta(y|x) - log pi_ref(y|x) ]
```

Then:
- rank the K responses by this score,
- compute nominal PL loss on the observed ranking,
- compute worst-case ranking by sorting scores ascending,
- compute robust listwise loss.

So the conceptual pipeline stays the same.
Only the score function changes from a linear feature score to an LLM-derived score.

---

## 9. Minimal experiment plan for the first real LLM round

Run only this first:

### Model
- Qwen2.5-0.5B-Instruct

### Parameterization
- 4-bit loading
- LoRA finetuning

### Data
- pairwise baseline: `HuggingFaceH4/ultrafeedback_binarized`
- then original `openbmb/UltraFeedback`

### Comparisons
- pairwise DPO baseline
- nominal listwise
- robust listwise
- nominal listwise + weight decay

### Noise / ranking
At first, do **not** add synthetic noise to the real LLM run.
First establish that:
- the nominal listwise trainer works,
- the robust listwise trainer runs stably,
- scores and rankings are computed correctly.

Only after the real trainer is stable should you consider controlled ranking corruption experiments again.

---

## 10. Recommended repository structure

```text
robust_listwise_llm/
├── docs/
│   ├── EXPERIMENT_PLAN.md
│   ├── AGENT_TASK_BRIEF.md
│   └── THEORY_TO_LLM_MAPPING.md
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   │   ├── load_pairwise_data.py
│   │   ├── load_fixed_list_data.py
│   │   └── collators.py
│   ├── losses/
│   │   ├── pairwise_losses.py
│   │   └── listwise_losses.py
│   ├── trainers/
│   │   ├── run_dpo_baseline.py
│   │   ├── run_nominal_listwise.py
│   │   └── run_robust_listwise.py
│   ├── models/
│   │   └── score_utils.py
│   └── scripts/
│       ├── sanity_check_scores.py
│       └── evaluate_generations.py
├── outputs/
│   ├── logs/
│   ├── plots/
│   └── checkpoints/
├── README.md
└── requirements.txt
```

---

## 11. What the vibe-coding agent should do first

The agent should **not** jump directly to the final robust trainer.

The correct order is:

1. environment bootstrap
2. model loading sanity check
3. standard pairwise DPO baseline
4. fixed-list dataset wrapper
5. nominal listwise loss
6. robust listwise loss
7. experiment comparison

If it skips step 3, debugging will become much harder.

---

## 12. Exact agent brief

Copy the following into a file named `docs/AGENT_TASK_BRIEF.md` and give it to the coding agent.

```md
# AGENT TASK BRIEF — Robust Listwise LLM Experiments

## Goal
Build the first real LLM experiment pipeline for robust listwise preference optimization.

## Main principles
- Use Hugging Face TRL as the training base.
- Use Qwen2.5-0.5B-Instruct as the base model.
- Use LoRA and 4-bit loading for fast iteration on a single 4090.
- Do not redesign the algorithm.
- Move in stages: pairwise baseline first, then nominal listwise, then robust listwise.

## Stage order

### Stage 0
Confirm the environment works:
- load model
- tokenize sample data
- run one forward pass
- save one tiny checkpoint

### Stage 1
Build and run a pairwise DPO baseline:
- dataset: HuggingFaceH4/ultrafeedback_binarized
- trainer: TRL DPOTrainer
- objective: standard pairwise preference optimization
- output: one working baseline script and one short result summary

### Stage 2
Build fixed-list data pipeline:
- dataset: openbmb/UltraFeedback
- group responses by prompt
- keep K=4 responses per prompt
- derive ranking labels from overall_score
- implement dataset wrapper and collator

### Stage 3
Implement nominal listwise training:
- compute one scalar score for each candidate response
- sort candidates by score
- compute nominal PL loss on the provided ranking
- train successfully end-to-end

### Stage 4
Implement robust listwise training:
- compute worst-case ranking from current scores
- compute robust loss:
  (1-rho) * nominal_PL + rho * worst_case_PL
- compare against:
  - nominal listwise
  - pairwise baseline
  - nominal + weight decay

## Constraints
- Do not expand to large models.
- Do not run huge sweeps.
- Keep the first experiments moderate and debuggable.
- Keep code modular.
- Prefer a working small experiment over a large unfinished framework.

## Required outputs
1. concise README
2. exact run commands
3. one pairwise baseline
4. one nominal listwise trainer
5. one robust listwise trainer
6. one compact comparison table
```

---

## 13. Exact shell commands to bootstrap

```bash
cd ~/work
mkdir -p robust_listwise_llm
cd robust_listwise_llm

mkdir -p docs data/raw data/processed src/data src/trainers src/losses src/models src/scripts outputs/logs outputs/plots outputs/checkpoints
touch README.md requirements.txt

conda create -n robust_listwise_llm python=3.10 -y
conda activate robust_listwise_llm

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets trl accelerate peft bitsandbytes wandb scipy numpy pandas matplotlib sentencepiece

pip freeze > requirements.txt
```

---

## 14. What you should ask the agent to verify

Before trusting any result, require the agent to verify:

1. model loads correctly in 4-bit + LoRA mode
2. DPO baseline runs end-to-end
3. fixed-list grouping is correct
4. K candidates really correspond to the same prompt
5. ranking labels are derived correctly
6. nominal PL matches the pairwise reduction behavior at K=2
7. robust loss with rho=0 equals nominal loss
8. worst-case ranking is really ascending score order
9. all comparisons share the same split and seed policy

---

## 15. Final recommendation

For speed and reliability, the best first real-LLM path is:

1. **TRL + Qwen2.5-0.5B-Instruct**
2. **4-bit + LoRA**
3. **pairwise DPO baseline first**
4. **then fixed-list nominal listwise**
5. **then robust listwise**

Do not start by trying to train the final robust method directly.
The most reliable way to get results is to make the training ladder work step by step.
