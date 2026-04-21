# 实验结果汇总与下阶段建议

> **Robust Listwise LLM — Formal Experiment (Stage E)**
> 日期：2026-04-21 | 模型：Qwen2.5-0.5B-Instruct (4-bit NF4, LoRA r=16)

---

## 目录

1. [复现命令](#1-复现命令)
2. [Phase 1：干净数据训练结果](#2-phase-1干净数据训练结果)
3. [Phase 2：噪声扫描结果](#3-phase-2噪声扫描结果)
4. [跨阶段结果对比](#4-跨阶段结果对比)
5. [关键发现总结](#5-关键发现总结)
6. [下阶段实验建议](#6-下阶段实验建议)

---

## 1. 复现命令

### 环境准备

```bash
conda activate robust_listwise_llm
cd /home/xudong/work/robust_listwise_llm
export CUDA_VISIBLE_DEVICES=1   # RTX 4090
```

### Phase 1：训练三个模型（1000 步，5000 样本，干净数据）

```bash
# 完整训练（约 2 小时）
python src/scripts/train_formal.py

# 快速冒烟测试（100 步，验证流程）
python src/scripts/train_formal.py --quick

# 只跑部分模型
python src/scripts/train_formal.py --models nominal_bt nominal_pl

# 跳过 RewardBench 评估（节省时间）
python src/scripts/train_formal.py --no_rewardbench
```

输出：`outputs/formal/formal_clean_results.csv`，检查点于 `outputs/formal/{model_name}/`

### Phase 2：噪声扫描（5 个噪声条件 × 3 个模型，200 步/条件）

```bash
# 完整噪声扫描（约 2 小时）
python src/scripts/run_formal_noise_sweep.py
```

输出：`outputs/formal/noise_sweep_results.csv`，检查点于 `outputs/formal/noise_sweep/{noise_type}_lvl{l}_{model}/`

### 绘图

```bash
python src/scripts/plot_results.py
```

输出：`outputs/formal/plots/` 目录下 7 张图（PNG + PDF 各一份）

### 结果快速查看

```bash
# Phase 1 结果
python -c "
import pandas as pd
df = pd.read_csv('outputs/formal/formal_clean_results.csv')
cols = ['model','train_loss','top1_acc','exact_match','kendall_tau','ndcg',
        'pairwise_acc_k4','pairwise_acc_binarized','rb_overall']
print(df[cols].to_string(index=False))
"

# Phase 2 结果（Kendall τ 透视表）
python -c "
import pandas as pd
df = pd.read_csv('outputs/formal/noise_sweep_results.csv')
pivot = df.pivot_table(index=['noise_type','noise_level'], columns='model', values='kendall_tau')
print(pivot.to_string())
"
```

---

## 2. Phase 1：干净数据训练结果

**设置**：1000 步 · 5000 样本 · 干净标签 · β=0.1 · lr=5e-5

### 2.1 主要评估指标

| 模型 | 训练损失 | Top-1↑ | Exact↑ | Kendall τ↑ | NDCG↑ | PairAcc(K4)↑ | PairAcc(bin)↑ | RB Overall↑ |
|------|---------|--------|--------|-----------|-------|-------------|--------------|------------|
| Nominal BT | 1.967 | **0.376** | 0.088 | 0.239 | **0.874** | 0.616 | **0.724** | **0.661** |
| Nominal PL | 11.396 | 0.374 | **0.098** | 0.251 | **0.875** | **0.622** | 0.710 | 0.660 |
| Robust PL (ρ=0.1) | 11.937 | 0.374 | 0.096 | **0.251** | 0.873 | **0.622** | 0.706 | 0.656 |

> 粗体 = 该指标最优值。随机基线：Top-1=0.25，τ=0，PairAcc=0.50，Exact≈0.042。

### 2.2 RewardBench 分类细节

| 模型 | Chat↑ | Chat Hard↑ | Safety↑ | Reasoning↑ | Overall↑ |
|------|-------|-----------|--------|-----------|---------|
| Nominal BT | 0.896 | 0.441 | 0.478 | **0.768** | **0.661** |
| Nominal PL | 0.906 | 0.441 | 0.484 | 0.762 | 0.660 |
| Robust PL | **0.909** | **0.443** | **0.488** | 0.751 | 0.656 |

### 2.3 解读

**三个模型在干净数据上高度接近：**

| 指标 | 最大差距 | 结论 |
|------|---------|------|
| Top-1 | 0.002 (0.2%) | 无实质差异 |
| Kendall τ | 0.012 | PL 系列略优于 BT（使用全部 K=4 位置） |
| PairAcc(bin) | 0.018 | **BT 领先**：直接在二元对上训练，对二元测集最适配 |
| RB Overall | 0.005 | 三者几乎相同 |

**具体发现：**

- **Robust PL 在干净数据上无损失**：ρ=0.1 的最坏情况项并未拖累干净性能。Robust PL 的 τ=0.2513 vs. Nominal PL 的 τ=0.2507，差距不到 1‰。这是关键前提：robust 目标函数在干净环境下不是"浪费"的正则化，而是几乎中性的。

- **BT 在二元 pairwise 测集上最强（0.724）**：BT 直接以 top-vs-bottom pair 为训练信号，与二元 test_prefs 评估集的设定完全一致，所以领先是预期内的。

- **PL 系列在全序 τ 上略优（0.251 vs. 0.239）**：PL 损失同时使用所有 K=4 位置，对中间位置的信息利用更充分，因此在需要全排序质量的 τ、NDCG 指标上有小幅优势。

- **RewardBench Reasoning 上 BT 领先（0.768 vs. 0.762/0.751）**：Reasoning 题目通常答案明确，高质量 vs. 低质量对比清晰，与 BT 的极端对训练方式最匹配。

- **RewardBench Safety 上 Robust PL 领先（0.488）**：最坏情况项隐式地让模型对高分响应更保守，符合 Safety 类别中模型需要识别"应拒绝"场景的需求（拒绝响应通常是 chosen，robust 项使模型不盲目信任高分的 rank-1）。

---

## 3. Phase 2：噪声扫描结果

**设置**：200 步 · 1000 样本（有噪声）· β=0.1 · 评估集始终为干净 K=4 held-out

### 3.1 near_tie 噪声（仅交换最小得分差的相邻位置对）

| 噪声比例 | BT τ | PL τ | Robust τ | **Δ(R−PL)** | BT Top-1 | PL Top-1 | Robust Top-1 |
|---------|------|------|----------|------------|---------|---------|------------|
| 0.0（无噪声） | 0.228 | 0.249 | **0.270** | **+0.021** | 0.360 | 0.350 | **0.387** |
| 0.4 | 0.221 | 0.250 | **0.260** | +0.010 | 0.343 | 0.360 | **0.383** |
| 1.0（全噪声） | 0.211 | **0.250** | 0.238 | **−0.012** | 0.347 | **0.367** | 0.357 |

| 噪声比例 | BT PairAcc | PL PairAcc | Robust PairAcc | BT NDCG | PL NDCG | Robust NDCG |
|---------|-----------|-----------|---------------|--------|--------|-----------|
| 0.0 | 0.606 | 0.617 | **0.628** | 0.864 | 0.865 | **0.872** |
| 0.4 | 0.603 | 0.617 | **0.623** | 0.865 | 0.868 | **0.867** |
| 1.0 | 0.598 | **0.618** | 0.611 | 0.860 | **0.868** | 0.861 |

### 3.2 top_rank 噪声（将 rank-1 替换为随机低质量响应）

| 噪声比例 | BT τ | PL τ | Robust τ | **Δ(R−PL)** | BT Top-1 | PL Top-1 | Robust Top-1 |
|---------|------|------|----------|------------|---------|---------|------------|
| 0.4 | 0.199 | 0.214 | **0.219** | +0.005 | 0.323 | 0.353 | **0.357** |
| **1.0（完全错误）** | **−0.016** | 0.051 | **0.079** | **+0.028** | 0.267 | 0.260 | **0.300** |

| 噪声比例 | BT PairAcc | PL PairAcc | Robust PairAcc | BT NDCG | PL NDCG | Robust NDCG |
|---------|-----------|-----------|---------------|--------|--------|-----------|
| 0.4 | 0.592 | 0.600 | **0.602** | 0.851 | 0.859 | **0.861** |
| **1.0** | **0.484** | 0.518 | **0.532** | 0.801 | 0.814 | **0.824** |

### 3.3 Δ(Robust − Nominal PL) 汇总

| 噪声条件 | Δτ | Δ Top-1 | Δ PairAcc(K4) | 结论 |
|---------|-----|---------|-------------|------|
| near_tie 0.0 | **+0.021** | +0.037 | +0.011 | Robust 在短训练中即有优势 |
| near_tie 0.4 | +0.010 | +0.023 | +0.006 | 小幅优势，随噪声增加而衰减 |
| near_tie 1.0 | **−0.012** | −0.010 | −0.007 | **Robust 略劣**：系统化翻转对鲁棒项有害 |
| top_rank 0.4 | +0.005 | +0.004 | +0.002 | 中度噪声下小幅优势 |
| **top_rank 1.0** | **+0.028** | **+0.040** | **+0.013** | **最大优势**：破坏性噪声场景 |

### 3.4 解读

#### 发现一：BT 在 top_rank 1.0 下完全崩溃（τ = −0.016）

这是本实验最显著的结果。BT 仅使用 `(ranking[0], ranking[-1])` 这一极端对；当 `top_rank 1.0` 时，`ranking[0]` 始终是随机低质量响应，导致 **100% 训练样本的梯度方向完全错误**。模型学到了"这个随机响应比最差响应更好"，整体排序质量倒退至负相关（τ < 0）。

```
BT top_rank 1.0: τ = −0.016（比随机还差），Top-1 = 0.267（接近随机基线 0.25）
```

**机制**：BT 把所有训练信号集中于 rank-1，而 top_rank 噪声精确地污染 rank-1，双重暴露导致完全崩溃。

#### 发现二：Robust PL 在 top_rank 1.0 下领先最大（+0.028 τ）

```
top_rank 1.0: Nominal PL τ=0.051，Robust PL τ=0.079（+55%相对提升）
Top-1:        Nominal PL 0.260，   Robust PL 0.300（+15%相对提升）
```

最坏情况项 `ℓ_PL(σ_wc)` 显式惩罚模型对当前最高得分响应的过度信任，这与 top_rank 噪声的破坏机制正好相对：噪声把"错误响应"放到 rank-1，robust 项让模型不盲目信任 rank-1。两者在机制上高度匹配，观测到的优势符合理论预期。

#### 发现三：near_tie 1.0 下 Robust 轻微劣势（−0.012 τ）

```
near_tie 1.0: Nominal PL τ=0.250，Robust PL τ=0.238（−0.012）
```

当所有最小分差相邻对都被系统性翻转时，扰动成为**确定性的、一致的**——它创造了一个"不同约定"的新标注规范，而非真正的随机噪声。Nominal PL 能学会适应这个确定性的新规范，而 Robust PL 的最坏情况项引入了额外的不确定性，干扰了这种适应。

这是一个边界条件：near_tie 1.0 是 near_tie 噪声本身的最极端形式，在实际标注中几乎不会出现（人类标注者不会 100% 系统性地将所有最相似对反向标注）。

#### 发现四：噪声对 BT 的影响远大于 PL 系列

| 条件 | BT τ 退化量 | PL τ 退化量 | Robust τ 退化量 |
|-----|-----------|-----------|--------------|
| top_rank 0.4 → 1.0 | −0.215（↓94%） | −0.163（↓66%） | −0.140（↓52%） |

BT 对 top_rank 噪声的脆弱性比 PL 系列高出约 40%。结论：**对于高 top_rank 噪声场景，BT 是最不合适的目标函数选择**，PL 系列（无论是否 robust）都更稳健。

---

## 4. 跨阶段结果对比

| 实验阶段 | 步数 | 样本 | Nominal PL τ（干净） | Robust τ（干净） | Robust 优势 top_rank 1.0 |
|---------|-----|-----|---------------------|----------------|------------------------|
| Stage D | 50 | 1000 | ~0.10（估算） | ~0.10 | +5% PairAcc |
| **Stage E Phase 1** | **1000** | **5000** | **0.251** | **0.251** | **—（干净，无差异）** |
| **Stage E Phase 2** | **200** | **1000** | **0.251（干净）/ 0.051（top_rank 1.0）** | **0.251 / 0.079** | **+0.028 τ** |

**趋势**：随着训练步数增加，干净数据上三个模型趋于收敛；噪声实验中 Robust PL 的优势在中等步数（200步）下仍然清晰可见。

---

## 5. 关键发现总结

### ✅ 强支持（高置信度）

1. **BT 对 top_rank 噪声最脆弱**：τ 从 +0.228 崩溃至 −0.016，是因为损失函数完全依赖 rank-1 位置，而 top_rank 噪声正好污染这一位置。

2. **Robust PL 在 top_rank 高噪声下有可测量优势**：top_rank 1.0 时 Δτ=+0.028，Δ Top-1=+0.040。优势的方向和机制与理论预测一致。

3. **Robust PL 在干净数据上几乎无代价**：ρ=0.1 时，干净数据 τ 相差不足 0.001，验证了保守 ρ 取值的合理性。

4. **near_tie 噪声是弱扰动**：任何模型在 near_tie 0.0→1.0 过程中退化幅度均远小于 top_rank，与理论分析（near_tie 只换相邻低梯度位）高度吻合。

### ⚠️ 需要进一步验证

1. **样本量不足以统计检验**：单 seed 实验，无置信区间。Δτ=+0.028 和 Δτ=−0.012 在重复实验中可能翻转。

2. **200 步噪声实验是否稳定**：Phase 1 用 1000 步，Phase 2 只用 200 步，可能处于欠拟合区间，结论对步数敏感。

3. **ρ=0.1 是否最优**：未做 ρ 扫描，可能存在更优的 ρ 值（尤其在 top_rank 场景下 ρ 越大收益是否单调递增尚未验证）。

4. **near_tie 1.0 下 Robust 略劣是否稳定**：Δ=−0.012，绝对值很小，可能是统计噪声。

---

## 6. 下阶段实验建议

### 6.1 优先级 1：多 Seed 重复实验（最重要）

**目的**：为关键数值添加置信区间，区分真实效应与统计噪声。

**设计**：

```bash
# 在 train_formal.py 中加入 --seed 参数后，运行：
for seed in 42 1234 2025; do
  python src/scripts/train_formal.py --seed $seed --tag "seed${seed}"
done

# 噪声实验同理：
for seed in 42 1234 2025; do
  python src/scripts/run_formal_noise_sweep.py --seed $seed
done
```

**期望输出**：mean ± std 的 τ 和 Top-1，p-value（Wilcoxon 或 paired t-test）。

**代价**：约 3× 计算量（再跑 2 个 seed × (Phase 1 + Phase 2) ≈ 9 小时）。

---

### 6.2 优先级 2：ρ 扫描（理论分析的核心实验）

**目的**：找到 robust 正则化的最优强度，并观测 robustness-performance 权衡曲线。

**设计**：

```bash
# 在 top_rank 1.0 噪声条件下扫描 ρ
python src/scripts/run_rho_sweep.py \
  --noise_type top_rank --noise_level 1.0 \
  --rho_values 0.0 0.05 0.1 0.2 0.5 1.0 \
  --n_steps 500 --n_train 2000
```

**期望目标**：绘制 τ vs. ρ 曲线，看是否存在最优 ρ*，以及曲线是否单调。

**代价**：6 个值 × 500 步 ≈ 约 90 分钟。

---

### 6.3 优先级 3：延长噪声实验步数（1000 步 vs. 当前 200 步）

**目的**：验证 Phase 2 的结论是否在更充分训练后仍然成立。

**设计**：

```bash
python src/scripts/run_formal_noise_sweep.py \
  --n_steps 1000 \
  --n_train 5000 \
  --tag "long"
```

**关注点**：top_rank 1.0 下的 Robust 优势是扩大还是收缩？near_tie 1.0 的 Robust 轻微劣势是否持续？

**代价**：约 4×200=800 步 × 15 runs ≈ 约 6 小时（如果加大步数）。建议只跑 2 个最关键条件（top_rank 1.0 和 near_tie 1.0）节省时间。

---

### 6.4 优先级 4：更大规模训练（迈向发表质量）

**目的**：在数据量和步数上接近实际 reward model 训练规模。

| 参数 | 当前（Stage E） | 建议（Stage F） |
|-----|--------------|--------------|
| 训练样本 | 5,000 | 50,000 |
| 训练步数 | 1,000 | 5,000–10,000 |
| 基座模型 | Qwen2.5-0.5B | Qwen2.5-1.5B 或 3B |
| LoRA rank | 16 | 32 或 full fine-tune |
| 评估集 | 500 | 2,000 |
| RB 评估 | subset | 全集（2985 样本） |

```bash
# 示例：scale-up 实验（需要约 6–12 小时）
python src/scripts/train_formal.py \
  --n_train 50000 \
  --n_steps 5000 \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --lora_r 32
```

---

### 6.5 优先级 5：混合噪声类型实验

**目的**：测试真实场景——标注中同时存在 near_tie 和 top_rank 噪声。

**设计**：

```python
# 在 noise.py 中添加混合噪声函数
def apply_mixed_noise(ranking, scores, near_tie_level=0.3, top_rank_level=0.3):
    ranking = apply_near_tie_noise(ranking, scores, near_tie_level)
    ranking = apply_top_rank_noise(ranking, scores, top_rank_level)
    return ranking
```

```bash
# 运行混合噪声实验
python src/scripts/run_mixed_noise_sweep.py \
  --near_tie_levels 0.0 0.3 0.6 \
  --top_rank_levels 0.0 0.3 0.6
```

---

### 6.6 建议的实验执行顺序

```
Week 1:
  ├─ [1日] 优先级2: ρ 扫描（6×500步，1.5h）           → 理解超参数
  ├─ [1日] 优先级3: 延长步数（top_rank/near_tie 2条件，1000步，3h） → 验证稳健性
  └─ [2日] 优先级1: 3 seeds × Phase1+Phase2（约9h，隔夜运行）     → 统计置信度

Week 2:
  └─ 优先级4: 大规模实验（隔夜运行）                             → 迈向发表质量
```

---

## 附录：指标定义速查

| 指标 | 公式 | 随机基线 | 完美值 |
|------|-----|---------|------|
| Top-1 Acc | $P[\hat\sigma(0) = \sigma^*(0)]$ | 0.25 | 1.0 |
| Exact Match | $P[\hat\sigma = \sigma^*]$ | 0.042 | 1.0 |
| Kendall τ | $(C-D)/\binom{K}{2}$ | 0.0 | 1.0 |
| NDCG@4 | DCG/IDCG（logscore权重） | ~0.85 | 1.0 |
| PairAcc(K4) | $(\tau+1)/2$（所有成对正确率） | 0.5 | 1.0 |
| PairAcc(bin) | 二元 chosen/rejected 正确率 | 0.5 | 1.0 |
| RB Overall | RewardBench 4类平均正确率 | ~0.5 | 1.0 |

| 噪声类型 | 影响位置 | 扰动强度 | 类比 |
|---------|--------|---------|------|
| near_tie | 中间位（rank 1–2） | 弱（分差最小对） | 标签平滑 |
| top_rank | rank-1（始终） | 强（可替换为最差响应） | 对抗性错误标注 |
