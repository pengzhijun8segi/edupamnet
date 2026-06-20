"""
step4_ro3_task_correlation.py
Step 4 — RO3: Task Relationship Stability

RO3 定义（论文 §1.7.2 / Chapter 3）：
  研究 Task1（正确性预测）与 Task2（选项选择预测）之间的关系，
  并评估该关系在不同条件下是否"稳定"。

数据可用性说明（重要）：
  - NeurIPS  : 有 AnswerValue（多选项标签），Task1+Task2 均可计算
  - ASSISTments: 无选项标签（开放式/技能型题目），Task2 不适用
  → RO3 的 Task1-Task2 相关性分析为 NeurIPS-specific
  → 跨平台稳定性部分由 Step 3 已计算的 FISS 指标承担
    （FISS 衡量"特征重要性模式"在两平台间的稳定性，
     是 RO3 跨平台维度的替代量化指标）

本脚本计算（NeurIPS, 5种子 [42,123,7,2026,999]）：
  1. Task1: 正确性预测概率 P(correct)       — 二分类
  2. Task2: 选项预测置信度 max P(option)    — 多分类
  3. Task1-Task2 相关性 (Spearman)，按学生能力分组（3组）
  4. 稳定性 = 相关性在5个种子间的标准差（越小越稳定）

输出: results/step4_ro3_results.json
"""

import os, sys, json, warnings
import numpy as np
warnings.filterwarnings('ignore')

DATA_PATH   = './data/'
OUTPUT_PATH = './results/'
SEEDS       = [42, 123, 7, 2026, 999]
SAMPLE_SIZE = 30000

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 65)
print("Step 4 — RO3: Task Relationship Stability (NeurIPS)")
print(f"  Seeds: {SEEDS}")
print("=" * 65)

from data_loader02 import CrossPlatformDataLoader
from data_loader_aligned import align_features, NEURIPS_MAP, UNIFIED_FEATURES
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

# ════════════════════════════════════════════════════════════════
# 1. 加载 NeurIPS 数据（含 Task1 + Task2 标签）
# ════════════════════════════════════════════════════════════════
print("\n[1] Loading NeurIPS data (Task1 + Task2 labels)...")
loader = CrossPlatformDataLoader(data_path=DATA_PATH)
X_raw, y_correct, y_option, feat_names = loader.load_neurips_dataset(task='both')
print(f"  X shape       : {X_raw.shape}")
print(f"  Task1 (correctness) balance : {y_correct.mean():.3f}")
print(f"  Task2 (option) classes      : {len(np.unique(y_option))}")
print(f"  Task2 class distribution    : {np.bincount(y_option.astype(int))}")

# 对齐到统一5特征（与RO1/RO2一致，便于跨RO引用同一特征空间）
X_aligned = align_features(X_raw, feat_names, NEURIPS_MAP)
print(f"  Aligned features ({len(UNIFIED_FEATURES)}): {UNIFIED_FEATURES}")

# ── 去重（按特征+两个标签联合）─────────────────────────────────
def dedup(X, y1, y2):
    combined = np.hstack([X, y1.reshape(-1,1), y2.reshape(-1,1)])
    _, idx = np.unique(combined.round(6), axis=0, return_index=True)
    return X[idx], y1[idx], y2[idx]

X_dd, y_correct_dd, y_option_dd = dedup(
    X_aligned, y_correct.astype(np.float32), y_option.astype(np.float32))
print(f"  After dedup   : {X_dd.shape}")

# ════════════════════════════════════════════════════════════════
# 2. 多种子循环
# ════════════════════════════════════════════════════════════════
all_results = []

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*65}")
    print(f"SEED {seed}  ({seed_idx+1}/{len(SEEDS)})")
    print(f"{'='*65}")
    np.random.seed(seed)

    # 分层采样（按 Task1 标签分层，保留正确率分布）
    if len(X_dd) > SAMPLE_SIZE:
        _, X_s, _, y_corr_s, _, y_opt_s = train_test_split(
            X_dd, y_correct_dd, y_option_dd,
            test_size=SAMPLE_SIZE/len(X_dd),
            stratify=y_correct_dd.astype(int), random_state=seed)
    else:
        X_s, y_corr_s, y_opt_s = X_dd, y_correct_dd, y_option_dd

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_s).astype(np.float32)

    # train/test 划分
    X_tr, X_te, yc_tr, yc_te, yo_tr, yo_te = train_test_split(
        X_sc, y_corr_s, y_opt_s, test_size=0.2,
        stratify=y_corr_s.astype(int), random_state=seed)

    print(f"  Sample: {X_sc.shape}  Task1 balance={y_corr_s.mean():.3f}")

    # ── Task1: 正确性预测概率 ────────────────────────────────
    task1_model = LogisticRegression(max_iter=500, random_state=seed)
    task1_model.fit(X_tr, yc_tr.astype(int))
    task1_prob = task1_model.predict_proba(X_te)[:, 1]   # P(correct)

    # ── Task2: 选项预测置信度 ────────────────────────────────
    task2_model = RandomForestClassifier(n_estimators=100,
                                          random_state=seed, n_jobs=-1)
    task2_model.fit(X_tr, yo_tr.astype(int))
    task2_prob_matrix = task2_model.predict_proba(X_te)
    task2_confidence  = task2_prob_matrix.max(axis=1)    # max P(option)

    # ── Task1-Task2 相关性（整体）────────────────────────────
    corr_overall, p_overall = spearmanr(task1_prob, task2_confidence)
    corr_overall = float(corr_overall) if not np.isnan(corr_overall) else 0.0
    print(f"  Overall Task1-Task2 correlation: {corr_overall:.4f}  (p={p_overall:.4f})")

    # ── 按学生能力分组（3组）──────────────────────────────────
    ability = X_te[:, UNIFIED_FEATURES.index('student_ability')]
    bounds  = np.percentile(ability, [0, 33.3, 66.7, 100])
    group_corrs = {}
    for g in range(3):
        mask = (ability >= bounds[g]) & (ability <= bounds[g+1])
        if mask.sum() < 10:
            continue
        c, p = spearmanr(task1_prob[mask], task2_confidence[mask])
        c = float(c) if not np.isnan(c) else 0.0
        group_corrs[f'group_{g+1}'] = {
            'n': int(mask.sum()), 'spearman': round(c, 6), 'p_value': round(float(p), 6)
        }
        print(f"    Group {g+1} (n={mask.sum()}): Spearman={c:.4f}")

    all_results.append({
        'seed': seed,
        'task1_task2_correlation_overall': round(corr_overall, 6),
        'p_value_overall': round(float(p_overall), 6),
        'group_correlations': group_corrs,
        'task1_acc': float((task1_model.predict(X_te) == yc_te.astype(int)).mean()),
        'task2_acc': float((task2_model.predict(X_te) == yo_te.astype(int)).mean()),
    })

# ════════════════════════════════════════════════════════════════
# 3. 跨种子稳定性分析
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("STABILITY ANALYSIS (across 5 seeds)")
print(f"{'='*65}")

overall_corrs = [r['task1_task2_correlation_overall'] for r in all_results]
mean_corr = float(np.mean(overall_corrs))
std_corr  = float(np.std(overall_corrs))
ci95      = float(1.96 * std_corr / np.sqrt(len(overall_corrs)))

print(f"\n  Task1-Task2 Correlation (overall):")
print(f"    Mean = {mean_corr:.4f}  Std = {std_corr:.4f}  (95%CI ±{ci95:.4f})")
for r in all_results:
    print(f"    Seed {r['seed']:5d}: {r['task1_task2_correlation_overall']:.4f}")

# 按组的跨种子方差
print(f"\n  Per-group correlation stability:")
group_stability = {}
for g in ['group_1', 'group_2', 'group_3']:
    vals = [r['group_correlations'][g]['spearman']
            for r in all_results if g in r['group_correlations']]
    if vals:
        gm, gs = float(np.mean(vals)), float(np.std(vals))
        group_stability[g] = {'mean': round(gm,6), 'std': round(gs,6)}
        print(f"    {g}: mean={gm:.4f}  std={gs:.4f}")

# 稳定性判断：std < 0.1 视为稳定
is_stable = std_corr < 0.10
print(f"\n  Stability verdict: {'STABLE' if is_stable else 'UNSTABLE'} "
      f"(threshold: std < 0.10)")

# Task1/Task2 准确率汇总
t1_accs = [r['task1_acc'] for r in all_results]
t2_accs = [r['task2_acc'] for r in all_results]
print(f"\n  Task1 (correctness) Acc: {np.mean(t1_accs):.4f} ± {np.std(t1_accs):.4f}")
print(f"  Task2 (option, {len(np.unique(y_option))}-class) Acc: "
      f"{np.mean(t2_accs):.4f} ± {np.std(t2_accs):.4f}")

# ════════════════════════════════════════════════════════════════
# 4. 保存
# ════════════════════════════════════════════════════════════════
final = {
    'scope_note': (
        "Task1-Task2 correlation analysis is NeurIPS-specific because "
        "ASSISTments lacks multiple-choice option labels (AnswerValue). "
        "Cross-platform stability for RO3 is therefore quantified via FISS "
        "(computed in Step 3 / run_all_seeds.py), which measures the "
        "stability of feature-importance patterns across NeurIPS and "
        "ASSISTments."
    ),
    'config': {'seeds': SEEDS, 'sample_size': SAMPLE_SIZE,
               'features': UNIFIED_FEATURES,
               'num_option_classes': int(len(np.unique(y_option)))},
    'per_seed_results': all_results,
    'aggregated': {
        'task1_task2_correlation': {
            'mean': round(mean_corr, 6),
            'std':  round(std_corr, 6),
            'ci95': round(ci95, 6),
        },
        'group_stability': group_stability,
        'task1_accuracy': {'mean': round(float(np.mean(t1_accs)),6),
                            'std':  round(float(np.std(t1_accs)),6)},
        'task2_accuracy': {'mean': round(float(np.mean(t2_accs)),6),
                            'std':  round(float(np.std(t2_accs)),6)},
        'stability_verdict': 'STABLE' if is_stable else 'UNSTABLE',
        'stability_threshold': 0.10,
    }
}

out = os.path.join(OUTPUT_PATH, 'step4_ro3_results.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(final, f, indent=2, ensure_ascii=False)
print(f"\n[SAVED] {out}")
print("\n>>> Step 4 (RO3) COMPLETE <<<")
