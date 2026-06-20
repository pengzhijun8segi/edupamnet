"""
step7_eci_redundancy_test.py
对照实验：特征冗余 vs ECI方差

假设：student_ability 与 historical_accuracy 在NeurIPS中完全冗余(r≈1.0)，
      这一冗余导致SHAP/LIME对这两个特征的归因分配不稳定，
      从而放大ECI（Spearman(SHAP,LIME)）在跨种子下的方差。

实验设计：
  条件A（5维，原始）: [student_ability, question_difficulty,
                        historical_accuracy, streak_correct, streak_incorrect]
  条件B（4维，去冗余）: 去掉 historical_accuracy
                        [student_ability, question_difficulty,
                         streak_correct, streak_incorrect]

对两条件分别跑5种子，比较ECI的 mean/std：
  若条件B的ECI std显著小于条件A → 支持"特征冗余放大ECI方差"假设

输出: results/step7_eci_redundancy_results.json
"""

import os, sys, json, warnings
import numpy as np
import torch
import torch.nn as nn
warnings.filterwarnings('ignore')

DATA_PATH   = './data/'
OUTPUT_PATH = './results/'
SEEDS       = [42, 123, 7, 2026, 999]
SAMPLE_SIZE = 30000
EPOCHS      = 60
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
SHAP_SAMPLES = 200
LIME_SAMPLES = 50

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 65)
print("Step 7 — ECI vs Feature Redundancy: Controlled Experiment")
print(f"  Seeds: {SEEDS}")
print("=" * 65)

from data_loader02 import CrossPlatformDataLoader
from data_loader_aligned import align_features, NEURIPS_MAP
from optimized_pamnet_implementation import OptimizedPAMNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset
import shap
import lime
import lime.lime_tabular

# ════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma, self.pos_weight = gamma, pos_weight
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none')
        pt = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()

def train_pamnet(model, train_loader, val_loader, platform,
                 num_epochs=60, device='cpu', patience=10):
    model.to(device)
    all_y = np.concatenate([y.numpy() for _, y in train_loader])
    pos, neg = float((all_y==1).sum()), float((all_y==0).sum())
    pw = torch.clamp(torch.tensor([neg/pos], dtype=torch.float32), 0.5, 2.0).to(device)
    crit = FocalLoss(gamma=2.0, pos_weight=pw)
    opt  = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs, eta_min=1e-5)
    best_f1, best_state, no_imp = -1, None, 0

    for epoch in range(num_epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            out = model(X_b, platform=platform)['output'].squeeze()
            loss = crit(out, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X_v, y_v in val_loader:
                out = model(X_v.to(device), platform=platform)['output']
                prob = torch.sigmoid(out).cpu().numpy().flatten()
                preds.extend((prob>0.5).astype(int).tolist())
                trues.extend(y_v.numpy().astype(int).tolist())
        val_f1 = f1_score(trues, preds, zero_division=0)
        if val_f1 > best_f1:
            best_f1, best_state, no_imp = val_f1, {k:v.clone() for k,v in model.state_dict().items()}, 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break
    model.load_state_dict(best_state)
    return model

def make_predict_fn(model, platform, device):
    def predict_proba(X):
        X_t = torch.tensor(X.astype(np.float32)).to(device)
        with torch.no_grad():
            out  = model(X_t, platform=platform)['output']
            prob = torch.sigmoid(out).cpu().numpy().flatten()
        return np.column_stack([1 - prob, prob])
    return predict_proba

def run_shap(rf_model, X_test, n_samples=200):
    te = X_test[:n_samples]
    explainer   = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(te)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    if sv.ndim > 2:
        sv = sv[:, :, 0]
    return np.abs(sv).mean(axis=0).flatten()

def run_lime(X_train, X_test, predict_fn, feature_names, n_samples=50, seed=42):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train, feature_names=feature_names,
        class_names=['incorrect', 'correct'],
        mode='classification', random_state=seed)
    imp = np.zeros(len(feature_names))
    n_ok = 0
    for i in range(min(n_samples, len(X_test))):
        try:
            exp  = explainer.explain_instance(
                X_test[i], predict_fn, num_features=len(feature_names))
            vals = dict(exp.as_list())
            for j, feat in enumerate(feature_names):
                for k, v in vals.items():
                    if feat.lower() in k.lower():
                        imp[j] += abs(v); break
            n_ok += 1
        except:
            continue
    if n_ok > 0:
        imp /= n_ok
    return imp

def stratified_sample(X, y, n, seed):
    if len(X) <= n:
        return X.astype(np.float32), y.astype(np.float32)
    _, Xs, _, ys = train_test_split(X, y, test_size=n/len(X),
                                     stratify=y.astype(int), random_state=seed)
    return Xs.astype(np.float32), ys.astype(np.float32)

def dedup(X, y):
    combined = np.hstack([X, y.reshape(-1,1)])
    _, idx = np.unique(combined.round(6), axis=0, return_index=True)
    return X[idx], y[idx]

def make_loaders(X, y, batch=256, val=0.2, seed=42):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=val, stratify=y.astype(int), random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=val, stratify=y_tr.astype(int), random_state=seed)
    return (DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                       batch_size=batch, shuffle=True),
            DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                       batch_size=batch),
            X_tr, y_tr, X_te, y_te)

# ════════════════════════════════════════════════════════════════
# 数据加载（NeurIPS only — ECI不稳定的问题出现在NeurIPS）
# ════════════════════════════════════════════════════════════════
print("\n[1] Loading NeurIPS data...")
loader = CrossPlatformDataLoader(data_path=DATA_PATH)
X_neu_raw, y_neu_raw, _, feat_neu = loader.load_neurips_dataset(task='correctness')
X_neu_al = align_features(X_neu_raw, feat_neu, NEURIPS_MAP)  # 5维，原顺序：
# [student_ability, question_difficulty, historical_accuracy, streak_correct, streak_incorrect]
X_neu_dd, y_neu_dd = dedup(X_neu_al, y_neu_raw.astype(np.float32))
print(f"  NeurIPS dedup: {X_neu_dd.shape}")

FEATURES_5D = ['student_ability', 'question_difficulty', 'historical_accuracy',
               'streak_correct', 'streak_incorrect']
FEATURES_4D = ['student_ability', 'question_difficulty',
               'streak_correct', 'streak_incorrect']  # 去掉 historical_accuracy（与student_ability冗余）

# 验证冗余
corr_check = np.corrcoef(X_neu_dd[:,0], X_neu_dd[:,2])[0,1]
print(f"  Verify redundancy: corr(student_ability, historical_accuracy) = {corr_check:.6f}")

# ════════════════════════════════════════════════════════════════
# 主循环：两个条件 × 5种子
# ════════════════════════════════════════════════════════════════
def run_condition(input_dim, feature_indices, feature_names, condition_label):
    print(f"\n{'#'*65}")
    print(f"CONDITION: {condition_label}  (dim={input_dim})")
    print(f"  Features: {feature_names}")
    print(f"{'#'*65}")

    results = []
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n  -- Seed {seed} ({seed_idx+1}/{len(SEEDS)}) --")
        set_seed(seed)

        X_full, y = stratified_sample(X_neu_dd, y_neu_dd, SAMPLE_SIZE, seed)
        X = X_full[:, feature_indices]
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)

        tr_l, val_l, X_tr, y_tr, X_te, y_te = make_loaders(X, y, seed=seed)
        model = OptimizedPAMNet(input_dim=input_dim)
        model = train_pamnet(model, tr_l, val_l, 'neurips', EPOCHS, DEVICE)

        predict_fn = make_predict_fn(model, 'neurips', DEVICE)

        # 代理RF
        soft_labels = (predict_fn(X_tr)[:, 1] > 0.5).astype(int)
        rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        rf.fit(X_tr, soft_labels)

        shap_imp = run_shap(rf, X_te, SHAP_SAMPLES)
        lime_imp = run_lime(X_tr, X_te, predict_fn, feature_names, LIME_SAMPLES, seed)

        eci, eci_p = spearmanr(shap_imp, lime_imp)
        eci = float(eci) if not np.isnan(eci) else 0.0

        print(f"    ECI = {eci:.4f}  (p={eci_p:.4f})")
        print(f"    SHAP: " + " | ".join(f"{n}={v:.4f}" for n,v in zip(feature_names, shap_imp)))
        print(f"    LIME: " + " | ".join(f"{n}={v:.4f}" for n,v in zip(feature_names, lime_imp)))

        results.append({
            'seed': seed,
            'eci': round(eci, 6),
            'eci_p': round(float(eci_p), 6),
            'shap_importance': shap_imp.tolist(),
            'lime_importance': lime_imp.tolist(),
        })
    return results

# 特征索引（在5维对齐特征中的位置）
idx_5d = [0, 1, 2, 3, 4]   # 全部
idx_4d = [0, 1, 3, 4]      # 去掉 index=2 (historical_accuracy)

results_5d = run_condition(5, idx_5d, FEATURES_5D, "A: 5D (with redundancy)")
results_4d = run_condition(4, idx_4d, FEATURES_4D, "B: 4D (redundancy removed)")

# ════════════════════════════════════════════════════════════════
# 对比分析
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("COMPARISON: ECI variance with vs without feature redundancy")
print(f"{'='*65}")

eci_5d = [r['eci'] for r in results_5d]
eci_4d = [r['eci'] for r in results_4d]

mean_5d, std_5d = float(np.mean(eci_5d)), float(np.std(eci_5d))
mean_4d, std_4d = float(np.mean(eci_4d)), float(np.std(eci_4d))

print(f"\n  Condition A (5D, with redundancy):")
print(f"    ECI values: {[round(v,4) for v in eci_5d]}")
print(f"    Mean = {mean_5d:.4f}  Std = {std_5d:.4f}")

print(f"\n  Condition B (4D, redundancy removed):")
print(f"    ECI values: {[round(v,4) for v in eci_4d]}")
print(f"    Mean = {mean_4d:.4f}  Std = {std_4d:.4f}")

variance_ratio = std_5d / std_4d if std_4d > 0 else float('inf')
print(f"\n  Std ratio (5D/4D) = {variance_ratio:.2f}x")

hypothesis_supported = std_4d < std_5d
print(f"\n  Hypothesis (redundancy amplifies ECI variance): "
      f"{'SUPPORTED' if hypothesis_supported else 'NOT SUPPORTED'}")

# F-test for variance difference
from scipy.stats import f as f_dist
if std_4d > 0 and std_5d > 0:
    F_stat = (std_5d**2) / (std_4d**2)
    df1, df2 = len(eci_5d)-1, len(eci_4d)-1
    p_value = 1 - f_dist.cdf(F_stat, df1, df2)
    print(f"  F-test: F={F_stat:.4f}  p={p_value:.4f} "
          f"({'significant' if p_value < 0.05 else 'not significant'} at α=0.05)")
else:
    F_stat, p_value = None, None

# ════════════════════════════════════════════════════════════════
# 保存
# ════════════════════════════════════════════════════════════════
final = {
    'hypothesis': (
        "Feature redundancy (student_ability ≈ historical_accuracy, "
        f"r={corr_check:.6f}) amplifies the cross-seed variance of ECI "
        "(Spearman correlation between SHAP and LIME importance)."
    ),
    'config': {'seeds': SEEDS, 'sample_size': SAMPLE_SIZE, 'epochs': EPOCHS,
               'redundancy_correlation': round(float(corr_check), 6)},
    'condition_A_5D': {
        'features': FEATURES_5D,
        'per_seed': results_5d,
        'eci_mean': round(mean_5d, 6),
        'eci_std':  round(std_5d, 6),
    },
    'condition_B_4D': {
        'features': FEATURES_4D,
        'per_seed': results_4d,
        'eci_mean': round(mean_4d, 6),
        'eci_std':  round(std_4d, 6),
    },
    'comparison': {
        'std_ratio_5d_over_4d': round(float(variance_ratio), 6),
        'hypothesis_supported': hypothesis_supported,
        'f_test_statistic': round(float(F_stat), 6) if F_stat else None,
        'f_test_p_value':   round(float(p_value), 6) if p_value else None,
    }
}

out = os.path.join(OUTPUT_PATH, 'step7_eci_redundancy_results.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(final, f, indent=2, ensure_ascii=False)
print(f"\n[SAVED] {out}")
print("\n>>> Step 7 (ECI redundancy controlled experiment) COMPLETE <<<")
