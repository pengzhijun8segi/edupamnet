"""
step9_supplementary_analyses.py
Step 9 — Supplementary Analyses for §1.4 Revised Criteria

本脚本分三个独立部分（Part A/B/C），对应§1.4重写后新增的评估要求：

  Part A (零成本，秒级):
    - RO1(1): EduPAMNet vs XGBoost 配对t检验
              (within-platform ceiling comparison)
    - RO3(3): MTL F1 gain 是否显著偏离0（单样本t检验）
    - RO2(1): FISS vs random-permutation null（精确置换检验）
    - RO2(2): ECI vs random-permutation null（精确置换检验）
    需要: results/final_results_all_seeds.json (run_all_seeds.py 的输出)

  Part B (轻量, ~20-30分钟):
    - RO1(2)部分: Naïve no-adaptation transfer baseline
      用RF在源平台训练，直接(无适配)应用到目标平台测试集，
      测量退化幅度，与PTS的退化幅度(1-PTS)对比

  Part C (重量级, ~2-3小时):
    - RO4(1)(2): 对 Platform-Aware 和 Platform-Agnostic 两种架构
      （复用Step8的两个模型）补做 Step6式 fairness + robustness 评估，
      比较两架构在disparity和robustness上是否有显著差异

输出: results/step9_supplementary_results.json
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
ROBUST_N    = 2000

os.makedirs(OUTPUT_PATH, exist_ok=True)

from scipy import stats
from itertools import permutations

results = {}

# ════════════════════════════════════════════════════════════════
# PART A — 零成本统计检验（配对t检验 + 置换检验）
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("PART A — Statistical Tests on Existing 5-Seed Results")
print("=" * 65)

all_seeds_path = os.path.join(OUTPUT_PATH, 'final_results_all_seeds.json')

if not os.path.exists(all_seeds_path):
    print(f"\n[ERROR] {all_seeds_path} not found.")
    print("  Part A requires the output of run_all_seeds.py.")
    print("  Please ensure 'final_results_all_seeds.json' is in ./results/")
    print("  Skipping Part A — proceeding to Part B and C.\n")
    results['part_A'] = {'status': 'SKIPPED', 'reason': 'final_results_all_seeds.json not found'}
else:
    with open(all_seeds_path, 'r', encoding='utf-8') as f:
        all_seeds_data = json.load(f)

    per_seed = all_seeds_data['per_seed_results']
    n_seeds  = len(per_seed)
    print(f"\nLoaded {n_seeds} seeds from {all_seeds_path}")

    # ── A1: RO1(1) EduPAMNet vs XGBoost paired t-test ──────────
    print("\n[A1] RO1(1): EduPAMNet vs XGBoost (within-platform ceiling)")
    edu_f1, xgb_f1 = [], []
    for r in per_seed:
        for plat, xgb_key in [('neurips', 'XGB_neurips'), ('assistments', 'XGB_assistments')]:
            edu_f1.append(r['edupamnet'][plat]['f1'])
            xgb_f1.append(r['baselines'][xgb_key]['f1'])

    t_a1, p_a1 = stats.ttest_rel(edu_f1, xgb_f1)
    diff_a1 = np.array(edu_f1) - np.array(xgb_f1)
    cohens_d_a1 = float(np.mean(diff_a1) / np.std(diff_a1, ddof=1))
    print(f"  EduPAMNet F1 mean = {np.mean(edu_f1):.4f}")
    print(f"  XGBoost   F1 mean = {np.mean(xgb_f1):.4f}")
    print(f"  Mean diff (EduPAMNet - XGB) = {np.mean(diff_a1):+.4f}")
    print(f"  Paired t-test: t = {t_a1:.4f}, p = {p_a1:.6f}, Cohen's d = {cohens_d_a1:.3f}")
    print(f"  Verdict: {'EduPAMNet significantly DIFFERENT from XGB' if p_a1 < 0.05 else 'NOT significantly different'}"
          f" ({'EduPAMNet BELOW ceiling' if np.mean(diff_a1) < 0 else 'EduPAMNet AT/ABOVE ceiling'})")

    results_a1 = {
        'edupamnet_f1_mean': round(float(np.mean(edu_f1)), 6),
        'xgb_f1_mean':       round(float(np.mean(xgb_f1)), 6),
        'mean_diff':         round(float(np.mean(diff_a1)), 6),
        't_statistic':       round(float(t_a1), 6),
        'p_value':           round(float(p_a1), 6),
        'cohens_d':          round(cohens_d_a1, 6),
        'n_pairs':           len(edu_f1),
        'verdict': ('EduPAMNet significantly below within-platform ceiling (XGBoost)'
                     if (p_a1 < 0.05 and np.mean(diff_a1) < 0) else
                     'EduPAMNet significantly above ceiling' if (p_a1 < 0.05) else
                     'No significant difference from ceiling')
    }

    # ── A2: RO3(3) MTL F1 gain — one-sample t-test vs 0 ────────
    print("\n[A2] RO3(3): MTL F1 gain — one-sample t-test (H0: gain = 0)")
    gains = []
    for r in per_seed:
        gains.append(r['f1_gain']['neurips'])
        gains.append(r['f1_gain']['assistments'])

    t_a2, p_a2 = stats.ttest_1samp(gains, 0.0)
    cohens_d_a2 = float(np.mean(gains) / np.std(gains, ddof=1))
    print(f"  MTL gain mean = {np.mean(gains):+.4f}  std = {np.std(gains, ddof=1):.4f}")
    print(f"  One-sample t-test vs 0: t = {t_a2:.4f}, p = {p_a2:.6f}, Cohen's d = {cohens_d_a2:.3f}")
    direction = "NEGATIVE (negative transfer)" if np.mean(gains) < 0 else "POSITIVE (positive transfer)"
    print(f"  Verdict: {'Significant ' + direction if p_a2 < 0.05 else 'Not significantly different from 0'}")

    results_a2 = {
        'mean_gain':   round(float(np.mean(gains)), 6),
        'std_gain':    round(float(np.std(gains, ddof=1)), 6),
        't_statistic': round(float(t_a2), 6),
        'p_value':     round(float(p_a2), 6),
        'cohens_d':    round(cohens_d_a2, 6),
        'n':           len(gains),
        'verdict': (f'Statistically significant {direction.lower()}' if p_a2 < 0.05
                    else 'Gain not significantly different from zero')
    }

    # ── A3/A4: FISS / ECI permutation tests (exact, 5 features => 120 perms) ──
    print("\n[A3] RO2(1): FISS vs random-permutation null (exact test, 5! = 120 perms)")

    def exact_perm_pvalue(vec_a, vec_b):
        """Exact permutation test for Spearman correlation with n=5 (5!=120 perms)."""
        n = len(vec_a)
        observed, _ = stats.spearmanr(vec_a, vec_b)
        null_corrs = []
        for perm in permutations(range(n)):
            perm_b = np.array(vec_b)[list(perm)]
            r, _ = stats.spearmanr(vec_a, perm_b)
            null_corrs.append(r)
        null_corrs = np.array(null_corrs)
        # two-sided p-value: proportion of |null| >= |observed|
        p = float(np.mean(np.abs(null_corrs) >= abs(observed) - 1e-12))
        return float(observed), p, float(np.mean(null_corrs)), float(np.std(null_corrs))

    fiss_pvals, fiss_obs = [], []
    for r in per_seed:
        shap_n = r['shap']['neurips']
        shap_a = r['shap']['assistments']
        obs, p, null_mean, null_std = exact_perm_pvalue(shap_n, shap_a)
        fiss_obs.append(obs)
        fiss_pvals.append(p)
        print(f"  Seed {r['seed']:5d}: FISS={obs:.4f}  null_mean={null_mean:.4f}±{null_std:.4f}  p={p:.4f}")

    n_sig_fiss = sum(p < 0.05 for p in fiss_pvals)
    print(f"  --> {n_sig_fiss}/{len(fiss_pvals)} seeds significant at α=0.05")
    print(f"  Mean FISS = {np.mean(fiss_obs):.4f}, mean p = {np.mean(fiss_pvals):.4f}")

    results_a3 = {
        'per_seed': [{'seed': r['seed'], 'fiss': round(o,6), 'p_value': round(p,6)}
                     for r, o, p in zip(per_seed, fiss_obs, fiss_pvals)],
        'mean_fiss': round(float(np.mean(fiss_obs)), 6),
        'mean_p':    round(float(np.mean(fiss_pvals)), 6),
        'n_significant': n_sig_fiss,
        'n_seeds': len(fiss_pvals),
        'verdict': (f'FISS significantly exceeds permutation null in {n_sig_fiss}/{len(fiss_pvals)} seeds')
    }

    print("\n[A4] RO2(2): ECI vs random-permutation null (exact test, per platform)")
    eci_results = {}
    for plat in ['neurips', 'assistments']:
        pvals, obs_list = [], []
        for r in per_seed:
            shap_v = r['shap'][plat]
            lime_v = r['lime'][plat]
            obs, p, null_mean, null_std = exact_perm_pvalue(shap_v, lime_v)
            obs_list.append(obs)
            pvals.append(p)
        n_sig = sum(p < 0.05 for p in pvals)
        print(f"  {plat:12s}: mean ECI={np.mean(obs_list):.4f}  mean p={np.mean(pvals):.4f}  "
              f"{n_sig}/{len(pvals)} seeds significant")
        eci_results[plat] = {
            'per_seed': [{'seed': r['seed'], 'eci': round(o,6), 'p_value': round(p,6)}
                         for r, o, p in zip(per_seed, obs_list, pvals)],
            'mean_eci': round(float(np.mean(obs_list)), 6),
            'mean_p':   round(float(np.mean(pvals)), 6),
            'n_significant': n_sig,
            'n_seeds': len(pvals),
        }

    results['part_A'] = {
        'status': 'COMPLETE',
        'RO1_1_aware_vs_ceiling': results_a1,
        'RO3_3_mtl_gain_significance': results_a2,
        'RO2_1_FISS_permutation_test': results_a3,
        'RO2_2_ECI_permutation_test': eci_results,
    }

print("\n[SAVED interim] Part A complete.")

# ════════════════════════════════════════════════════════════════
# COMMON UTILITIES (Parts B & C)
# ════════════════════════════════════════════════════════════════
from data_loader02 import CrossPlatformDataLoader
from data_loader_aligned import align_features, NEURIPS_MAP, ASSISTMENTS_MAP, UNIFIED_FEATURES, INPUT_DIM
from optimized_pamnet_implementation import OptimizedPAMNet, ImprovedSharedEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def dedup(X, y):
    combined = np.hstack([X, y.reshape(-1,1)])
    _, idx = np.unique(combined.round(6), axis=0, return_index=True)
    return X[idx], y[idx]

def stratified_sample(X, y, n, seed):
    if len(X) <= n:
        return X.astype(np.float32), y.astype(np.float32)
    _, Xs, _, ys = train_test_split(X, y, test_size=n/len(X),
                                     stratify=y.astype(int), random_state=seed)
    return Xs.astype(np.float32), ys.astype(np.float32)

print("\n[1] Loading shared data (for Parts B & C)...")
loader = CrossPlatformDataLoader(data_path=DATA_PATH)
X_neu_raw, y_neu_raw, _, feat_neu = loader.load_neurips_dataset(task='correctness')
X_ast_raw, y_ast_raw, _, feat_ast = loader.load_assistments_dataset(task='correctness')
X_neu_al = align_features(X_neu_raw, feat_neu, NEURIPS_MAP)
X_ast_al = align_features(X_ast_raw, feat_ast, ASSISTMENTS_MAP)
X_neu_dd, y_neu_dd = dedup(X_neu_al, y_neu_raw.astype(np.float32))
X_ast_dd, y_ast_dd = dedup(X_ast_al, y_ast_raw.astype(np.float32))
print(f"  NeurIPS dedup: {X_neu_dd.shape}  ASSISTments dedup: {X_ast_dd.shape}")

# ════════════════════════════════════════════════════════════════
# PART B — Naive No-Adaptation Transfer Baseline (RO1.2)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART B — Naive No-Adaptation Transfer Baseline (RO1.2)")
print("=" * 65)
print("Train RF on source platform, apply DIRECTLY (no adaptation) to")
print("target platform's test set. Measures naive-transfer degradation,")
print("for comparison against PTS-based representation-transfer degradation.")

naive_results = []
for seed in SEEDS:
    print(f"\n  Seed {seed}...")
    set_seed(seed)
    X_neu, y_neu = stratified_sample(X_neu_dd, y_neu_dd, SAMPLE_SIZE, seed)
    X_ast, y_ast = stratified_sample(X_ast_dd, y_ast_dd, SAMPLE_SIZE, seed)
    sc_n, sc_a = StandardScaler(), StandardScaler()
    X_neu_sc = sc_n.fit_transform(X_neu).astype(np.float32)
    X_ast_sc = sc_a.fit_transform(X_ast).astype(np.float32)

    # Split each platform into train/test
    Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(
        X_neu_sc, y_neu, test_size=0.2, stratify=y_neu.astype(int), random_state=seed)
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
        X_ast_sc, y_ast, test_size=0.2, stratify=y_ast.astype(int), random_state=seed)

    # Native models (within-platform ceiling, re-derived here for paired comparison)
    rf_neu = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf_neu.fit(Xn_tr, yn_tr.astype(int))
    rf_ast = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf_ast.fit(Xa_tr, ya_tr.astype(int))

    native_neu_f1 = f1_score(yn_te.astype(int), rf_neu.predict(Xn_te), zero_division=0)
    native_ast_f1 = f1_score(ya_te.astype(int), rf_ast.predict(Xa_te), zero_division=0)

    # Naive transfer: apply rf_neu (trained on NeurIPS, in NeurIPS-scaler space)
    # directly to ASSISTments test data — re-scale ASSISTments features using
    # NeurIPS's scaler to simulate "no adaptation" (same nominal feature space,
    # no target-domain adaptation of the input distribution).
    Xa_te_neuscale = sc_n.transform(sc_a.inverse_transform(Xa_te)).astype(np.float32)
    Xn_te_astscale = sc_a.transform(sc_n.inverse_transform(Xn_te)).astype(np.float32)

    transfer_neu2ast_f1 = f1_score(ya_te.astype(int), rf_neu.predict(Xa_te_neuscale), zero_division=0)
    transfer_ast2neu_f1 = f1_score(yn_te.astype(int), rf_ast.predict(Xn_te_astscale), zero_division=0)

    drop_neu2ast = native_ast_f1 - transfer_neu2ast_f1
    drop_ast2neu = native_neu_f1 - transfer_ast2neu_f1

    print(f"    Native   NeurIPS F1={native_neu_f1:.4f}  ASSISTments F1={native_ast_f1:.4f}")
    print(f"    Naive transfer N->A F1={transfer_neu2ast_f1:.4f}  drop={drop_neu2ast:+.4f}")
    print(f"    Naive transfer A->N F1={transfer_ast2neu_f1:.4f}  drop={drop_ast2neu:+.4f}")

    naive_results.append({
        'seed': seed,
        'native_neurips_f1':     round(float(native_neu_f1), 6),
        'native_assistments_f1': round(float(native_ast_f1), 6),
        'naive_transfer_n2a_f1': round(float(transfer_neu2ast_f1), 6),
        'naive_transfer_a2n_f1': round(float(transfer_ast2neu_f1), 6),
        'drop_n2a': round(float(drop_neu2ast), 6),
        'drop_a2n': round(float(drop_ast2neu), 6),
    })

drops_n2a = [r['drop_n2a'] for r in naive_results]
drops_a2n = [r['drop_a2n'] for r in naive_results]
naive_drop_overall = (np.mean(drops_n2a) + np.mean(drops_a2n)) / 2

print(f"\n  Naive-transfer F1 drop (mean±std):")
print(f"    N->A: {np.mean(drops_n2a):.4f} ± {np.std(drops_n2a):.4f}")
print(f"    A->N: {np.mean(drops_a2n):.4f} ± {np.std(drops_a2n):.4f}")
print(f"    Overall: {naive_drop_overall:.4f}")
print(f"\n  Compare to PTS-based representation-transfer 'drop' = 1 - PTS = {1-0.9922:.4f}")
print(f"  (PTS=0.9922 from Step 5)")

# significance test: is naive-transfer drop significantly larger than PTS-based drop (1-PTS per seed)?
pts_drops_n2a = [1 - 0.9993] * 5  # placeholder; replaced below if step5 file available
pts_drops_a2n = [1 - 0.9851] * 5

step5_path = os.path.join(OUTPUT_PATH, 'step5_pts_results.json')
if os.path.exists(step5_path):
    with open(step5_path,encoding='utf-8') as f:
        pts_data = json.load(f)
    pts_drops_n2a = [1 - r['pts_neu2ast'] for r in pts_data['per_seed_results']]
    pts_drops_a2n = [1 - r['pts_ast2neu'] for r in pts_data['per_seed_results']]

all_naive_drops = drops_n2a + drops_a2n
all_pts_drops   = pts_drops_n2a + pts_drops_a2n
t_b, p_b = stats.ttest_rel(all_naive_drops, all_pts_drops)
print(f"\n  Paired t-test (naive-transfer drop vs PTS-based drop): t={t_b:.4f}, p={p_b:.6f}")
print(f"  Verdict: {'Naive transfer degrades SIGNIFICANTLY MORE than representation-based transfer' if (p_b<0.05 and np.mean(all_naive_drops)>np.mean(all_pts_drops)) else 'No significant difference / unexpected direction'}")

results['part_B'] = {
    'description': (
        "Naive no-adaptation transfer: RF trained on source platform, "
        "applied directly to target platform test data (rescaled into "
        "source feature distribution, no target-domain adaptation)."
    ),
    'per_seed': naive_results,
    'naive_drop_n2a_mean': round(float(np.mean(drops_n2a)), 6),
    'naive_drop_n2a_std':  round(float(np.std(drops_n2a)), 6),
    'naive_drop_a2n_mean': round(float(np.mean(drops_a2n)), 6),
    'naive_drop_a2n_std':  round(float(np.std(drops_a2n)), 6),
    'pts_based_drop_n2a':  round(float(np.mean(pts_drops_n2a)), 6),
    'pts_based_drop_a2n':  round(float(np.mean(pts_drops_a2n)), 6),
    'paired_ttest_naive_vs_pts': {'t_statistic': round(float(t_b),6), 'p_value': round(float(p_b),6)},
}

# ════════════════════════════════════════════════════════════════
# PART C — Aware vs Agnostic: Fairness + Robustness Comparison (RO4.1/4.2)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART C — Aware vs Agnostic: Fairness + Robustness (RO4.1/4.2)")
print("=" * 65)
print("Retrains both architectures (as in Step 8) and applies Step-6-style")
print("fairness (DP/F1/EqOdds across ability tertiles) and robustness")
print("(Gaussian noise, FGSM) evaluation to BOTH, for direct comparison.")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma, self.pos_weight = gamma, pos_weight
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none')
        pt = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()

def make_pos_weight(y, device):
    pos, neg = float((y==1).sum()), float((y==0).sum())
    return torch.clamp(torch.tensor([neg/pos], dtype=torch.float32), 0.5, 2.0).to(device)

def make_loaders(X, y, batch=256, val=0.2, seed=42):
    from torch.utils.data import DataLoader, TensorDataset
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=val, stratify=y.astype(int), random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=val, stratify=y_tr.astype(int), random_state=seed)
    return (DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                       batch_size=batch, shuffle=True),
            DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                       batch_size=batch),
            X_te, y_te)

def train_aware(train_l, val_l, platform, epochs, device, patience=10):
    from torch.utils.data import DataLoader
    model = OptimizedPAMNet(input_dim=INPUT_DIM).to(device)
    all_y = np.concatenate([y.numpy() for _, y in train_l])
    pw = make_pos_weight(all_y, device)
    crit = FocalLoss(gamma=2.0, pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    best_f1, best_state, no_imp = -1, None, 0
    for epoch in range(epochs):
        model.train()
        for X_b, y_b in train_l:
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
            for X_v, y_v in val_l:
                out = model(X_v.to(device), platform=platform)['output']
                prob = torch.sigmoid(out).cpu().numpy().flatten()
                preds.extend((prob>0.5).astype(int).tolist())
                trues.extend(y_v.numpy().astype(int).tolist())
        val_f1 = f1_score(trues, preds, zero_division=0)
        if val_f1 > best_f1:
            best_f1, best_state, no_imp = val_f1, {k:v.clone() for k,v in model.state_dict().items()}, 0
        else:
            no_imp += 1
        if no_imp >= patience: break
    model.load_state_dict(best_state)
    model.eval()
    return model

class PlatformAgnosticMTL(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128,64,32], dropout_rate=0.3):
        super().__init__()
        self.shared_encoder = ImprovedSharedEncoder(input_dim, hidden_dims, dropout_rate)
        feat_dim = self.shared_encoder.output_dim
        self.heads = nn.ModuleDict({
            'neurips': nn.Linear(feat_dim, 1),
            'assistments': nn.Linear(feat_dim, 1),
        })
    def forward(self, x, platform):
        return self.heads[platform](self.shared_encoder(x))

def train_agnostic(neu_tr, neu_val, ast_tr, ast_val, epochs, device, patience=10):
    model = PlatformAgnosticMTL(INPUT_DIM).to(device)
    pw_n = make_pos_weight(np.concatenate([y.numpy() for _,y in neu_tr]), device)
    pw_a = make_pos_weight(np.concatenate([y.numpy() for _,y in ast_tr]), device)
    crit_n, crit_a = FocalLoss(2.0, pw_n), FocalLoss(2.0, pw_a)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    best_score, best_state, no_imp = -1, None, 0
    for epoch in range(epochs):
        model.train()
        for X_b,y_b in neu_tr:
            X_b,y_b=X_b.to(device),y_b.to(device)
            opt.zero_grad()
            loss = crit_n(model(X_b,'neurips').squeeze(), y_b)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        for X_b,y_b in ast_tr:
            X_b,y_b=X_b.to(device),y_b.to(device)
            opt.zero_grad()
            loss = crit_a(model(X_b,'assistments').squeeze(), y_b)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        sched.step()
        model.eval()
        f1s=[]
        for val_l, plat in [(neu_val,'neurips'),(ast_val,'assistments')]:
            preds,trues=[],[]
            with torch.no_grad():
                for X_v,y_v in val_l:
                    prob=torch.sigmoid(model(X_v.to(device),plat)).cpu().numpy().flatten()
                    preds.extend((prob>0.5).astype(int).tolist())
                    trues.extend(y_v.numpy().astype(int).tolist())
            f1s.append(f1_score(trues,preds,zero_division=0))
        score=sum(f1s)
        if score>best_score:
            best_score,best_state,no_imp=score,{k:v.clone() for k,v in model.state_dict().items()},0
        else: no_imp+=1
        if no_imp>=patience: break
    model.load_state_dict(best_state); model.eval()
    return model

def get_predictions(model, X, platform, is_agnostic=False):
    model.eval()
    with torch.no_grad():
        if is_agnostic:
            out = model(torch.tensor(X,dtype=torch.float32).to(DEVICE), platform)
        else:
            out = model(torch.tensor(X,dtype=torch.float32).to(DEVICE), platform=platform)['output']
        prob = torch.sigmoid(out).cpu().numpy().flatten()
    return (prob>0.5).astype(int), prob

def eval_robustness(model, X, y, platform, is_agnostic):
    pred0,_ = get_predictions(model,X,y,platform,is_agnostic) if False else get_predictions(model,X,platform,is_agnostic)
    base_acc = accuracy_score(y.astype(int), pred0)
    out = {'baseline_acc': float(base_acc)}
    for nl in [0.05, 0.1, 0.2]:
        accs=[]
        for _ in range(3):
            Xn = X + np.random.normal(0,nl,X.shape).astype(np.float32)
            pred,_ = get_predictions(model,Xn,platform,is_agnostic)
            accs.append(accuracy_score(y.astype(int),pred))
        out[f'noise_{nl}_drop'] = float(base_acc-np.mean(accs))
    # FGSM
    with torch.enable_grad():
        X_t = torch.tensor(X,dtype=torch.float32,device=DEVICE); X_t.requires_grad_(True)
        y_t = torch.tensor(y,dtype=torch.float32,device=DEVICE)
        if is_agnostic:
            out_t = model(X_t, platform).squeeze()
        else:
            out_t = model(X_t, platform=platform)['output'].squeeze()
        loss = nn.functional.binary_cross_entropy_with_logits(out_t, y_t)
        grads = torch.autograd.grad(loss, X_t, allow_unused=True)[0]
    if grads is not None:
        grad_sign = grads.detach().sign()
        for eps in [0.05, 0.1]:
            X_adv=(X_t.detach()+eps*grad_sign).cpu().numpy()
            pred_adv,_=get_predictions(model,X_adv,platform,is_agnostic)
            out[f'fgsm_{eps}_drop']=float(base_acc-accuracy_score(y.astype(int),pred_adv))
    else:
        for eps in [0.05,0.1]: out[f'fgsm_{eps}_drop']=None
    return out

def eval_fairness(model, X, y, platform, is_agnostic, ability_idx=0, n_groups=3):
    pred,_ = get_predictions(model,X,platform,is_agnostic)
    ability = X[:,ability_idx]
    bounds = np.percentile(ability, np.linspace(0,100,n_groups+1))
    pos_rates, f1s, tprs, fprs = [],[],[],[]
    for g in range(n_groups):
        mask=(ability>=bounds[g])&(ability<=bounds[g+1])
        if mask.sum()<10: continue
        yt,yp=y[mask].astype(int),pred[mask]
        tp=np.sum((yt==1)&(yp==1)); fn=np.sum((yt==1)&(yp==0))
        fp=np.sum((yt==0)&(yp==1)); tn=np.sum((yt==0)&(yp==0))
        pos_rates.append(yp.mean())
        f1s.append(f1_score(yt,yp,zero_division=0))
        tprs.append(tp/(tp+fn) if (tp+fn)>0 else 0)
        fprs.append(fp/(fp+tn) if (fp+tn)>0 else 0)
    return {
        'dp_diff': float(max(pos_rates)-min(pos_rates)),
        'f1_diff': float(max(f1s)-min(f1s)),
        'eq_odds_tpr_diff': float(max(tprs)-min(tprs)),
        'eq_odds_fpr_diff': float(max(fprs)-min(fprs)),
    }

part_c_results=[]
for seed in SEEDS:
    print(f"\n  Seed {seed}...")
    set_seed(seed)
    X_neu, y_neu = stratified_sample(X_neu_dd, y_neu_dd, SAMPLE_SIZE, seed)
    X_ast, y_ast = stratified_sample(X_ast_dd, y_ast_dd, SAMPLE_SIZE, seed)
    sc_n, sc_a = StandardScaler(), StandardScaler()
    X_neu = sc_n.fit_transform(X_neu).astype(np.float32)
    X_ast = sc_a.fit_transform(X_ast).astype(np.float32)

    neu_tr,neu_val,X_neu_te,y_neu_te = make_loaders(X_neu,y_neu,seed=seed)
    ast_tr,ast_val,X_ast_te,y_ast_te = make_loaders(X_ast,y_ast,seed=seed)

    print("    Training Aware...")
    set_seed(seed)
    model_aware_neu = train_aware(neu_tr,neu_val,'neurips',EPOCHS,DEVICE)
    set_seed(seed)
    model_aware_ast = train_aware(ast_tr,ast_val,'assistments',EPOCHS,DEVICE)

    print("    Training Agnostic...")
    set_seed(seed)
    model_agnostic = train_agnostic(neu_tr,neu_val,ast_tr,ast_val,EPOCHS,DEVICE)

    def sub(X,y,n,seed):
        if len(X)<=n: return X,y
        idx=np.random.RandomState(seed).choice(len(X),n,replace=False)
        return X[idx],y[idx]
    X_neu_r,y_neu_r = sub(X_neu_te,y_neu_te,ROBUST_N,seed)
    X_ast_r,y_ast_r = sub(X_ast_te,y_ast_te,ROBUST_N,seed)

    # Robustness
    rob_aware_neu = eval_robustness(model_aware_neu, X_neu_r, y_neu_r, 'neurips', False)
    rob_aware_ast = eval_robustness(model_aware_ast, X_ast_r, y_ast_r, 'assistments', False)
    rob_agno_neu  = eval_robustness(model_agnostic, X_neu_r, y_neu_r, 'neurips', True)
    rob_agno_ast  = eval_robustness(model_agnostic, X_ast_r, y_ast_r, 'assistments', True)

    # Fairness
    fair_aware_neu = eval_fairness(model_aware_neu, X_neu_te, y_neu_te, 'neurips', False)
    fair_aware_ast = eval_fairness(model_aware_ast, X_ast_te, y_ast_te, 'assistments', False)
    fair_agno_neu  = eval_fairness(model_agnostic, X_neu_te, y_neu_te, 'neurips', True)
    fair_agno_ast  = eval_fairness(model_agnostic, X_ast_te, y_ast_te, 'assistments', True)

    print(f"    DP_diff  Aware: N={fair_aware_neu['dp_diff']:.4f} A={fair_aware_ast['dp_diff']:.4f}")
    print(f"    DP_diff  Agno : N={fair_agno_neu['dp_diff']:.4f} A={fair_agno_ast['dp_diff']:.4f}")
    print(f"    noise0.2 drop Aware: N={rob_aware_neu['noise_0.2_drop']:.4f} A={rob_aware_ast['noise_0.2_drop']:.4f}")
    print(f"    noise0.2 drop Agno : N={rob_agno_neu['noise_0.2_drop']:.4f} A={rob_agno_ast['noise_0.2_drop']:.4f}")

    part_c_results.append({
        'seed': seed,
        'fairness': {
            'aware':    {'neurips': fair_aware_neu, 'assistments': fair_aware_ast},
            'agnostic': {'neurips': fair_agno_neu,  'assistments': fair_agno_ast},
        },
        'robustness': {
            'aware':    {'neurips': rob_aware_neu, 'assistments': rob_aware_ast},
            'agnostic': {'neurips': rob_agno_neu,  'assistments': rob_agno_ast},
        }
    })

# Aggregate Part C
def agg(vals):
    arr=np.array(vals,dtype=float)
    return {'mean':round(float(arr.mean()),6),'std':round(float(arr.std()),6)}

dp_aware = [r['fairness']['aware'][p]['dp_diff'] for r in part_c_results for p in ['neurips','assistments']]
dp_agno  = [r['fairness']['agnostic'][p]['dp_diff'] for r in part_c_results for p in ['neurips','assistments']]
noise_aware = [r['robustness']['aware'][p]['noise_0.2_drop'] for r in part_c_results for p in ['neurips','assistments']]
noise_agno  = [r['robustness']['agnostic'][p]['noise_0.2_drop'] for r in part_c_results for p in ['neurips','assistments']]

t_dp,p_dp = stats.ttest_rel(dp_aware, dp_agno)
t_noise,p_noise = stats.ttest_rel(noise_aware, noise_agno)

print(f"\n  Aggregated (n=10 = 5 seeds x 2 platforms):")
print(f"    DP_diff   Aware={agg(dp_aware)} vs Agnostic={agg(dp_agno)}")
print(f"      paired t-test: t={t_dp:.4f}, p={p_dp:.4f}")
print(f"    Noise(0.2) drop Aware={agg(noise_aware)} vs Agnostic={agg(noise_agno)}")
print(f"      paired t-test: t={t_noise:.4f}, p={p_noise:.4f}")

results['part_C'] = {
    'description': "Step-6-style fairness and robustness comparison: Platform-Aware vs Platform-Agnostic (Step 8 architectures), 5 seeds.",
    'per_seed': part_c_results,
    'aggregated': {
        'dp_diff':  {'aware': agg(dp_aware), 'agnostic': agg(dp_agno),
                     'paired_ttest': {'t': round(float(t_dp),6), 'p': round(float(p_dp),6)}},
        'noise_0.2_drop': {'aware': agg(noise_aware), 'agnostic': agg(noise_agno),
                            'paired_ttest': {'t': round(float(t_noise),6), 'p': round(float(p_noise),6)}},
    }
}

# ════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════
out = os.path.join(OUTPUT_PATH, 'step9_supplementary_results.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n[SAVED] {out}")
print("\n>>> Step 9 COMPLETE <<<")
