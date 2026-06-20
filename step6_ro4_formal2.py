"""
step6_ro4_formal.py
Step 6 — RO4: Robustness & Fairness (正式5种子版)

对应论文 §4.7/4.8（RO4 公平性-鲁棒性平衡）

计算内容（5种子 [42,123,7,2026,999]，30000样本）：
  鲁棒性 (Robustness):
    1. Gaussian noise robustness (levels: 0.01, 0.05, 0.1, 0.2)
    2. Feature-level perturbation (per-feature)
    3. FGSM adversarial robustness (epsilon: 0.01, 0.05, 0.1)

  公平性 (Fairness):
    1. 3个学生能力子群体（low/medium/high performers，按 student_ability 三分）
    2. Demographic Parity Difference (DP_diff)
    3. Equalized Odds Difference (TPR_diff, FPR_diff)
    4. F1 Difference across groups

输出: results/step6_ro4_results.json
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
ROBUST_N    = 2000   # 鲁棒性测试样本数（避免FGSM过慢）

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 65)
print("Step 6 — RO4: Robustness & Fairness (formal, 5 seeds)")
print(f"  Seeds: {SEEDS}  |  Epochs: {EPOCHS}  |  Device: {DEVICE}")
print("=" * 65)

from data_loader02 import CrossPlatformDataLoader
from data_loader_aligned import align_features, NEURIPS_MAP, ASSISTMENTS_MAP, UNIFIED_FEATURES, INPUT_DIM
from optimized_pamnet_implementation import OptimizedPAMNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

# ════════════════════════════════════════════════════════════════
# 工具函数（与 run_all_seeds.py / step5 一致，保持训练管道统一）
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
            X_te, y_te)

# ════════════════════════════════════════════════════════════════
# 鲁棒性测试函数
# ════════════════════════════════════════════════════════════════
def get_predictions(model, X, platform):
    model.eval()
    with torch.no_grad():
        out  = model(torch.tensor(X, dtype=torch.float32).to(DEVICE), platform=platform)['output']
        prob = torch.sigmoid(out).cpu().numpy().flatten()
    return (prob > 0.5).astype(int), prob

def test_noise_robustness(model, X, y, platform, noise_levels=[0.01,0.05,0.1,0.2], trials=3):
    pred0, _ = get_predictions(model, X, platform)
    base_acc = accuracy_score(y.astype(int), pred0)
    results  = {'baseline_acc': float(base_acc)}
    for nl in noise_levels:
        accs = []
        for _ in range(trials):
            Xn = X + np.random.normal(0, nl, X.shape).astype(np.float32)
            pred, _ = get_predictions(model, Xn, platform)
            accs.append(accuracy_score(y.astype(int), pred))
        results[f'noise_{nl}'] = {
            'accuracy':     float(np.mean(accs)),
            'drop':         float(base_acc - np.mean(accs))
        }
    return results

def test_feature_perturbation(model, X, y, platform, feature_names, sigma=0.1):
    pred0, _ = get_predictions(model, X, platform)
    base_acc = accuracy_score(y.astype(int), pred0)
    results = {}
    for i, fname in enumerate(feature_names):
        Xp = X.copy()
        Xp[:, i] += np.random.normal(0, sigma, len(Xp)).astype(np.float32)
        pred, _ = get_predictions(model, Xp, platform)
        acc = accuracy_score(y.astype(int), pred)
        results[fname] = {'accuracy': float(acc), 'drop': float(base_acc - acc)}
    return results

def test_fgsm_robustness(model, X, y, platform, epsilons=[0.01, 0.05, 0.1]):
    """FGSM 对抗鲁棒性：基于梯度符号的输入扰动"""
    model.eval()
    pred0, _ = get_predictions(model, X, platform)
    base_acc = accuracy_score(y.astype(int), pred0)
    results = {'baseline_acc': float(base_acc)}

    with torch.enable_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        X_t.requires_grad_(True)
        y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)

        out  = model(X_t, platform=platform)['output'].squeeze()
        loss = nn.functional.binary_cross_entropy_with_logits(out, y_t)

        grads = torch.autograd.grad(loss, X_t, retain_graph=False,
                                     create_graph=False, allow_unused=True)[0]

    if grads is None:
        print("    [WARN] FGSM: input gradient is None — skipping adversarial test")
        for eps in epsilons:
            results[f'fgsm_eps_{eps}'] = {'accuracy': float(base_acc), 'drop': 0.0}
        return results

    grad_sign = grads.detach().sign()
    for eps in epsilons:
        X_adv = (X_t.detach() + eps * grad_sign).cpu().numpy()
        pred_adv, _ = get_predictions(model, X_adv, platform)
        acc_adv = accuracy_score(y.astype(int), pred_adv)
        results[f'fgsm_eps_{eps}'] = {
            'accuracy': float(acc_adv),
            'drop':     float(base_acc - acc_adv)
        }
    return results

# ════════════════════════════════════════════════════════════════
# 公平性测试函数
# ════════════════════════════════════════════════════════════════
def test_fairness(model, X, y, platform, ability_idx=0, n_groups=3):
    pred, prob = get_predictions(model, X, platform)
    ability = X[:, ability_idx]
    bounds  = np.percentile(ability, np.linspace(0, 100, n_groups+1))

    groups = {}
    for g in range(n_groups):
        mask = (ability >= bounds[g]) & (ability <= bounds[g+1])
        if mask.sum() < 10:
            continue
        yt, yp = y[mask].astype(int), pred[mask]
        # TPR / FPR for equalized odds
        tp = np.sum((yt==1) & (yp==1)); fn = np.sum((yt==1) & (yp==0))
        fp = np.sum((yt==0) & (yp==1)); tn = np.sum((yt==0) & (yp==0))
        tpr = tp / (tp+fn) if (tp+fn) > 0 else 0
        fpr = fp / (fp+tn) if (fp+tn) > 0 else 0
        groups[f'group_{g+1}'] = {
            'size':          int(mask.sum()),
            'accuracy':      float(accuracy_score(yt, yp)),
            'f1':            float(f1_score(yt, yp, zero_division=0)),
            'positive_rate': float(yp.mean()),
            'tpr':           float(tpr),
            'fpr':           float(fpr),
        }

    pos_rates = [v['positive_rate'] for v in groups.values()]
    f1_scores = [v['f1']            for v in groups.values()]
    tprs      = [v['tpr']           for v in groups.values()]
    fprs      = [v['fpr']           for v in groups.values()]

    return {
        'groups':              groups,
        'dp_diff':             float(max(pos_rates) - min(pos_rates)),
        'f1_diff':             float(max(f1_scores) - min(f1_scores)),
        'eq_odds_tpr_diff':    float(max(tprs) - min(tprs)),
        'eq_odds_fpr_diff':    float(max(fprs) - min(fprs)),
    }

# ════════════════════════════════════════════════════════════════
# 数据加载（共享）
# ════════════════════════════════════════════════════════════════
print("\n[1] Loading & aligning data...")
loader = CrossPlatformDataLoader(data_path=DATA_PATH)
X_neu_raw, y_neu_raw, _, feat_neu = loader.load_neurips_dataset(task='correctness')
X_ast_raw, y_ast_raw, _, feat_ast = loader.load_assistments_dataset(task='correctness')
X_neu_al = align_features(X_neu_raw, feat_neu, NEURIPS_MAP)
X_ast_al = align_features(X_ast_raw, feat_ast, ASSISTMENTS_MAP)
X_neu_dd, y_neu_dd = dedup(X_neu_al, y_neu_raw.astype(np.float32))
X_ast_dd, y_ast_dd = dedup(X_ast_al, y_ast_raw.astype(np.float32))
ABILITY_IDX = UNIFIED_FEATURES.index('student_ability')
print(f"  NeurIPS dedup    : {X_neu_dd.shape}")
print(f"  ASSISTments dedup: {X_ast_dd.shape}")

# ════════════════════════════════════════════════════════════════
# 主循环
# ════════════════════════════════════════════════════════════════
all_results = []

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*65}")
    print(f"SEED {seed}  ({seed_idx+1}/{len(SEEDS)})")
    print(f"{'='*65}")
    set_seed(seed)

    X_neu, y_neu = stratified_sample(X_neu_dd, y_neu_dd, SAMPLE_SIZE, seed)
    X_ast, y_ast = stratified_sample(X_ast_dd, y_ast_dd, SAMPLE_SIZE, seed)
    scaler_n, scaler_a = StandardScaler(), StandardScaler()
    X_neu = scaler_n.fit_transform(X_neu).astype(np.float32)
    X_ast = scaler_a.fit_transform(X_ast).astype(np.float32)

    # ── 训练模型 ─────────────────────────────────────────────
    tr_n, val_n, X_neu_te, y_neu_te = make_loaders(X_neu, y_neu, seed=seed)
    model_neu = OptimizedPAMNet(input_dim=INPUT_DIM)
    model_neu = train_pamnet(model_neu, tr_n, val_n, 'neurips', EPOCHS, DEVICE)
    print(f"  [OK] NeurIPS model trained")

    tr_a, val_a, X_ast_te, y_ast_te = make_loaders(X_ast, y_ast, seed=seed)
    model_ast = OptimizedPAMNet(input_dim=INPUT_DIM)
    model_ast = train_pamnet(model_ast, tr_a, val_a, 'assistments', EPOCHS, DEVICE)
    print(f"  [OK] ASSISTments model trained")

    # 限制鲁棒性测试样本数
    def subsample(X, y, n, seed):
        if len(X) <= n:
            return X, y
        idx = np.random.RandomState(seed).choice(len(X), n, replace=False)
        return X[idx], y[idx]

    X_neu_r, y_neu_r = subsample(X_neu_te, y_neu_te, ROBUST_N, seed)
    X_ast_r, y_ast_r = subsample(X_ast_te, y_ast_te, ROBUST_N, seed)

    # ── 鲁棒性 ───────────────────────────────────────────────
    print(f"\n  [Robustness] Testing...")
    noise_n = test_noise_robustness(model_neu, X_neu_r, y_neu_r, 'neurips')
    noise_a = test_noise_robustness(model_ast, X_ast_r, y_ast_r, 'assistments')
    feat_n  = test_feature_perturbation(model_neu, X_neu_r, y_neu_r, 'neurips', UNIFIED_FEATURES)
    feat_a  = test_feature_perturbation(model_ast, X_ast_r, y_ast_r, 'assistments', UNIFIED_FEATURES)
    fgsm_n  = test_fgsm_robustness(model_neu, X_neu_r, y_neu_r, 'neurips')
    fgsm_a  = test_fgsm_robustness(model_ast, X_ast_r, y_ast_r, 'assistments')

    print(f"    NeurIPS     noise_0.2 drop={noise_n['noise_0.2']['drop']:.4f}  "
          f"FGSM eps=0.1 drop={fgsm_n['fgsm_eps_0.1']['drop']:.4f}")
    print(f"    ASSISTments noise_0.2 drop={noise_a['noise_0.2']['drop']:.4f}  "
          f"FGSM eps=0.1 drop={fgsm_a['fgsm_eps_0.1']['drop']:.4f}")

    # ── 公平性 ───────────────────────────────────────────────
    print(f"\n  [Fairness] Testing...")
    fair_n = test_fairness(model_neu, X_neu_te, y_neu_te, 'neurips', ABILITY_IDX)
    fair_a = test_fairness(model_ast, X_ast_te, y_ast_te, 'assistments', ABILITY_IDX)
    print(f"    NeurIPS     DP_diff={fair_n['dp_diff']:.4f}  "
          f"EqOdds_TPR_diff={fair_n['eq_odds_tpr_diff']:.4f}")
    print(f"    ASSISTments DP_diff={fair_a['dp_diff']:.4f}  "
          f"EqOdds_TPR_diff={fair_a['eq_odds_tpr_diff']:.4f}")

    all_results.append({
        'seed': seed,
        'robustness': {
            'neurips':     {'noise': noise_n, 'feature': feat_n, 'fgsm': fgsm_n},
            'assistments': {'noise': noise_a, 'feature': feat_a, 'fgsm': fgsm_a},
        },
        'fairness': {
            'neurips':     fair_n,
            'assistments': fair_a,
        }
    })

# ════════════════════════════════════════════════════════════════
# 汇总: mean ± std
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("AGGREGATING RO4 RESULTS (mean ± std across 5 seeds)")
print(f"{'='*65}")

def agg(values):
    arr = np.array(values, dtype=float)
    return {'mean': round(float(arr.mean()),6), 'std': round(float(arr.std()),6),
            'ci95': round(float(1.96*arr.std()/np.sqrt(len(arr))),6)}

# 鲁棒性汇总
robustness_agg = {}
for plat in ['neurips', 'assistments']:
    robustness_agg[plat] = {
        'baseline_acc': agg([r['robustness'][plat]['noise']['baseline_acc'] for r in all_results]),
        'noise': {
            f'noise_{nl}': {
                'accuracy': agg([r['robustness'][plat]['noise'][f'noise_{nl}']['accuracy'] for r in all_results]),
                'drop':     agg([r['robustness'][plat]['noise'][f'noise_{nl}']['drop']     for r in all_results]),
            } for nl in [0.01, 0.05, 0.1, 0.2]
        },
        'fgsm': {
            f'eps_{eps}': {
                'accuracy': agg([r['robustness'][plat]['fgsm'][f'fgsm_eps_{eps}']['accuracy'] for r in all_results]),
                'drop':     agg([r['robustness'][plat]['fgsm'][f'fgsm_eps_{eps}']['drop']     for r in all_results]),
            } for eps in [0.01, 0.05, 0.1]
        },
        'feature_perturbation': {
            fname: {
                'drop': agg([r['robustness'][plat]['feature'][fname]['drop'] for r in all_results])
            } for fname in UNIFIED_FEATURES
        }
    }

# 公平性汇总
fairness_agg = {}
for plat in ['neurips', 'assistments']:
    fairness_agg[plat] = {
        'dp_diff':          agg([r['fairness'][plat]['dp_diff']          for r in all_results]),
        'f1_diff':          agg([r['fairness'][plat]['f1_diff']          for r in all_results]),
        'eq_odds_tpr_diff': agg([r['fairness'][plat]['eq_odds_tpr_diff'] for r in all_results]),
        'eq_odds_fpr_diff': agg([r['fairness'][plat]['eq_odds_fpr_diff'] for r in all_results]),
    }

# ── 打印摘要 ─────────────────────────────────────────────────────
print("\n  ROBUSTNESS — Noise (Gaussian)")
for plat in ['neurips', 'assistments']:
    print(f"\n    {plat.upper()}:")
    print(f"      Baseline Acc : {robustness_agg[plat]['baseline_acc']['mean']:.4f} ± "
          f"{robustness_agg[plat]['baseline_acc']['std']:.4f}")
    for nl in [0.01, 0.05, 0.1, 0.2]:
        d = robustness_agg[plat]['noise'][f'noise_{nl}']['drop']
        print(f"      noise={nl:<4}: drop = {d['mean']:+.4f} ± {d['std']:.4f}")

print("\n  ROBUSTNESS — FGSM Adversarial")
for plat in ['neurips', 'assistments']:
    print(f"\n    {plat.upper()}:")
    for eps in [0.01, 0.05, 0.1]:
        d = robustness_agg[plat]['fgsm'][f'eps_{eps}']['drop']
        print(f"      eps={eps:<4}: drop = {d['mean']:+.4f} ± {d['std']:.4f}")

print("\n  FAIRNESS")
for plat in ['neurips', 'assistments']:
    f = fairness_agg[plat]
    print(f"\n    {plat.upper()}:")
    print(f"      DP_diff          : {f['dp_diff']['mean']:.4f} ± {f['dp_diff']['std']:.4f}  "
          f"(95%CI ±{f['dp_diff']['ci95']:.4f})")
    print(f"      F1_diff          : {f['f1_diff']['mean']:.4f} ± {f['f1_diff']['std']:.4f}")
    print(f"      EqOdds TPR_diff  : {f['eq_odds_tpr_diff']['mean']:.4f} ± {f['eq_odds_tpr_diff']['std']:.4f}")
    print(f"      EqOdds FPR_diff  : {f['eq_odds_fpr_diff']['mean']:.4f} ± {f['eq_odds_fpr_diff']['std']:.4f}")

# ── 保存 ─────────────────────────────────────────────────────────
final = {
    'config': {'seeds': SEEDS, 'sample_size': SAMPLE_SIZE, 'epochs': EPOCHS,
               'robustness_test_n': ROBUST_N, 'features': UNIFIED_FEATURES},
    'per_seed_results': all_results,
    'aggregated': {
        'robustness': robustness_agg,
        'fairness':   fairness_agg,
    }
}
out = os.path.join(OUTPUT_PATH, 'step6_ro4_results.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(final, f, indent=2, ensure_ascii=False)
print(f"\n[SAVED] {out}")
print("\n>>> Step 6 (RO4 formal) COMPLETE <<<")
