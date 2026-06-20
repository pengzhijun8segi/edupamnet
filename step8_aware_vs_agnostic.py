"""
step8_aware_vs_agnostic.py
Step 8 — Platform-Aware vs Platform-Agnostic Architecture Comparison

对应 §1.1 "Platform-Aware Modelling" 的核心声称：
  "platform-aware approaches can achieve X% performance improvement
   compared to generic (platform-agnostic) methods"

两种架构（使用同一 ImprovedSharedEncoder，确保公平对比）：

  A) Platform-Aware (EduPAMNet核心结构):
     - 每个平台一个独立模型
     - shared_encoder(5→32) + BalancedDecoder（平台特异，更大容量）
     - 各自在自己平台数据上单独训练

  B) Platform-Agnostic (MTLClassifier，对应代码库中 mtl_baseline_runner.py):
     - 单一模型，单一 shared_encoder(5→32)
     - 每平台一个简单线性头 (Linear+Sigmoid)，容量远小于BalancedDecoder
     - 两平台数据联合(pooled)训练，共享同一encoder的同一次更新

5种子 [42,123,7,2026,999]，30000样本，5维对齐特征，与run_all_seeds.py完全一致的
数据管道，确保可与已有RO1结果（platform-aware: F1=0.7186/0.6953）配对比较。

输出: results/step8_aware_vs_agnostic_results.json
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

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 65)
print("Step 8 — Platform-Aware vs Platform-Agnostic Comparison")
print(f"  Seeds: {SEEDS}  |  Epochs: {EPOCHS}  |  Device: {DEVICE}")
print("=" * 65)

from data_loader02 import CrossPlatformDataLoader
from data_loader_aligned import align_features, NEURIPS_MAP, ASSISTMENTS_MAP, UNIFIED_FEATURES, INPUT_DIM
from optimized_pamnet_implementation import OptimizedPAMNet, ImprovedSharedEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats

# ════════════════════════════════════════════════════════════════
# 工具函数（与 run_all_seeds.py 保持一致）
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

def compute_metrics(y_true, y_pred, y_prob=None):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1':       float(f1_score(y_true, y_pred, zero_division=0)),
        'auc':      float(roc_auc_score(y_true, y_prob)) if y_prob is not None else None
    }

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

def make_pos_weight(loader, device):
    all_y = np.concatenate([y.numpy() for _, y in loader])
    pos, neg = float((all_y==1).sum()), float((all_y==0).sum())
    return torch.clamp(torch.tensor([neg/pos], dtype=torch.float32), 0.5, 2.0).to(device)

def eval_loader(model_fn, X, y, device):
    """model_fn(X_tensor) -> logits"""
    with torch.no_grad():
        out  = model_fn(torch.tensor(X, dtype=torch.float32).to(device))
        prob = torch.sigmoid(out).cpu().numpy().flatten()
    pred = (prob > 0.5).astype(int)
    return compute_metrics(y.astype(int), pred, prob)

# ════════════════════════════════════════════════════════════════
# A) Platform-Aware: EduPAMNet (shared_encoder + BalancedDecoder, 各平台独立训练)
# ════════════════════════════════════════════════════════════════
def train_aware(X_tr_l, X_val_l, platform, epochs, device, patience=10):
    model = OptimizedPAMNet(input_dim=INPUT_DIM).to(device)
    pw    = make_pos_weight(X_tr_l, device)
    crit  = FocalLoss(gamma=2.0, pos_weight=pw)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    best_f1, best_state, no_imp = -1, None, 0

    for epoch in range(epochs):
        model.train()
        for X_b, y_b in X_tr_l:
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
            for X_v, y_v in X_val_l:
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
    model.eval()
    return model

# ════════════════════════════════════════════════════════════════
# B) Platform-Agnostic: MTLClassifier
#    单一 shared_encoder + 简单线性头, 两平台数据联合训练
# ════════════════════════════════════════════════════════════════
class PlatformAgnosticMTL(nn.Module):
    """对应 mtl_baseline_runner.py 的 MTLClassifier，复用 ImprovedSharedEncoder 以保证编码器容量一致"""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super().__init__()
        self.shared_encoder = ImprovedSharedEncoder(input_dim, hidden_dims, dropout_rate)
        feat_dim = self.shared_encoder.output_dim
        self.heads = nn.ModuleDict({
            'neurips':     nn.Linear(feat_dim, 1),
            'assistments': nn.Linear(feat_dim, 1),
        })

    def forward(self, x, platform):
        feat = self.shared_encoder(x)
        return self.heads[platform](feat)

def train_agnostic(neu_tr_l, neu_val_l, ast_tr_l, ast_val_l, epochs, device, patience=10):
    model = PlatformAgnosticMTL(INPUT_DIM).to(device)
    pw_n  = make_pos_weight(neu_tr_l, device)
    pw_a  = make_pos_weight(ast_tr_l, device)
    crit_n = FocalLoss(gamma=2.0, pos_weight=pw_n)
    crit_a = FocalLoss(gamma=2.0, pos_weight=pw_a)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    best_score, best_state, no_imp = -1, None, 0

    for epoch in range(epochs):
        model.train()
        # 联合训练：每个epoch内依次过两平台的batch（共享encoder的同一组参数更新）
        for X_b, y_b in neu_tr_l:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            out  = model(X_b, platform='neurips').squeeze()
            loss = crit_n(out, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        for X_b, y_b in ast_tr_l:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            out  = model(X_b, platform='assistments').squeeze()
            loss = crit_a(out, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # 联合验证分数 = 两平台 F1 之和
        model.eval()
        f1s = []
        for val_l, plat in [(neu_val_l, 'neurips'), (ast_val_l, 'assistments')]:
            preds, trues = [], []
            with torch.no_grad():
                for X_v, y_v in val_l:
                    out = model(X_v.to(device), platform=plat)
                    prob = torch.sigmoid(out).cpu().numpy().flatten()
                    preds.extend((prob>0.5).astype(int).tolist())
                    trues.extend(y_v.numpy().astype(int).tolist())
            f1s.append(f1_score(trues, preds, zero_division=0))
        score = sum(f1s)
        if score > best_score:
            best_score, best_state, no_imp = score, {k:v.clone() for k,v in model.state_dict().items()}, 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model

# ════════════════════════════════════════════════════════════════
# 数据加载（共享，与之前所有step一致）
# ════════════════════════════════════════════════════════════════
print("\n[1] Loading & aligning data...")
loader = CrossPlatformDataLoader(data_path=DATA_PATH)
X_neu_raw, y_neu_raw, _, feat_neu = loader.load_neurips_dataset(task='correctness')
X_ast_raw, y_ast_raw, _, feat_ast = loader.load_assistments_dataset(task='correctness')
X_neu_al = align_features(X_neu_raw, feat_neu, NEURIPS_MAP)
X_ast_al = align_features(X_ast_raw, feat_ast, ASSISTMENTS_MAP)
X_neu_dd, y_neu_dd = dedup(X_neu_al, y_neu_raw.astype(np.float32))
X_ast_dd, y_ast_dd = dedup(X_ast_al, y_ast_raw.astype(np.float32))
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

    neu_tr_l, neu_val_l, X_neu_te, y_neu_te = make_loaders(X_neu, y_neu, seed=seed)
    ast_tr_l, ast_val_l, X_ast_te, y_ast_te = make_loaders(X_ast, y_ast, seed=seed)

    # ── A) Platform-Aware ────────────────────────────────────
    print("\n  [A] Training Platform-Aware (EduPAMNet, per-platform decoders)...")
    set_seed(seed)
    model_aware_neu = train_aware(neu_tr_l, neu_val_l, 'neurips',     EPOCHS, DEVICE)
    set_seed(seed)
    model_aware_ast = train_aware(ast_tr_l, ast_val_l, 'assistments', EPOCHS, DEVICE)

    m_neu = eval_loader(lambda x: model_aware_neu(x, platform='neurips')['output'], X_neu_te, y_neu_te, DEVICE)
    m_ast = eval_loader(lambda x: model_aware_ast(x, platform='assistments')['output'], X_ast_te, y_ast_te, DEVICE)
    print(f"    Aware NeurIPS     F1={m_neu['f1']:.4f}  Acc={m_neu['accuracy']:.4f}")
    print(f"    Aware ASSISTments F1={m_ast['f1']:.4f}  Acc={m_ast['accuracy']:.4f}")

    # ── B) Platform-Agnostic ─────────────────────────────────
    print("\n  [B] Training Platform-Agnostic (MTLClassifier, shared encoder + simple heads, pooled training)...")
    set_seed(seed)
    model_agnostic = train_agnostic(neu_tr_l, neu_val_l, ast_tr_l, ast_val_l, EPOCHS, DEVICE)

    g_neu = eval_loader(lambda x: model_agnostic(x, platform='neurips'),     X_neu_te, y_neu_te, DEVICE)
    g_ast = eval_loader(lambda x: model_agnostic(x, platform='assistments'), X_ast_te, y_ast_te, DEVICE)
    print(f"    Agnostic NeurIPS     F1={g_neu['f1']:.4f}  Acc={g_neu['accuracy']:.4f}")
    print(f"    Agnostic ASSISTments F1={g_ast['f1']:.4f}  Acc={g_ast['accuracy']:.4f}")

    # ── 差异 ─────────────────────────────────────────────────
    diff_neu_f1  = m_neu['f1']  - g_neu['f1']
    diff_ast_f1  = m_ast['f1']  - g_ast['f1']
    pct_neu      = (diff_neu_f1 / g_neu['f1'] * 100) if g_neu['f1'] > 0 else 0
    pct_ast      = (diff_ast_f1 / g_ast['f1'] * 100) if g_ast['f1'] > 0 else 0
    print(f"\n    F1 diff (Aware - Agnostic):  NeurIPS={diff_neu_f1:+.4f} ({pct_neu:+.2f}%)  "
          f"ASSISTments={diff_ast_f1:+.4f} ({pct_ast:+.2f}%)")

    all_results.append({
        'seed': seed,
        'aware':    {'neurips': m_neu, 'assistments': m_ast},
        'agnostic': {'neurips': g_neu, 'assistments': g_ast},
        'f1_diff':  {'neurips': round(float(diff_neu_f1), 6), 'assistments': round(float(diff_ast_f1), 6)},
        'pct_diff': {'neurips': round(float(pct_neu), 4),     'assistments': round(float(pct_ast), 4)},
    })

# ════════════════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("AGGREGATING RESULTS (mean ± std across 5 seeds)")
print(f"{'='*65}")

def agg(values):
    arr = np.array(values, dtype=float)
    return {'mean': round(float(arr.mean()),6), 'std': round(float(arr.std()),6),
            'ci95': round(float(1.96*arr.std()/np.sqrt(len(arr))),6)}

aware_f1_neu = [r['aware']['neurips']['f1']     for r in all_results]
aware_f1_ast = [r['aware']['assistments']['f1'] for r in all_results]
agno_f1_neu  = [r['agnostic']['neurips']['f1']     for r in all_results]
agno_f1_ast  = [r['agnostic']['assistments']['f1'] for r in all_results]
diff_neu     = [r['f1_diff']['neurips']     for r in all_results]
diff_ast     = [r['f1_diff']['assistments'] for r in all_results]
pct_neu_l    = [r['pct_diff']['neurips']     for r in all_results]
pct_ast_l    = [r['pct_diff']['assistments'] for r in all_results]

# paired t-test (aware vs agnostic, pooled across both platforms x 5 seeds = n=10)
all_aware = aware_f1_neu + aware_f1_ast
all_agno  = agno_f1_neu  + agno_f1_ast
t_stat, p_val = stats.ttest_rel(all_aware, all_agno)

print(f"\n  Platform-Aware F1:")
print(f"    NeurIPS     : {agg(aware_f1_neu)['mean']:.4f} ± {agg(aware_f1_neu)['std']:.4f}")
print(f"    ASSISTments : {agg(aware_f1_ast)['mean']:.4f} ± {agg(aware_f1_ast)['std']:.4f}")
print(f"\n  Platform-Agnostic F1:")
print(f"    NeurIPS     : {agg(agno_f1_neu)['mean']:.4f} ± {agg(agno_f1_neu)['std']:.4f}")
print(f"    ASSISTments : {agg(agno_f1_ast)['mean']:.4f} ± {agg(agno_f1_ast)['std']:.4f}")
print(f"\n  F1 Difference (Aware - Agnostic):")
print(f"    NeurIPS     : {agg(diff_neu)['mean']:+.4f} ± {agg(diff_neu)['std']:.4f}  "
      f"({agg(pct_neu_l)['mean']:+.2f}% ± {agg(pct_neu_l)['std']:.2f}%)")
print(f"    ASSISTments : {agg(diff_ast)['mean']:+.4f} ± {agg(diff_ast)['std']:.4f}  "
      f"({agg(pct_ast_l)['mean']:+.2f}% ± {agg(pct_ast_l)['std']:.2f}%)")
print(f"\n  Paired t-test (Aware vs Agnostic, n=10 [2 platforms x 5 seeds]):")
print(f"    t = {t_stat:.4f}  p = {p_val:.4f}  "
      f"({'significant' if p_val < 0.05 else 'not significant'} at α=0.05)")

overall_pct = (agg(pct_neu_l)['mean'] + agg(pct_ast_l)['mean']) / 2
print(f"\n  Overall average improvement: {overall_pct:+.2f}%")

# ── 保存 ─────────────────────────────────────────────────────────
final = {
    'description': (
        "Platform-Aware (EduPAMNet: shared encoder + platform-specific "
        "BalancedDecoder, trained separately per platform) vs Platform-"
        "Agnostic (MTLClassifier: shared encoder + simple linear heads, "
        "jointly trained on pooled cross-platform data). Both use the "
        "same ImprovedSharedEncoder architecture and identical 5D aligned "
        "feature space, 5 seeds [42,123,7,2026,999], 30000 samples/platform."
    ),
    'config': {'seeds': SEEDS, 'sample_size': SAMPLE_SIZE, 'epochs': EPOCHS},
    'per_seed_results': all_results,
    'aggregated': {
        'aware_f1':    {'neurips': agg(aware_f1_neu), 'assistments': agg(aware_f1_ast)},
        'agnostic_f1': {'neurips': agg(agno_f1_neu),  'assistments': agg(agno_f1_ast)},
        'f1_diff':     {'neurips': agg(diff_neu), 'assistments': agg(diff_ast)},
        'pct_diff':    {'neurips': agg(pct_neu_l), 'assistments': agg(pct_ast_l)},
        'overall_pct_improvement': round(float(overall_pct), 4),
        'paired_ttest': {'t_statistic': round(float(t_stat),6), 'p_value': round(float(p_val),6)},
    }
}
out = os.path.join(OUTPUT_PATH, 'step8_aware_vs_agnostic_results.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(final, f, indent=2, ensure_ascii=False)
print(f"\n[SAVED] {out}")
print("\n>>> Step 8 COMPLETE <<<")
