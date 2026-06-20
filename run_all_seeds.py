"""
run_all_seeds.py
统一5种子循环实验脚本
SEEDS       = [42, 123, 7, 2026, 999]
涵盖：Step 2 (RO1) + Step 3 (RO2)
输出：每个指标的 mean ± std，符合论文 §3.9.2 要求

运行：python run_all_seeds.py
输出：results/final_results_all_seeds.json
"""

import os, sys, json, warnings
import numpy as np
import torch
import torch.nn as nn
warnings.filterwarnings('ignore')

# ── 配置 ────────────────────────────────────────────────────────
DATA_PATH   = './data/'
OUTPUT_PATH = './results/'
SEEDS       = [42, 123, 7, 2026, 999]
SAMPLE_SIZE = 30000
EPOCHS      = 80
BATCH_SIZE  = 256
CV_FOLDS    = 5
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
SHAP_SAMPLES = 200
LIME_SAMPLES = 50

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'interpretability'), exist_ok=True)

print("=" * 65)
print("Multi-Seed Experiment Runner")
print(f"  Seeds       : {SEEDS}")
print(f"  Sample size : {SAMPLE_SIZE:,}  |  Epochs: {EPOCHS}")
print(f"  Device      : {DEVICE}")
print("=" * 65)

# ── 导入 ────────────────────────────────────────────────────────
from data_loader_aligned import load_aligned_data, UNIFIED_FEATURES, INPUT_DIM
from optimized_pamnet_implementation import OptimizedPAMNet
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
import shap
import lime
import lime.lime_tabular

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed")

# ════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(y_true, y_pred, y_prob=None):
    return {
        'accuracy':  float(accuracy_score(y_true, y_pred)),
        'f1':        float(f1_score(y_true, y_pred, average='binary', zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, average='binary', zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, average='binary', zero_division=0)),
        'auc':       float(roc_auc_score(y_true, y_prob)) if y_prob is not None else None
    }

def cv_baseline(model, X, y, cv=5, seed=42):
    skf   = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    folds = []
    for tr, te in skf.split(X, y):
        m = model.__class__(**model.get_params())
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        prob = m.predict_proba(X[te])[:, 1] if hasattr(m, 'predict_proba') else None
        folds.append(compute_metrics(y[te], pred, prob))
    return {k: float(np.mean([f[k] for f in folds if f[k] is not None]))
            for k in folds[0]}

# ── Focal Loss ──────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none')
        pt  = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()

# ── EduPAMNet 训练 ───────────────────────────────────────────────
def train_pamnet(model, train_loader, val_loader, platform,
                 num_epochs=80, device='cpu'):
    model.to(device)
    all_y   = np.concatenate([y.numpy() for _, y in train_loader])
    pos     = float((all_y == 1).sum())
    neg     = float((all_y == 0).sum())
    pw      = torch.clamp(torch.tensor([neg / pos], dtype=torch.float32),
                          0.5, 2.0).to(device)
    crit    = FocalLoss(gamma=2.0, pos_weight=pw)
    opt     = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
                  opt, T_max=num_epochs, eta_min=1e-5)
    best_f1, best_state, no_imp, patience = -1, None, 0, 15

    for epoch in range(num_epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            out  = model(X_b, platform=platform)['output'].squeeze()
            loss = crit(out, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X_v, y_v in val_loader:
                out  = model(X_v.to(device), platform=platform)['output']
                prob = torch.sigmoid(out).cpu().numpy().flatten()
                preds.extend((prob > 0.5).astype(int).tolist())
                trues.extend(y_v.numpy().astype(int).tolist())
        val_f1 = f1_score(trues, preds, zero_division=0)

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    model.load_state_dict(best_state)
    return model

# ── SHAP 分析 ───────────────────────────────────────────────────
def run_shap(rf_model, X_test, n_samples=200):
    te          = X_test[:n_samples]
    explainer   = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(te)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    if sv.ndim > 2:
        sv = sv[:, :, 0]
    return np.abs(sv).mean(axis=0).flatten()

# ── LIME 分析 ───────────────────────────────────────────────────
def run_lime(X_train, X_test, predict_fn, feature_names,
             n_samples=50, seed=42):
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

def make_predict_fn(model, platform, device):
    def predict_proba(X):
        X_t = torch.tensor(X.astype(np.float32)).to(device)
        with torch.no_grad():
            out  = model(X_t, platform=platform)['output']
            prob = torch.sigmoid(out).cpu().numpy().flatten()
        return np.column_stack([1 - prob, prob])
    return predict_proba

# ════════════════════════════════════════════════════════════════
# 主循环
# ════════════════════════════════════════════════════════════════
print("\n[1] Loading data (once, shared across seeds)...")
# 数据加载与对齐只做一次（采样在每个seed内进行）
from data_loader02 import CrossPlatformDataLoader
from data_loader_aligned import align_features, NEURIPS_MAP, ASSISTMENTS_MAP

loader = CrossPlatformDataLoader(data_path=DATA_PATH)
X_neu_raw, y_neu_raw, _, feat_neu = loader.load_neurips_dataset(task='correctness')
X_ast_raw, y_ast_raw, _, feat_ast = loader.load_assistments_dataset(task='correctness')

X_neu_al = align_features(X_neu_raw, feat_neu, NEURIPS_MAP)
X_ast_al = align_features(X_ast_raw, feat_ast, ASSISTMENTS_MAP)
feature_names = UNIFIED_FEATURES
input_dim     = INPUT_DIM

print(f"  NeurIPS aligned   : {X_neu_al.shape}")
print(f"  ASSISTments aligned: {X_ast_al.shape}")

# 存储每个seed的结果
all_results = []

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*65}")
    print(f"SEED {seed}  ({seed_idx+1}/{len(SEEDS)})")
    print(f"{'='*65}")
    set_seed(seed)

    # ── 分层采样（每个seed不同采样）──────────────────────────
    from sklearn.preprocessing import StandardScaler

    def stratified_sample(X, y, n, seed):
        if len(X) <= n:
            return X.astype(np.float32), y.astype(np.float32)
        from sklearn.model_selection import train_test_split as tts
        _, Xs, _, ys = tts(X, y, test_size=n/len(X),
                            stratify=y.astype(int), random_state=seed)
        return Xs.astype(np.float32), ys.astype(np.float32)

    # 去重
    def dedup(X, y):
        combined = np.hstack([X, y.reshape(-1,1)])
        _, idx   = np.unique(combined.round(6), axis=0, return_index=True)
        return X[idx], y[idx]

    X_neu_dd, y_neu_dd = dedup(X_neu_al, y_neu_raw.astype(np.float32))
    X_ast_dd, y_ast_dd = dedup(X_ast_al, y_ast_raw.astype(np.float32))

    X_neu, y_neu = stratified_sample(X_neu_dd, y_neu_dd, SAMPLE_SIZE, seed)
    X_ast, y_ast = stratified_sample(X_ast_dd, y_ast_dd, SAMPLE_SIZE, seed)

    scaler_n = StandardScaler()
    scaler_a = StandardScaler()
    X_neu = scaler_n.fit_transform(X_neu).astype(np.float32)
    X_ast = scaler_a.fit_transform(X_ast).astype(np.float32)

    print(f"  NeurIPS : {X_neu.shape}  balance={y_neu.mean():.3f}")
    print(f"  ASSISTments: {X_ast.shape}  balance={y_ast.mean():.3f}")

    # ── Step 2: 基线模型 ──────────────────────────────────────
    print(f"\n  [RO1] Baseline models...")
    lr  = LogisticRegression(max_iter=500, random_state=seed)
    rf  = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)

    bl = {}
    bl['LR_neurips']     = cv_baseline(lr, X_neu, y_neu.astype(int), CV_FOLDS, seed)
    bl['LR_assistments'] = cv_baseline(lr, X_ast, y_ast.astype(int), CV_FOLDS, seed)
    bl['RF_neurips']     = cv_baseline(rf, X_neu, y_neu.astype(int), CV_FOLDS, seed)
    bl['RF_assistments'] = cv_baseline(rf, X_ast, y_ast.astype(int), CV_FOLDS, seed)
    if HAS_XGB:
        xgb = XGBClassifier(n_estimators=200, learning_rate=0.1,
                             use_label_encoder=False, eval_metric='logloss',
                             random_state=seed, n_jobs=-1)
        bl['XGB_neurips']     = cv_baseline(xgb, X_neu, y_neu.astype(int), CV_FOLDS, seed)
        bl['XGB_assistments'] = cv_baseline(xgb, X_ast, y_ast.astype(int), CV_FOLDS, seed)

    neu_base = np.mean([bl['RF_neurips']['f1'], bl['LR_neurips']['f1']])
    ast_base = np.mean([bl['RF_assistments']['f1'], bl['LR_assistments']['f1']])
    print(f"  Baseline F1  NeurIPS={neu_base:.4f}  ASSISTments={ast_base:.4f}")

    # ── Step 2: EduPAMNet 训练 ───────────────────────────────
    print(f"\n  [RO1] Training EduPAMNet...")

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

    tr_n, val_n, X_neu_te, y_neu_te = make_loaders(X_neu, y_neu, seed=seed)
    model_neu = OptimizedPAMNet(input_dim=input_dim)
    model_neu = train_pamnet(model_neu, tr_n, val_n, 'neurips', EPOCHS, DEVICE)

    tr_a, val_a, X_ast_te, y_ast_te = make_loaders(X_ast, y_ast, seed=seed)
    model_ast = OptimizedPAMNet(input_dim=input_dim)
    model_ast = train_pamnet(model_ast, tr_a, val_a, 'assistments', EPOCHS, DEVICE)

    # 评估
    def eval_model(model, X_te, y_te, platform):
        model.eval()
        with torch.no_grad():
            out  = model(torch.tensor(X_te).to(DEVICE), platform=platform)['output']
            prob = torch.sigmoid(out).cpu().numpy().flatten()
            pred = (prob > 0.5).astype(int)
        return compute_metrics(y_te.astype(int), pred, prob)

    neu_m = eval_model(model_neu, X_neu_te, y_neu_te, 'neurips')
    ast_m = eval_model(model_ast, X_ast_te, y_ast_te, 'assistments')
    print(f"  EduPAMNet NeurIPS     F1={neu_m['f1']:.4f}  Acc={neu_m['accuracy']:.4f}")
    print(f"  EduPAMNet ASSISTments F1={ast_m['f1']:.4f}  Acc={ast_m['accuracy']:.4f}")

    neu_gain = neu_m['f1'] - neu_base
    ast_gain = ast_m['f1'] - ast_base

    # ── Step 3: SHAP / LIME / ECI / FISS ────────────────────
    print(f"\n  [RO2] SHAP & LIME analysis...")

    # 代理RF
    predict_neu = make_predict_fn(model_neu, 'neurips', DEVICE)
    predict_ast = make_predict_fn(model_ast, 'assistments', DEVICE)

    rf_sur_n = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf_sur_a = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)

    X_tr_n, X_te_n, _, _ = train_test_split(
        X_neu, y_neu, test_size=0.2, stratify=y_neu.astype(int), random_state=seed)
    X_tr_a, X_te_a, _, _ = train_test_split(
        X_ast, y_ast, test_size=0.2, stratify=y_ast.astype(int), random_state=seed)

    soft_n = (predict_neu(X_tr_n)[:, 1] > 0.5).astype(int)
    soft_a = (predict_ast(X_tr_a)[:, 1] > 0.5).astype(int)
    rf_sur_n.fit(X_tr_n, soft_n)
    rf_sur_a.fit(X_tr_a, soft_a)

    shap_n = run_shap(rf_sur_n, X_te_n, SHAP_SAMPLES)
    shap_a = run_shap(rf_sur_a, X_te_a, SHAP_SAMPLES)

    lime_n = run_lime(X_tr_n, X_te_n, predict_neu, feature_names, LIME_SAMPLES, seed)
    lime_a = run_lime(X_tr_a, X_te_a, predict_ast, feature_names, LIME_SAMPLES, seed)

    # ECI
    eci_n, eci_n_p = spearmanr(shap_n, lime_n)
    eci_a, eci_a_p = spearmanr(shap_a, lime_a)
    eci_n = float(eci_n) if not np.isnan(eci_n) else 0.0
    eci_a = float(eci_a) if not np.isnan(eci_a) else 0.0

    # FISS
    fiss, fiss_p = spearmanr(shap_n, shap_a)
    fiss = float(fiss) if not np.isnan(fiss) else 0.0

    print(f"  ECI  NeurIPS={eci_n:.4f}  ASSISTments={eci_a:.4f}  "
          f"Mean={(eci_n+eci_a)/2:.4f}")
    print(f"  FISS={fiss:.4f}  p={fiss_p:.4f}")

    # ── 保存本次seed结果 ──────────────────────────────────────
    seed_result = {
        'seed': seed,
        'baselines': {k: {m: round(v,6) if v else None for m,v in met.items()}
                      for k, met in bl.items()},
        'single_task_baseline': {
            'neurips_f1':     round(float(neu_base), 6),
            'assistments_f1': round(float(ast_base), 6),
        },
        'edupamnet': {
            'neurips':     {k: round(v,6) if v else None for k,v in neu_m.items()},
            'assistments': {k: round(v,6) if v else None for k,v in ast_m.items()},
        },
        'f1_gain': {
            'neurips':     round(float(neu_gain), 6),
            'assistments': round(float(ast_gain), 6),
        },
        'shap': {
            'neurips':     shap_n.tolist(),
            'assistments': shap_a.tolist(),
        },
        'lime': {
            'neurips':     lime_n.tolist(),
            'assistments': lime_a.tolist(),
        },
        'eci': {
            'neurips':     round(eci_n, 6),
            'assistments': round(eci_a, 6),
            'mean':        round((eci_n + eci_a) / 2, 6),
        },
        'fiss': round(fiss, 6),
    }
    all_results.append(seed_result)
    print(f"  [Seed {seed} done]")

# ════════════════════════════════════════════════════════════════
# 汇总统计：mean ± std
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("AGGREGATING RESULTS (mean ± std across 5 seeds)")
print(f"{'='*65}")

def agg(values):
    arr = [v for v in values if v is not None]
    return {'mean': round(float(np.mean(arr)), 6),
            'std':  round(float(np.std(arr)),  6),
            'ci95': round(float(1.96 * np.std(arr) / np.sqrt(len(arr))), 6)}

# RO1 汇总
metric_keys = ['accuracy', 'f1', 'precision', 'recall', 'auc']
model_keys  = ['LR_neurips', 'LR_assistments', 'RF_neurips', 'RF_assistments']
if HAS_XGB:
    model_keys += ['XGB_neurips', 'XGB_assistments']

baseline_agg = {}
for mk in model_keys:
    baseline_agg[mk] = {
        met: agg([r['baselines'][mk][met] for r in all_results])
        for met in metric_keys
    }

pamnet_agg = {
    'neurips':     {met: agg([r['edupamnet']['neurips'][met]
                               for r in all_results]) for met in metric_keys},
    'assistments': {met: agg([r['edupamnet']['assistments'][met]
                               for r in all_results]) for met in metric_keys},
}

gain_agg = {
    'neurips':     agg([r['f1_gain']['neurips']     for r in all_results]),
    'assistments': agg([r['f1_gain']['assistments'] for r in all_results]),
}

# RO2 汇总
shap_agg = {
    'neurips':     [agg([r['shap']['neurips'][i]     for r in all_results])
                    for i in range(input_dim)],
    'assistments': [agg([r['shap']['assistments'][i] for r in all_results])
                    for i in range(input_dim)],
}
eci_agg = {
    'neurips':     agg([r['eci']['neurips']     for r in all_results]),
    'assistments': agg([r['eci']['assistments'] for r in all_results]),
    'mean':        agg([r['eci']['mean']        for r in all_results]),
}
fiss_agg = agg([r['fiss'] for r in all_results])

# ── 打印摘要 ─────────────────────────────────────────────────────
print(f"\n  {'Model':32s}  {'F1 mean±std':>14}  {'Acc mean±std':>14}  {'AUC mean±std':>14}")
print(f"  {'-'*76}")
for mk in model_keys:
    f1  = baseline_agg[mk]['f1']
    acc = baseline_agg[mk]['accuracy']
    auc = baseline_agg[mk]['auc']
    print(f"  {mk:32s}  {f1['mean']:.4f}±{f1['std']:.4f}   "
          f"{acc['mean']:.4f}±{acc['std']:.4f}   "
          f"{auc['mean']:.4f}±{auc['std']:.4f}")

for plat in ['neurips', 'assistments']:
    m   = pamnet_agg[plat]
    print(f"  {'EduPAMNet '+plat:32s}  {m['f1']['mean']:.4f}±{m['f1']['std']:.4f}   "
          f"{m['accuracy']['mean']:.4f}±{m['accuracy']['std']:.4f}   "
          f"{m['auc']['mean']:.4f}±{m['auc']['std']:.4f}")

print(f"\n  MTL F1 gain:")
print(f"    NeurIPS     : {gain_agg['neurips']['mean']:+.4f} ± {gain_agg['neurips']['std']:.4f}")
print(f"    ASSISTments : {gain_agg['assistments']['mean']:+.4f} ± {gain_agg['assistments']['std']:.4f}")

print(f"\n  RO2 Metrics:")
print(f"    FISS : {fiss_agg['mean']:.4f} ± {fiss_agg['std']:.4f}  "
      f"(95%CI ±{fiss_agg['ci95']:.4f})")
print(f"    ECI  NeurIPS    : {eci_agg['neurips']['mean']:.4f} ± {eci_agg['neurips']['std']:.4f}")
print(f"    ECI  ASSISTments: {eci_agg['assistments']['mean']:.4f} ± {eci_agg['assistments']['std']:.4f}")
print(f"    ECI  Mean       : {eci_agg['mean']['mean']:.4f} ± {eci_agg['mean']['std']:.4f}  "
      f"(95%CI ±{eci_agg['mean']['ci95']:.4f})")

print(f"\n  SHAP feature importance (mean across seeds):")
for i, fn in enumerate(feature_names):
    sn = shap_agg['neurips'][i]
    sa = shap_agg['assistments'][i]
    print(f"    {fn:25s}  NeurIPS={sn['mean']:.4f}±{sn['std']:.4f}  "
          f"ASSISTments={sa['mean']:.4f}±{sa['std']:.4f}")

# ── 保存最终结果 ──────────────────────────────────────────────────
final = {
    'config': {
        'seeds':       SEEDS,
        'sample_size': SAMPLE_SIZE,
        'epochs':      EPOCHS,
        'input_dim':   input_dim,
        'features':    feature_names,
        'device':      DEVICE,
    },
    'per_seed_results': all_results,
    'aggregated': {
        'baselines':   baseline_agg,
        'edupamnet':   pamnet_agg,
        'f1_gain':     gain_agg,
        'shap':        {'neurips': shap_agg['neurips'],
                        'assistments': shap_agg['assistments'],
                        'feature_names': feature_names},
        'eci':         eci_agg,
        'fiss':        fiss_agg,
    }
}

out = os.path.join(OUTPUT_PATH, 'final_results_all_seeds.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(final, f, indent=2, ensure_ascii=False)
print(f"\n[SAVED] {out}")
print("\n>>> All seeds complete — ready for Step 4 (RO3) <<<")
