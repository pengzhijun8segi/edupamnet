"""
step2_no_adversarial.py
EduPAMNet 简化测试版 — 禁用域对抗，纯分类训练
目的：测试在无域对抗干扰下 EduPAMNet 能否超越基线

架构保留：
  ✓ SharedEncoder（共享编码器）
  ✓ Platform-specific Decoder（平台专用解码器）
  ✓ Focal Loss（处理类别不平衡）
  ✗ Domain Adversarial（禁用）
  ✗ Optimal Transport（禁用）
  ✗ CrossTaskAttention（禁用，减少复杂度）
"""

import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

DATA_PATH    = './data/'
OUTPUT_PATH  = './results/'
SAMPLE_SIZE  = 30000
RANDOM_STATE = 42
EPOCHS       = 50
BATCH_SIZE   = 256
CV_FOLDS     = 5
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 65)
print("Step 2 — No-Adversarial Test")
print(f"  Device : {DEVICE}  |  Epochs : {EPOCHS}")
print("=" * 65)

from data_loader_aligned import load_aligned_data, INPUT_DIM, UNIFIED_FEATURES
from optimized_pamnet_implementation import OptimizedPAMNet
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ════════════════════════════════════════════════════════════════
# 纯分类训练函数（无域对抗）
# ════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma    = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight, reduction='none')
        pt  = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


def train_pure_classification(model, train_loader, val_loader,
                               platform, num_epochs=80, device='cpu'):
    """
    纯分类训练：只优化 Focal Loss，无任何域对抗损失。
    """
    model.to(device)

    # 类别权重
    all_y = np.concatenate([y.numpy() for _, y in train_loader])
    pos   = float((all_y == 1).sum())
    neg   = float((all_y == 0).sum())
    pw    = torch.tensor([neg / pos], dtype=torch.float32).to(device)
    pw    = torch.clamp(pw, 0.5, 2.0)
    print(f"  Neg={int(neg):,}  Pos={int(pos):,}  pos_weight={pw.item():.4f}")

    criterion = FocalLoss(gamma=2.0, pos_weight=pw)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5)

    best_f1    = -1
    best_state = None
    patience   = 15
    no_improve = 0

    for epoch in range(num_epochs):
        # ── 训练 ──────────────────────────────────────────────
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out  = model(X_b, platform=platform)['output'].squeeze()
            loss = criterion(out, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # ── 验证 ──────────────────────────────────────────────
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X_v, y_v in val_loader:
                out  = model(X_v.to(device), platform=platform)['output']
                prob = torch.sigmoid(out).cpu().numpy().flatten()
                preds.extend((prob > 0.5).astype(int).tolist())
                trues.extend(y_v.numpy().astype(int).tolist())

        val_f1  = f1_score(trues, preds, zero_division=0)
        val_acc = accuracy_score(trues, preds)

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}  val_acc={val_acc:.4f}  "
                  f"val_f1={val_f1:.4f}  best_f1={best_f1:.4f}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}  best_f1={best_f1:.4f}")
            break

    model.load_state_dict(best_state)
    return model

# ════════════════════════════════════════════════════════════════
# 1. 数据加载
# ════════════════════════════════════════════════════════════════
print("\n[1] Loading data...")
X_neu, y_neu, X_ast, y_ast, input_dim = load_aligned_data(
    data_path=DATA_PATH, sample_size=SAMPLE_SIZE,
    random_state=RANDOM_STATE)
print(f"  NeurIPS     : {X_neu.shape}  balance={y_neu.mean():.3f}")
print(f"  ASSISTments : {X_ast.shape}  balance={y_ast.mean():.3f}")

# ════════════════════════════════════════════════════════════════
# 2. 评估函数
# ════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred, y_prob=None, label=""):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average='binary', zero_division=0)
    prec = precision_score(y_true, y_pred, average='binary', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='binary', zero_division=0)
    auc  = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    if label:
        auc_str = f"  AUC={auc:.4f}" if auc else ""
        print(f"  {label:32s}  Acc={acc:.4f}  F1={f1:.4f}"
              f"  P={prec:.4f}  R={rec:.4f}{auc_str}")
    return {'accuracy': float(acc), 'f1': float(f1),
            'precision': float(prec), 'recall': float(rec),
            'auc': float(auc) if auc else None}

def cv_baseline(model, X, y, cv=5, label=""):
    skf   = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    for tr, te in skf.split(X, y):
        m = model.__class__(**model.get_params())
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        prob = m.predict_proba(X[te])[:, 1] if hasattr(m, 'predict_proba') else None
        folds.append(compute_metrics(y[te], pred, prob))
    avg = {k: float(np.mean([f[k] for f in folds if f[k] is not None]))
           for k in folds[0]}
    if label:
        auc_str = f"  AUC={avg['auc']:.4f}" if avg.get('auc') else ""
        print(f"  {label:32s}  Acc={avg['accuracy']:.4f}  F1={avg['f1']:.4f}"
              f"  P={avg['precision']:.4f}  R={avg['recall']:.4f}{auc_str}")
    return avg

# ════════════════════════════════════════════════════════════════
# 3. 基线
# ════════════════════════════════════════════════════════════════
print("\n[2] Baseline models...")
baselines = {}
lr = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)

baselines['LR_neurips']     = cv_baseline(lr, X_neu, y_neu.astype(int), label="LR  NeurIPS")
baselines['LR_assistments'] = cv_baseline(lr, X_ast, y_ast.astype(int), label="LR  ASSISTments")
baselines['RF_neurips']     = cv_baseline(rf, X_neu, y_neu.astype(int), label="RF  NeurIPS")
baselines['RF_assistments'] = cv_baseline(rf, X_ast, y_ast.astype(int), label="RF  ASSISTments")
if HAS_XGB:
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.1,
                         use_label_encoder=False, eval_metric='logloss',
                         random_state=RANDOM_STATE, n_jobs=-1)
    baselines['XGB_neurips']     = cv_baseline(xgb, X_neu, y_neu.astype(int), label="XGB NeurIPS")
    baselines['XGB_assistments'] = cv_baseline(xgb, X_ast, y_ast.astype(int), label="XGB ASSISTments")

neu_baseline = np.mean([baselines['RF_neurips']['f1'],
                         baselines['LR_neurips']['f1']])
ast_baseline = np.mean([baselines['RF_assistments']['f1'],
                         baselines['LR_assistments']['f1']])
print(f"\n  Baseline F1  NeurIPS={neu_baseline:.4f}  ASSISTments={ast_baseline:.4f}")

# ════════════════════════════════════════════════════════════════
# 4. EduPAMNet 纯分类训练
# ════════════════════════════════════════════════════════════════
print(f"\n[3] EduPAMNet — pure classification (no adversarial)...")

def make_loaders(X, y, batch=256, val=0.2):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=val,
        stratify=y.astype(int), random_state=RANDOM_STATE)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=val,
        stratify=y_tr.astype(int), random_state=RANDOM_STATE)
    return (DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                       batch_size=batch, shuffle=True),
            DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                       batch_size=batch),
            X_te, y_te)

print("\n  -- NeurIPS --")
tr_n, val_n, X_neu_te, y_neu_te = make_loaders(X_neu, y_neu)
model_neu = OptimizedPAMNet(input_dim=input_dim)
model_neu = train_pure_classification(
    model_neu, tr_n, val_n, 'neurips', EPOCHS, DEVICE)

print("\n  -- ASSISTments --")
tr_a, val_a, X_ast_te, y_ast_te = make_loaders(X_ast, y_ast)
model_ast = OptimizedPAMNet(input_dim=input_dim)
model_ast = train_pure_classification(
    model_ast, tr_a, val_a, 'assistments', EPOCHS, DEVICE)

# ════════════════════════════════════════════════════════════════
# 5. 评估
# ════════════════════════════════════════════════════════════════
print("\n[4] Evaluation...")

def eval_model(model, X_te, y_te, platform, label):
    model.eval()
    with torch.no_grad():
        out  = model(torch.tensor(X_te).to(DEVICE), platform=platform)['output']
        prob = torch.sigmoid(out).cpu().numpy().flatten()
        pred = (prob > 0.5).astype(int)
    return compute_metrics(y_te.astype(int), pred, prob, label=label)

neu_m = eval_model(model_neu, X_neu_te, y_neu_te, 'neurips',
                    'EduPAMNet (no-adv) NeurIPS')
ast_m = eval_model(model_ast, X_ast_te, y_ast_te, 'assistments',
                    'EduPAMNet (no-adv) ASSISTments')

neu_gain = neu_m['f1'] - neu_baseline
ast_gain = ast_m['f1'] - ast_baseline

# ════════════════════════════════════════════════════════════════
# 6. 结果摘要
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("SUMMARY — No-Adversarial Test")
print("=" * 65)
print(f"  {'Model':35s}  {'F1':>7}  {'Acc':>7}  {'AUC':>7}")
print(f"  {'-'*57}")
for name, m in [
    ('LR  NeurIPS',          baselines['LR_neurips']),
    ('RF  NeurIPS',          baselines['RF_neurips']),
    ('XGB NeurIPS',          baselines.get('XGB_neurips', {})),
    ('EduPAMNet NeurIPS',    neu_m),
]:
    if not m: continue
    auc = f"{m['auc']:.4f}" if m.get('auc') else "  N/A"
    print(f"  {name:35s}  {m['f1']:.4f}   {m['accuracy']:.4f}   {auc}")

print(f"  {'-'*57}")
for name, m in [
    ('LR  ASSISTments',      baselines['LR_assistments']),
    ('RF  ASSISTments',      baselines['RF_assistments']),
    ('XGB ASSISTments',      baselines.get('XGB_assistments', {})),
    ('EduPAMNet ASSISTments',ast_m),
]:
    if not m: continue
    auc = f"{m['auc']:.4f}" if m.get('auc') else "  N/A"
    print(f"  {name:35s}  {m['f1']:.4f}   {m['accuracy']:.4f}   {auc}")

print(f"\n  MTL gain (vs RF+LR mean):")
print(f"    NeurIPS     : {neu_gain:+.4f}  "
      f"({'✓ POSITIVE' if neu_gain > 0 else '✗ negative'})")
print(f"    ASSISTments : {ast_gain:+.4f}  "
      f"({'✓ POSITIVE' if ast_gain > 0 else '✗ negative'})")

# 保存
results = {
    'mode': 'no_adversarial',
    'baselines': {k: {m: round(v,6) if v else None for m,v in met.items()}
                  for k, met in baselines.items()},
    'edupamnet_no_adv': {
        'neurips':     {k: round(v,6) if v else None for k,v in neu_m.items()},
        'assistments': {k: round(v,6) if v else None for k,v in ast_m.items()},
    },
    'f1_gain': {
        'neurips':     round(float(neu_gain), 6),
        'assistments': round(float(ast_gain), 6),
    }
}
with open(os.path.join(OUTPUT_PATH, 'step2_no_adv_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n[SAVED] results/step2_no_adv_results.json")

# 决策建议
print("\n" + "=" * 65)
if neu_gain > 0 and ast_gain > 0:
    print(">>> 结论：禁用域对抗后 EduPAMNet 超越基线")
    print(">>> 建议：在论文中说明域对抗对低维特征有负面影响，")
    print("         EduPAMNet 在纯多任务学习模式下有效")
elif neu_gain > 0 or ast_gain > 0:
    print(">>> 结论：部分平台超越基线")
    print(">>> 建议：分析两平台差异，报告条件性有效结论")
else:
    print(">>> 结论：即使禁用域对抗，EduPAMNet 仍不及基线")
    print(">>> 建议：采用诚实报告方案，分析架构局限性")
    print("         5维特征空间可能不足以体现深度学习优势")
