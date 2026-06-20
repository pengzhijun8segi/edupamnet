"""
step2_ro1_training.py  (v3 — 修复域对抗权重)
Step 2 — RO1: Cross-Platform High-Performance Prediction

修复：降低域对抗损失权重，解决 EduPAMNet recall 崩塌问题
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
print("Step 2 — RO1: Cross-Platform High-Performance Prediction (v3)")
print(f"  Device      : {DEVICE}")
print(f"  Sample size : {SAMPLE_SIZE:,}  |  Epochs: {EPOCHS}")
print("=" * 65)

# ── 导入模块 ────────────────────────────────────────────────────
from data_loader_aligned import load_aligned_data, INPUT_DIM, UNIFIED_FEATURES
from optimized_pamnet_implementation import (
    OptimizedPAMNet, train_optimized_pamnet,
    AdvancedMultiTaskLoss, get_advanced_optimizer
)
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
    print("[WARN] xgboost not installed")

# ════════════════════════════════════════════════════════════════
# 自定义训练函数（降低域对抗权重）
# ════════════════════════════════════════════════════════════════
def train_pamnet_conservative(model, train_loader, val_loader,
                               source_loader, num_epochs=50,
                               device='cpu', target_platform='neurips'):
    """
    保守版训练：降低域对抗权重，专注分类任务。
    domain_weight: 0.15 → 0.05
    adversarial_weight: 0.12 → 0.03
    balance_weight: 0.20 → 0.10
    """
    from optimized_pamnet_implementation import get_advanced_optimizer
    import numpy as np

    model.to(device)

    # 计算类别权重
    all_labels = []
    for _, y_batch in train_loader:
        all_labels.extend(y_batch.numpy())
    all_labels = np.array(all_labels)
    pos_count = np.sum(all_labels == 1)
    neg_count = np.sum(all_labels == 0)
    pos_weight = torch.clamp(
        torch.tensor([neg_count / pos_count], dtype=torch.float32),
        0.5, 2.0).to(device)
    print(f"  Class dist — Neg:{neg_count:,} Pos:{pos_count:,}  "
          f"pos_weight={pos_weight.item():.4f}")

    # 保守损失权重
    criterion = AdvancedMultiTaskLoss(
        task_weight=1.0,
        domain_weight=0.05,       # 原 0.15 → 降低
        ot_weight=0.02,
        adversarial_weight=0.03,  # 原 0.12 → 大幅降低
        balance_weight=0.10       # 原 0.20 → 降低
    )

    optimizer = get_advanced_optimizer(model, lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5)

    # 预缓存源域特征
    src_features = []
    model.eval()
    with torch.no_grad():
        for X_src, _ in source_loader:
            feat = model.shared_encoder(X_src.to(device))
            src_features.append(feat)
    src_features = torch.cat(src_features, dim=0)

    best_score = -1
    best_state = None
    patience_counter = 0
    patience = 15

    print(f"  Training {num_epochs} epochs (conservative weights)...")
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            predictions = model(X_batch, platform=target_platform)
            src_idx = torch.randint(0, len(src_features),
                                    (len(X_batch),))
            domain_labels = torch.zeros(len(X_batch)).to(device)
            loss, _ = criterion(predictions, y_batch,
                                domain_labels, pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # 验证
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                out  = model(X_val.to(device), platform=target_platform)['output']
                prob = torch.sigmoid(out).cpu().numpy().flatten()
                val_preds.extend((prob > 0.5).astype(int).tolist())
                val_true.extend(y_val.numpy().astype(int).tolist())

        val_f1  = f1_score(val_true, val_preds, zero_division=0)
        val_acc = accuracy_score(val_true, val_preds)
        score   = val_f1 + val_acc

        if score > best_score:
            best_score = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs}  "
                  f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  "
                  f"best={best_score:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    print(f"  Best combined score: {best_score:.4f}")
    return model

# ════════════════════════════════════════════════════════════════
# 1. 数据加载
# ════════════════════════════════════════════════════════════════
print("\n[1] Loading & aligning datasets...")
X_neu, y_neu, X_ast, y_ast, input_dim = load_aligned_data(
    data_path=DATA_PATH, sample_size=SAMPLE_SIZE,
    random_state=RANDOM_STATE)
print(f"  Features ({input_dim}): {UNIFIED_FEATURES}")
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
    skf = StratifiedKFold(n_splits=cv, shuffle=True,
                          random_state=RANDOM_STATE)
    folds = []
    for tr, te in skf.split(X, y):
        m = model.__class__(**model.get_params())
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        prob = (m.predict_proba(X[te])[:, 1]
                if hasattr(m, 'predict_proba') else None)
        folds.append(compute_metrics(y[te], pred, prob))
    avg = {k: float(np.mean([f[k] for f in folds
                              if f[k] is not None]))
           for k in folds[0]}
    if label:
        auc_str = f"  AUC={avg['auc']:.4f}" if avg.get('auc') else ""
        print(f"  {label:32s}  Acc={avg['accuracy']:.4f}  F1={avg['f1']:.4f}"
              f"  P={avg['precision']:.4f}  R={avg['recall']:.4f}{auc_str}")
    return avg

# ════════════════════════════════════════════════════════════════
# 3. 基线模型
# ════════════════════════════════════════════════════════════════
print("\n[2] Training baseline models (5-fold CV)...")
baselines = {}

lr = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                             n_jobs=-1)

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
print(f"\n  >> True single-task baseline F1:")
print(f"     NeurIPS     = {neu_baseline:.4f}")
print(f"     ASSISTments = {ast_baseline:.4f}")

# ════════════════════════════════════════════════════════════════
# 4. EduPAMNet 训练（保守版）
# ════════════════════════════════════════════════════════════════
print(f"\n[3] Training EduPAMNet (conservative, {EPOCHS} epochs)...")

def prepare_loaders(X, y, X_src, y_src, batch=256, val_ratio=0.2):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=val_ratio,
        stratify=y.astype(int), random_state=RANDOM_STATE)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=val_ratio,
        stratify=y_tr.astype(int), random_state=RANDOM_STATE)
    train_l = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                         batch_size=batch, shuffle=True)
    val_l   = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                         batch_size=batch)
    src_l   = DataLoader(TensorDataset(torch.tensor(X_src), torch.tensor(y_src)),
                         batch_size=batch, shuffle=True)
    return train_l, val_l, src_l, X_te, y_te

print("\n  -- NeurIPS model --")
train_l, val_l, src_l, X_neu_te, y_neu_te = prepare_loaders(
    X_neu, y_neu, X_ast, y_ast)
model_neu = OptimizedPAMNet(input_dim=input_dim)
model_neu = train_pamnet_conservative(
    model_neu, train_l, val_l, src_l,
    num_epochs=EPOCHS, device=DEVICE, target_platform='neurips')

print("\n  -- ASSISTments model --")
train_l2, val_l2, src_l2, X_ast_te, y_ast_te = prepare_loaders(
    X_ast, y_ast, X_neu, y_neu)
model_ast = OptimizedPAMNet(input_dim=input_dim)
model_ast = train_pamnet_conservative(
    model_ast, train_l2, val_l2, src_l2,
    num_epochs=EPOCHS, device=DEVICE, target_platform='assistments')

# ════════════════════════════════════════════════════════════════
# 5. 评估
# ════════════════════════════════════════════════════════════════
print("\n[4] Evaluating EduPAMNet...")

def eval_pamnet(model, X_te, y_te, platform):
    model.eval()
    with torch.no_grad():
        out  = model(torch.tensor(X_te).to(DEVICE), platform=platform)['output']
        prob = torch.sigmoid(out).cpu().numpy()
        pred = (prob > 0.5).astype(int)
    return compute_metrics(y_te.astype(int), pred, prob,
                           label=f"EduPAMNet {platform}")

neu_m = eval_pamnet(model_neu, X_neu_te, y_neu_te, 'neurips')
ast_m = eval_pamnet(model_ast, X_ast_te, y_ast_te, 'assistments')

# ════════════════════════════════════════════════════════════════
# 6. 多任务提升量
# ════════════════════════════════════════════════════════════════
print("\n[5] Multi-task improvement...")
neu_gain = neu_m['f1'] - neu_baseline
ast_gain = ast_m['f1'] - ast_baseline
f1_cons  = 1 - abs(neu_m['f1'] - ast_m['f1'])
acc_cons = 1 - abs(neu_m['accuracy'] - ast_m['accuracy'])
print(f"  NeurIPS    F1 gain : {neu_gain:+.4f}  ({neu_baseline:.4f} → {neu_m['f1']:.4f})")
print(f"  ASSISTments F1 gain: {ast_gain:+.4f}  ({ast_baseline:.4f} → {ast_m['f1']:.4f})")
print(f"  Cross-platform F1  consistency : {f1_cons:.4f}")
print(f"  Cross-platform Acc consistency : {acc_cons:.4f}")

# ════════════════════════════════════════════════════════════════
# 7. 保存
# ════════════════════════════════════════════════════════════════
enc_dim      = model_neu.shared_encoder.output_dim
dec_params_n = sum(p.numel() for p in model_neu.neurips_decoder.parameters())
dec_params_a = sum(p.numel() for p in model_ast.assistments_decoder.parameters())
total_params = sum(p.numel() for p in model_neu.parameters())

results = {
    'sampling': {
        'method': 'stratified_dedup',
        'size_per_platform': SAMPLE_SIZE,
        'input_dim': input_dim,
        'unified_features': UNIFIED_FEATURES,
    },
    'baselines': {k: {m: round(v, 6) if v is not None else None
                      for m, v in met.items()}
                  for k, met in baselines.items()},
    'single_task_baseline': {
        'neurips_f1':     round(float(neu_baseline), 6),
        'assistments_f1': round(float(ast_baseline), 6),
        'method':         'mean(RF_f1, LR_f1)'
    },
    'edupamnet': {
        'neurips':     {k: round(v, 6) if v is not None else None
                        for k, v in neu_m.items()},
        'assistments': {k: round(v, 6) if v is not None else None
                        for k, v in ast_m.items()},
    },
    'multi_task_effectiveness': {
        'neurips_f1_gain':     round(float(neu_gain), 6),
        'assistments_f1_gain': round(float(ast_gain), 6),
        'f1_consistency':      round(float(f1_cons), 6),
        'acc_consistency':     round(float(acc_cons), 6),
        'baseline_method':     'true_empirical',
    },
    'architecture': {
        'input_dim':                  input_dim,
        'shared_encoder_output_dim':  int(enc_dim),
        'dimension_consistent':       True,
        'neurips_decoder_params':     int(dec_params_n),
        'assistments_decoder_params': int(dec_params_a),
        'total_params':               int(total_params),
    },
    'training_config': {
        'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
        'cv_folds': CV_FOLDS, 'device': DEVICE,
        'domain_weight': 0.05, 'adversarial_weight': 0.03,
        'balance_weight': 0.10,
    }
}

with open(os.path.join(OUTPUT_PATH, 'step2_ro1_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
torch.save(model_neu.state_dict(),
           os.path.join(OUTPUT_PATH, 'model_neurips.pt'))
torch.save(model_ast.state_dict(),
           os.path.join(OUTPUT_PATH, 'model_assistments.pt'))
with open(os.path.join(OUTPUT_PATH, 'model_dims.json'), 'w') as f:
    json.dump({'input_dim': input_dim,
               'unified_features': UNIFIED_FEATURES}, f, indent=2)

print(f"\n[SAVED] results/step2_ro1_results.json + model checkpoints")

# ════════════════════════════════════════════════════════════════
# 8. 最终摘要
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("RO1 SUMMARY")
print("=" * 65)
print(f"  {'Model':32s}  {'Acc':>7}  {'F1':>7}  {'AUC':>7}")
print(f"  {'-'*55}")
for name, m in [('LR  NeurIPS', baselines['LR_neurips']),
                ('RF  NeurIPS', baselines['RF_neurips']),
                ('LR  ASSISTments', baselines['LR_assistments']),
                ('RF  ASSISTments', baselines['RF_assistments'])]:
    auc = f"{m['auc']:.4f}" if m.get('auc') else "  N/A"
    print(f"  {name:32s}  {m['accuracy']:.4f}   {m['f1']:.4f}   {auc}")
if HAS_XGB:
    for name, key in [('XGB NeurIPS', 'XGB_neurips'),
                      ('XGB ASSISTments', 'XGB_assistments')]:
        m = baselines[key]
        print(f"  {name:32s}  {m['accuracy']:.4f}   {m['f1']:.4f}   {m['auc']:.4f}")
print(f"  {'EduPAMNet NeurIPS':32s}  {neu_m['accuracy']:.4f}   "
      f"{neu_m['f1']:.4f}   {neu_m['auc']:.4f}")
print(f"  {'EduPAMNet ASSISTments':32s}  {ast_m['accuracy']:.4f}   "
      f"{ast_m['f1']:.4f}   {ast_m['auc']:.4f}")
print()
print(f"  MTL gain   NeurIPS={neu_gain:+.4f}   ASSISTments={ast_gain:+.4f}")
print(f"  Consistency F1={f1_cons:.4f}   Acc={acc_cons:.4f}")
print()
print(">>> Step 2 COMPLETE — proceed to Step 3 (RO2 Interpretability) <<<")
