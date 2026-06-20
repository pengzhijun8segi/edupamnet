"""
step5_pts_computation.py
Step 5 — PTS: Pattern Transfer Score

PTS 定义（对应论文 §1.7.2）：
  评估"在平台A上学到的共享表征（shared_encoder输出），
  迁移到平台B后能保留多少预测性能"。

计算方法：
  1. 训练 model_A（NeurIPS）和 model_B（ASSISTments）
  2. 提取各自 shared_encoder 的32维表征
  3. 训练4个线性探针（Logistic Regression）：
     - native_A  : encoder_A(X_A) → y_A   （平台A原生表征）
     - native_B  : encoder_B(X_B) → y_B   （平台B原生表征）
     - transfer_A2B : encoder_A(X_B) → y_B  （A的表征用于B任务）
     - transfer_B2A : encoder_B(X_A) → y_A  （B的表征用于A任务）
  4. PTS_A2B = F1(transfer_A2B) / F1(native_B)   （A→B 性能保留率）
     PTS_B2A = F1(transfer_B2A) / F1(native_A)   （B→A 性能保留率）

  PTS=1.0 表示完全迁移（源域表征与目标域原生表征同等有效）
  PTS<1.0 表示迁移损失
  PTS>1.0 表示源域表征甚至比目标域原生表征更好（强迁移）

5种子: [42, 123, 7, 2026, 999]
输出: results/step5_pts_results.json
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
print("Step 5 — PTS: Pattern Transfer Score")
print(f"  Seeds: {SEEDS}  |  Epochs: {EPOCHS}  |  Device: {DEVICE}")
print("=" * 65)

from data_loader02 import CrossPlatformDataLoader
from data_loader_aligned import align_features, NEURIPS_MAP, ASSISTMENTS_MAP, UNIFIED_FEATURES, INPUT_DIM
from optimized_pamnet_implementation import OptimizedPAMNet
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

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

def extract_encoder_features(model, X):
    """提取 shared_encoder 输出（32维表征）"""
    model.eval()
    with torch.no_grad():
        feat = model.shared_encoder(torch.tensor(X, dtype=torch.float32).to(DEVICE))
    return feat.cpu().numpy()

def train_probe(X_repr, y, X_repr_te, y_te, seed):
    """用表征训练LR探针，返回F1"""
    lr = LogisticRegression(max_iter=500, random_state=seed)
    lr.fit(X_repr, y.astype(int))
    pred = lr.predict(X_repr_te)
    return f1_score(y_te.astype(int), pred, zero_division=0)

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

    # ── 训练两平台模型 ────────────────────────────────────────
    tr_n, val_n, X_neu_tr, y_neu_tr, X_neu_te, y_neu_te = make_loaders(X_neu, y_neu, seed=seed)
    model_neu = OptimizedPAMNet(input_dim=INPUT_DIM)
    model_neu = train_pamnet(model_neu, tr_n, val_n, 'neurips', EPOCHS, DEVICE)
    print(f"  [OK] NeurIPS model trained")

    tr_a, val_a, X_ast_tr, y_ast_tr, X_ast_te, y_ast_te = make_loaders(X_ast, y_ast, seed=seed)
    model_ast = OptimizedPAMNet(input_dim=INPUT_DIM)
    model_ast = train_pamnet(model_ast, tr_a, val_a, 'assistments', EPOCHS, DEVICE)
    print(f"  [OK] ASSISTments model trained")

    # ── 提取表征 ─────────────────────────────────────────────
    # encoder_neu 应用于 NeurIPS 数据（native）
    repr_neu_native_tr = extract_encoder_features(model_neu, X_neu_tr)
    repr_neu_native_te = extract_encoder_features(model_neu, X_neu_te)
    # encoder_ast 应用于 ASSISTments 数据（native）
    repr_ast_native_tr = extract_encoder_features(model_ast, X_ast_tr)
    repr_ast_native_te = extract_encoder_features(model_ast, X_ast_te)
    # encoder_neu 应用于 ASSISTments 数据（transfer A→B, A=NeurIPS, B=ASSISTments）
    repr_neu2ast_tr = extract_encoder_features(model_neu, X_ast_tr)
    repr_neu2ast_te = extract_encoder_features(model_neu, X_ast_te)
    # encoder_ast 应用于 NeurIPS 数据（transfer B→A, B=ASSISTments, A=NeurIPS）
    repr_ast2neu_tr = extract_encoder_features(model_ast, X_neu_tr)
    repr_ast2neu_te = extract_encoder_features(model_ast, X_neu_te)

    # ── 训练探针并评估 ───────────────────────────────────────
    f1_native_neu  = train_probe(repr_neu_native_tr, y_neu_tr, repr_neu_native_te, y_neu_te, seed)
    f1_native_ast  = train_probe(repr_ast_native_tr, y_ast_tr, repr_ast_native_te, y_ast_te, seed)
    f1_neu2ast     = train_probe(repr_neu2ast_tr,    y_ast_tr, repr_neu2ast_te,    y_ast_te, seed)
    f1_ast2neu     = train_probe(repr_ast2neu_tr,    y_neu_tr, repr_ast2neu_te,    y_neu_te, seed)

    pts_neu2ast = f1_neu2ast / f1_native_ast if f1_native_ast > 0 else 0
    pts_ast2neu = f1_ast2neu / f1_native_neu if f1_native_neu > 0 else 0

    print(f"\n  F1 native_NeurIPS     = {f1_native_neu:.4f}")
    print(f"  F1 native_ASSISTments = {f1_native_ast:.4f}")
    print(f"  F1 NeurIPS→ASSISTments(transfer) = {f1_neu2ast:.4f}")
    print(f"  F1 ASSISTments→NeurIPS(transfer) = {f1_ast2neu:.4f}")
    print(f"  PTS (NeurIPS→ASSISTments) = {pts_neu2ast:.4f}")
    print(f"  PTS (ASSISTments→NeurIPS) = {pts_ast2neu:.4f}")

    all_results.append({
        'seed': seed,
        'f1_native_neurips':     round(float(f1_native_neu), 6),
        'f1_native_assistments': round(float(f1_native_ast), 6),
        'f1_transfer_neu2ast':   round(float(f1_neu2ast), 6),
        'f1_transfer_ast2neu':   round(float(f1_ast2neu), 6),
        'pts_neu2ast':           round(float(pts_neu2ast), 6),
        'pts_ast2neu':           round(float(pts_ast2neu), 6),
    })

# ════════════════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("AGGREGATING PTS RESULTS (mean ± std across 5 seeds)")
print(f"{'='*65}")

def agg(values):
    arr = np.array(values)
    return {'mean': round(float(arr.mean()), 6),
            'std':  round(float(arr.std()), 6),
            'ci95': round(float(1.96 * arr.std() / np.sqrt(len(arr))), 6)}

pts_neu2ast_agg = agg([r['pts_neu2ast'] for r in all_results])
pts_ast2neu_agg = agg([r['pts_ast2neu'] for r in all_results])
pts_mean_agg    = agg([(r['pts_neu2ast'] + r['pts_ast2neu'])/2 for r in all_results])

f1_native_neu_agg = agg([r['f1_native_neurips']     for r in all_results])
f1_native_ast_agg = agg([r['f1_native_assistments'] for r in all_results])
f1_n2a_agg        = agg([r['f1_transfer_neu2ast']   for r in all_results])
f1_a2n_agg        = agg([r['f1_transfer_ast2neu']   for r in all_results])

print(f"\n  F1 native NeurIPS      : {f1_native_neu_agg['mean']:.4f} ± {f1_native_neu_agg['std']:.4f}")
print(f"  F1 native ASSISTments  : {f1_native_ast_agg['mean']:.4f} ± {f1_native_ast_agg['std']:.4f}")
print(f"  F1 transfer N→A        : {f1_n2a_agg['mean']:.4f} ± {f1_n2a_agg['std']:.4f}")
print(f"  F1 transfer A→N        : {f1_a2n_agg['mean']:.4f} ± {f1_a2n_agg['std']:.4f}")
print()
print(f"  PTS (NeurIPS→ASSISTments) : {pts_neu2ast_agg['mean']:.4f} ± {pts_neu2ast_agg['std']:.4f}  "
      f"(95%CI ±{pts_neu2ast_agg['ci95']:.4f})")
print(f"  PTS (ASSISTments→NeurIPS) : {pts_ast2neu_agg['mean']:.4f} ± {pts_ast2neu_agg['std']:.4f}  "
      f"(95%CI ±{pts_ast2neu_agg['ci95']:.4f})")
print(f"  PTS (overall mean)        : {pts_mean_agg['mean']:.4f} ± {pts_mean_agg['std']:.4f}  "
      f"(95%CI ±{pts_mean_agg['ci95']:.4f})")

interpretation = (
    "Strong transfer (PTS≥0.9)" if pts_mean_agg['mean'] >= 0.9 else
    "Moderate transfer (0.7≤PTS<0.9)" if pts_mean_agg['mean'] >= 0.7 else
    "Weak transfer (PTS<0.7)"
)
print(f"\n  Interpretation: {interpretation}")

# ── 保存 ─────────────────────────────────────────────────────────
final = {
    'definition': (
        "PTS = F1(probe trained on source-platform encoder representations, "
        "evaluated on target platform) / F1(probe trained on target-platform "
        "native encoder representations, evaluated on target platform)"
    ),
    'config': {'seeds': SEEDS, 'sample_size': SAMPLE_SIZE, 'epochs': EPOCHS},
    'per_seed_results': all_results,
    'aggregated': {
        'f1_native_neurips':     f1_native_neu_agg,
        'f1_native_assistments': f1_native_ast_agg,
        'f1_transfer_neu2ast':   f1_n2a_agg,
        'f1_transfer_ast2neu':   f1_a2n_agg,
        'pts_neu2ast':           pts_neu2ast_agg,
        'pts_ast2neu':           pts_ast2neu_agg,
        'pts_overall':           pts_mean_agg,
        'interpretation':        interpretation,
    }
}

out = os.path.join(OUTPUT_PATH, 'step5_pts_results.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(final, f, indent=2, ensure_ascii=False)
print(f"\n[SAVED] {out}")
print("\n>>> PTS computation complete <<<")
