"""
step1_ro4_verify.py  (v3 — 语义对齐版)
Step 1: RO4 Environment Verification
- 使用 data_loader_aligned.py 对齐两平台到统一 8 维特征
- 分层采样 30,000 行
- 验证 OptimizedPAMNet 训练管道正常
- 验证 RO4 鲁棒性 + 公平性计算管道正常
"""

import os, sys
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

DATA_PATH    = './data/'
SAMPLE_SIZE  = 3000
RANDOM_STATE = 42
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 65)
print("Step 1: RO4 Environment Verification  (v3 aligned)")
print(f"  Device      : {DEVICE}")
print(f"  Sample size : {SAMPLE_SIZE:,} per dataset (stratified)")
print("=" * 65)

# ── 1. 模块导入 ─────────────────────────────────────────────────
print("\n[1] Checking module imports...")
try:
    from data_loader_aligned import load_aligned_data, INPUT_DIM
    print("  [OK] data_loader_aligned")
except Exception as e:
    print(f"  [FAIL] data_loader_aligned: {e}"); sys.exit(1)

try:
    from optimized_pamnet_implementation import OptimizedPAMNet, train_optimized_pamnet
    print("  [OK] optimized_pamnet_implementation")
except Exception as e:
    print(f"  [FAIL] optimized_pamnet_implementation: {e}"); sys.exit(1)

try:
    from pamnet_fairness_robustness_analysisv2 import IntegratedFairnessAnalyzer
    print("  [OK] pamnet_fairness_robustness_analysisv2")
except Exception as e:
    print(f"  [FAIL] pamnet_fairness_robustness_analysisv2: {e}"); sys.exit(1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

# ── 2. 加载对齐数据 ─────────────────────────────────────────────
print("\n[2] Loading & aligning datasets...")
X_neu, y_neu, X_ast, y_ast, input_dim = load_aligned_data(
    data_path=DATA_PATH, sample_size=SAMPLE_SIZE,
    random_state=RANDOM_STATE)
print(f"\n  input_dim = {input_dim}  ← 两平台统一维度")

# ── 3. 快速训练（3 epoch）──────────────────────────────────────
print("\n[3] Quick training (3 epochs — pipeline check only)...")

def make_loaders(X, y, X_src, y_src, batch=256):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2,
        stratify=y.astype(int), random_state=RANDOM_STATE)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2,
        stratify=y_tr.astype(int), random_state=RANDOM_STATE)
    train_l = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=batch, shuffle=True)
    val_l = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=batch)
    src_l = DataLoader(
        TensorDataset(torch.tensor(X_src), torch.tensor(y_src)),
        batch_size=batch, shuffle=True)
    return train_l, val_l, src_l, X_te, y_te

train_l, val_l, src_l, X_neu_te, y_neu_te = make_loaders(
    X_neu, y_neu, X_ast, y_ast)
model_neu = OptimizedPAMNet(input_dim=input_dim)
model_neu = train_optimized_pamnet(
    model_neu, train_l, val_l, src_l,
    num_epochs=3, device=DEVICE, target_platform='neurips')
print("  [OK] NeurIPS model trained")

train_l2, val_l2, src_l2, X_ast_te, y_ast_te = make_loaders(
    X_ast, y_ast, X_neu, y_neu)
model_ast = OptimizedPAMNet(input_dim=input_dim)
model_ast = train_optimized_pamnet(
    model_ast, train_l2, val_l2, src_l2,
    num_epochs=3, device=DEVICE, target_platform='assistments')
print("  [OK] ASSISTments model trained")

# ── 4. RO4 鲁棒性测试 ───────────────────────────────────────────
print("\n[4] RO4 — Robustness testing...")

def test_robustness(model, X_te, y_te, platform,
                    noise_levels=[0.01, 0.05, 0.1, 0.2]):
    model.eval()
    results = {}
    with torch.no_grad():
        out  = model(torch.tensor(X_te).to(DEVICE), platform=platform)['output']
        prob = torch.sigmoid(out).cpu().numpy()
        pred = (prob > 0.5).astype(int)
    base_acc = accuracy_score(y_te.astype(int), pred)
    base_f1  = f1_score(y_te.astype(int), pred, zero_division=0)
    results['baseline'] = {'accuracy': base_acc, 'f1': base_f1}
    for nl in noise_levels:
        accs = []
        for _ in range(3):
            Xn = X_te + np.random.normal(0, nl, X_te.shape).astype(np.float32)
            with torch.no_grad():
                out  = model(torch.tensor(Xn).to(DEVICE), platform=platform)['output']
                prob = torch.sigmoid(out).cpu().numpy()
                accs.append(accuracy_score(y_te.astype(int),
                                           (prob > 0.5).astype(int)))
        results[f'noise_{nl}'] = {
            'accuracy': float(np.mean(accs)),
            'drop':     float(base_acc - np.mean(accs))
        }
    return results

rob_n = test_robustness(model_neu, X_neu_te, y_neu_te, 'neurips')
rob_a = test_robustness(model_ast, X_ast_te, y_ast_te, 'assistments')

print(f"  NeurIPS     baseline Acc={rob_n['baseline']['accuracy']:.4f}  "
      f"F1={rob_n['baseline']['f1']:.4f}")
for k, v in rob_n.items():
    if k != 'baseline':
        print(f"    {k}: Acc={v['accuracy']:.4f}  drop={v['drop']:+.4f}")

print(f"  ASSISTments baseline Acc={rob_a['baseline']['accuracy']:.4f}  "
      f"F1={rob_a['baseline']['f1']:.4f}")
for k, v in rob_a.items():
    if k != 'baseline':
        print(f"    {k}: Acc={v['accuracy']:.4f}  drop={v['drop']:+.4f}")

# ── 5. RO4 公平性测试 ───────────────────────────────────────────
print("\n[5] RO4 — Fairness testing...")

def test_fairness(model, X_te, y_te, platform, n_groups=3):
    model.eval()
    with torch.no_grad():
        out  = model(torch.tensor(X_te).to(DEVICE), platform=platform)['output']
        prob = torch.sigmoid(out).cpu().numpy()
        pred = (prob > 0.5).astype(int)
    # 按 student_ability（第0列，对齐后固定位置）分组
    perf   = X_te[:, 0]
    bounds = np.percentile(perf, np.linspace(0, 100, n_groups + 1))
    groups = {}
    for i in range(n_groups):
        mask = (perf >= bounds[i]) & (perf < bounds[i + 1])
        if mask.sum() < 10:
            continue
        groups[f'group_{i+1}'] = {
            'size':          int(mask.sum()),
            'accuracy':      float(accuracy_score(y_te[mask].astype(int), pred[mask])),
            'f1':            float(f1_score(y_te[mask].astype(int),
                                            pred[mask], zero_division=0)),
            'positive_rate': float(pred[mask].mean())
        }
    pos_rates = [v['positive_rate'] for v in groups.values()]
    f1_scores  = [v['f1']            for v in groups.values()]
    dp_diff    = max(pos_rates) - min(pos_rates) if pos_rates else 0
    f1_diff    = max(f1_scores) - min(f1_scores) if f1_scores else 0
    return groups, dp_diff, f1_diff

grp_n, dp_n, f1d_n = test_fairness(model_neu, X_neu_te, y_neu_te, 'neurips')
grp_a, dp_a, f1d_a = test_fairness(model_ast, X_ast_te, y_ast_te, 'assistments')

print(f"  NeurIPS     DP_diff={dp_n:.4f}  F1_diff={f1d_n:.4f}")
for g, v in grp_n.items():
    print(f"    {g}: n={v['size']}  acc={v['accuracy']:.4f}  f1={v['f1']:.4f}")
print(f"  ASSISTments DP_diff={dp_a:.4f}  F1_diff={f1d_a:.4f}")
for g, v in grp_a.items():
    print(f"    {g}: n={v['size']}  acc={v['accuracy']:.4f}  f1={v['f1']:.4f}")

# ── 6. 综合判断 ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
checks = {
    "Module imports":                        True,
    f"NeurIPS loaded & aligned (dim={input_dim})":     X_neu.shape[1] == input_dim,
    f"ASSISTments loaded & aligned (dim={input_dim})": X_ast.shape[1] == input_dim,
    f"NeurIPS sampled {SAMPLE_SIZE:,}":      X_neu.shape[0] == SAMPLE_SIZE,
    f"ASSISTments sampled {SAMPLE_SIZE:,}":  X_ast.shape[0] == SAMPLE_SIZE,
    "NeurIPS model trained":                 rob_n['baseline']['accuracy'] > 0.4,
    "ASSISTments model trained":             rob_a['baseline']['accuracy'] > 0.4,
    "Robustness pipeline OK":                True,
    "Fairness pipeline OK":                  True,
}
all_pass = all(checks.values())
for k, v in checks.items():
    print(f"  {'[PASS]' if v else '[FAIL]'}  {k}")

print()
if all_pass:
    print(">>> Step 1 PASSED — ready for Step 2 (RO1 full training) <<<")
else:
    print(">>> Some checks FAILED — fix before proceeding <<<")
    sys.exit(1)
