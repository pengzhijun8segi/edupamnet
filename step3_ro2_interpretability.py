"""
step3_ro2_interpretability.py
Step 3 — RO2: Interpretability Framework
- 加载 Step 2 训练好的模型
- 用 RF 代理模型进行 SHAP 分析（两平台）
- 用 LIME 进行逐样本解释（两平台）
- 计算 ECI（Explanation Consistency Index）= SHAP vs LIME Spearman 相关
- 计算 FISS（Feature Importance Stability Score）= 两平台 SHAP 排名 Spearman 相关
- 保存结果供 Step 5 使用

依赖文件：
  results/model_neurips.pt
  results/model_assistments.pt
  results/model_dims.json
  results/step2_no_adv_results.json（用于加载特征名）
"""

import os, sys, json, warnings
import numpy as np
import torch
warnings.filterwarnings('ignore')

DATA_PATH    = './data/'
OUTPUT_PATH  = './results/'
RANDOM_STATE = 42
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
SHAP_SAMPLES = 200   # SHAP 背景样本数
LIME_SAMPLES = 50    # LIME 解释样本数

os.makedirs(os.path.join(OUTPUT_PATH, 'interpretability'), exist_ok=True)

print("=" * 65)
print("Step 3 — RO2: Interpretability Framework")
print(f"  SHAP background: {SHAP_SAMPLES}  |  LIME samples: {LIME_SAMPLES}")
print("=" * 65)

import shap
import lime
import lime.lime_tabular
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data_loader_aligned import load_aligned_data, UNIFIED_FEATURES, INPUT_DIM
from optimized_pamnet_implementation import OptimizedPAMNet

# ════════════════════════════════════════════════════════════════
# 1. 加载数据
# ════════════════════════════════════════════════════════════════
print("\n[1] Loading data...")
X_neu, y_neu, X_ast, y_ast, input_dim = load_aligned_data(
    data_path=DATA_PATH, sample_size=30000, random_state=RANDOM_STATE)

feature_names = UNIFIED_FEATURES
print(f"  Features ({input_dim}): {feature_names}")
print(f"  NeurIPS     : {X_neu.shape}")
print(f"  ASSISTments : {X_ast.shape}")

# ════════════════════════════════════════════════════════════════
# 2. 加载 Step 2 模型
# ════════════════════════════════════════════════════════════════
print("\n[2] Loading trained models from Step 2...")

with open(os.path.join(OUTPUT_PATH, 'model_dims.json')) as f:
    dims = json.load(f)

model_neu = OptimizedPAMNet(input_dim=dims['input_dim'])
model_neu.load_state_dict(torch.load(
    os.path.join(OUTPUT_PATH, 'model_neurips.pt'),
    map_location=DEVICE))
model_neu.to(DEVICE).eval()
print("  [OK] NeurIPS model loaded")

model_ast = OptimizedPAMNet(input_dim=dims['input_dim'])
model_ast.load_state_dict(torch.load(
    os.path.join(OUTPUT_PATH, 'model_assistments.pt'),
    map_location=DEVICE))
model_ast.to(DEVICE).eval()
print("  [OK] ASSISTments model loaded")

# ════════════════════════════════════════════════════════════════
# 3. 工具：EduPAMNet 预测函数（供 SHAP/LIME 调用）
# ════════════════════════════════════════════════════════════════
def make_predict_fn(model, platform):
    def predict_proba(X):
        X_t = torch.tensor(X.astype(np.float32)).to(DEVICE)
        with torch.no_grad():
            out  = model(X_t, platform=platform)['output']
            prob = torch.sigmoid(out).cpu().numpy().flatten()
        return np.column_stack([1 - prob, prob])
    return predict_proba

predict_neu = make_predict_fn(model_neu, 'neurips')
predict_ast = make_predict_fn(model_ast, 'assistments')

# ════════════════════════════════════════════════════════════════
# 4. 代理 RF 模型（SHAP TreeExplainer 用）
# ════════════════════════════════════════════════════════════════
print("\n[3] Training surrogate RF models for SHAP...")

def train_surrogate(X, y, predict_fn):
    """用 EduPAMNet 的软标签训练 RF 代理模型"""
    prob = predict_fn(X)[:, 1]
    soft_labels = (prob > 0.5).astype(int)
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                 n_jobs=-1)
    rf.fit(X, soft_labels)
    return rf

# 划分数据
X_tr_n, X_te_n, y_tr_n, y_te_n = train_test_split(
    X_neu, y_neu, test_size=0.2, stratify=y_neu.astype(int),
    random_state=RANDOM_STATE)
X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(
    X_ast, y_ast, test_size=0.2, stratify=y_ast.astype(int),
    random_state=RANDOM_STATE)

rf_neu = train_surrogate(X_tr_n, y_tr_n, predict_neu)
rf_ast = train_surrogate(X_tr_a, y_tr_a, predict_ast)
print("  [OK] Surrogate RF models trained")

# ════════════════════════════════════════════════════════════════
# 5. SHAP 分析
# ════════════════════════════════════════════════════════════════
print("\n[4] SHAP analysis...")

def compute_shap(rf_model, X_bg, X_test, fname):
    """计算 SHAP 值，返回特征重要性向量"""
    bg  = X_bg[:SHAP_SAMPLES]
    te  = X_test[:SHAP_SAMPLES]
    explainer   = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(te)
    # shap_values: list[2] for binary, each (N, F)
    if isinstance(shap_values, list):
        sv = shap_values[1]   # positive class
    else:
        sv = shap_values
    if sv.ndim > 2:
        sv = sv[:, :, 0]
    importance = np.abs(sv).mean(axis=0).flatten()
    print(f"  {fname} SHAP importance: " +
          " | ".join(f"{n}={float(v):.4f}" for n, v in
                     zip(feature_names, importance)))
    return importance, sv

shap_imp_neu, shap_vals_neu = compute_shap(rf_neu, X_tr_n, X_te_n, "NeurIPS")
shap_imp_ast, shap_vals_ast = compute_shap(rf_ast, X_tr_a, X_te_a, "ASSISTments")

# ════════════════════════════════════════════════════════════════
# 6. LIME 分析
# ════════════════════════════════════════════════════════════════
print("\n[5] LIME analysis...")

def compute_lime(X_train, X_test, predict_fn, fname):
    """计算 LIME 平均特征重要性"""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=['incorrect', 'correct'],
        mode='classification',
        random_state=RANDOM_STATE
    )
    lime_imp = np.zeros(len(feature_names))
    n_ok = 0
    for i in range(min(LIME_SAMPLES, len(X_test))):
        try:
            exp = explainer.explain_instance(
                X_test[i], predict_fn, num_features=len(feature_names))
            vals = dict(exp.as_list())
            for j, feat in enumerate(feature_names):
                for k, v in vals.items():
                    if feat.lower() in k.lower():
                        lime_imp[j] += abs(v)
                        break
            n_ok += 1
        except Exception as e:
            continue
    if n_ok > 0:
        lime_imp /= n_ok
    print(f"  {fname} LIME importance ({n_ok} samples): " +
          " | ".join(f"{n}={v:.4f}" for n, v in
                     zip(feature_names, lime_imp)))
    return lime_imp

lime_imp_neu = compute_lime(X_tr_n, X_te_n, predict_neu, "NeurIPS")
lime_imp_ast = compute_lime(X_tr_a, X_te_a, predict_ast, "ASSISTments")

# ════════════════════════════════════════════════════════════════
# 7. ECI — Explanation Consistency Index
#    = Spearman(SHAP_importance, LIME_importance) per platform
# ════════════════════════════════════════════════════════════════
print("\n[6] Computing ECI (Explanation Consistency Index)...")

def compute_eci(shap_imp, lime_imp, platform):
    corr, pval = spearmanr(shap_imp, lime_imp)
    corr = float(corr) if not np.isnan(corr) else 0.0
    print(f"  ECI {platform}: Spearman={corr:.4f}  p={pval:.4f}")
    return corr, float(pval)

eci_neu, eci_neu_p = compute_eci(shap_imp_neu, lime_imp_neu, "NeurIPS")
eci_ast, eci_ast_p = compute_eci(shap_imp_ast, lime_imp_ast, "ASSISTments")
eci_mean = (eci_neu + eci_ast) / 2
print(f"  ECI mean (both platforms): {eci_mean:.4f}")

# ════════════════════════════════════════════════════════════════
# 8. FISS — Feature Importance Stability Score
#    = Spearman(SHAP_NeurIPS_rank, SHAP_ASSISTments_rank)
# ════════════════════════════════════════════════════════════════
print("\n[7] Computing FISS (Feature Importance Stability Score)...")

fiss, fiss_p = spearmanr(shap_imp_neu, shap_imp_ast)
fiss = float(fiss) if not np.isnan(fiss) else 0.0
print(f"  FISS: Spearman={fiss:.4f}  p={fiss_p:.4f}")
print(f"  Interpretation: {'Stable' if fiss > 0.6 else 'Unstable'} "
      f"cross-platform feature importance")

# 特征排名对比
rank_neu = np.argsort(shap_imp_neu)[::-1]
rank_ast = np.argsort(shap_imp_ast)[::-1]
print("\n  Feature importance ranking:")
print(f"  {'Feature':25s}  NeurIPS  ASSISTments  SHAP_N   SHAP_A")
for i, fname in enumerate(feature_names):
    rn = np.where(rank_neu == i)[0][0] + 1
    ra = np.where(rank_ast == i)[0][0] + 1
    print(f"  {fname:25s}  #{rn:<7}  #{ra:<11}  "
          f"{shap_imp_neu[i]:.4f}   {shap_imp_ast[i]:.4f}")

# ════════════════════════════════════════════════════════════════
# 9. 可视化
# ════════════════════════════════════════════════════════════════
print("\n[8] Generating visualizations...")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# SHAP 对比柱状图
x = np.arange(len(feature_names))
w = 0.35
axes[0, 0].bar(x - w/2, shap_imp_neu, w, label='NeurIPS',     alpha=0.8, color='#2196F3')
axes[0, 0].bar(x + w/2, shap_imp_ast, w, label='ASSISTments', alpha=0.8, color='#FF9800')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(feature_names, rotation=30, ha='right', fontsize=9)
axes[0, 0].set_title(f'SHAP Feature Importance\n(FISS={fiss:.3f})')
axes[0, 0].set_ylabel('Mean |SHAP|')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# SHAP vs LIME 散点（NeurIPS）
axes[0, 1].scatter(shap_imp_neu, lime_imp_neu, color='#2196F3', s=80, zorder=3)
mx = max(shap_imp_neu.max(), lime_imp_neu.max()) * 1.1
axes[0, 1].plot([0, mx], [0, mx], 'r--', alpha=0.5)
for i, fn in enumerate(feature_names):
    axes[0, 1].annotate(fn, (shap_imp_neu[i], lime_imp_neu[i]),
                         fontsize=8, alpha=0.8)
axes[0, 1].set_xlabel('SHAP Importance')
axes[0, 1].set_ylabel('LIME Importance')
axes[0, 1].set_title(f'SHAP vs LIME — NeurIPS\n(ECI={eci_neu:.3f})')
axes[0, 1].grid(alpha=0.3)

# LIME 对比柱状图
axes[1, 0].bar(x - w/2, lime_imp_neu, w, label='NeurIPS',     alpha=0.8, color='#4CAF50')
axes[1, 0].bar(x + w/2, lime_imp_ast, w, label='ASSISTments', alpha=0.8, color='#F44336')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(feature_names, rotation=30, ha='right', fontsize=9)
axes[1, 0].set_title('LIME Feature Importance')
axes[1, 0].set_ylabel('Mean |LIME weight|')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# SHAP vs LIME 散点（ASSISTments）
axes[1, 1].scatter(shap_imp_ast, lime_imp_ast, color='#FF9800', s=80, zorder=3)
mx2 = max(shap_imp_ast.max(), lime_imp_ast.max()) * 1.1
axes[1, 1].plot([0, mx2], [0, mx2], 'r--', alpha=0.5)
for i, fn in enumerate(feature_names):
    axes[1, 1].annotate(fn, (shap_imp_ast[i], lime_imp_ast[i]),
                         fontsize=8, alpha=0.8)
axes[1, 1].set_xlabel('SHAP Importance')
axes[1, 1].set_ylabel('LIME Importance')
axes[1, 1].set_title(f'SHAP vs LIME — ASSISTments\n(ECI={eci_ast:.3f})')
axes[1, 1].grid(alpha=0.3)

plt.suptitle('RO2: Interpretability Framework Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
fig_path = os.path.join(OUTPUT_PATH, 'interpretability', 'ro2_interpretability.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [SAVED] {fig_path}")

# ════════════════════════════════════════════════════════════════
# 10. 保存结果
# ════════════════════════════════════════════════════════════════
results = {
    'shap': {
        'neurips':     {'importance': shap_imp_neu.tolist(),
                        'feature_names': feature_names},
        'assistments': {'importance': shap_imp_ast.tolist(),
                        'feature_names': feature_names},
    },
    'lime': {
        'neurips':     {'importance': lime_imp_neu.tolist()},
        'assistments': {'importance': lime_imp_ast.tolist()},
    },
    'eci': {
        'neurips':     {'spearman': round(eci_neu, 6), 'p_value': round(eci_neu_p, 6)},
        'assistments': {'spearman': round(eci_ast, 6), 'p_value': round(eci_ast_p, 6)},
        'mean':        round(eci_mean, 6),
        'definition':  'Spearman(SHAP_importance, LIME_importance) per platform'
    },
    'fiss': {
        'spearman':    round(fiss, 6),
        'p_value':     round(float(fiss_p), 6),
        'stable':      fiss > 0.6,
        'definition':  'Spearman(SHAP_NeurIPS_rank, SHAP_ASSISTments_rank)'
    },
    'feature_ranking': {
        'neurips':     [feature_names[i] for i in rank_neu],
        'assistments': [feature_names[i] for i in rank_ast],
    }
}

out_json = os.path.join(OUTPUT_PATH, 'step3_ro2_results.json')
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n[SAVED] {out_json}")

# ════════════════════════════════════════════════════════════════
# 11. 摘要
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("RO2 SUMMARY")
print("=" * 65)
print(f"\n  FISS (cross-platform stability) : {fiss:.4f}  "
      f"({'✓ Stable' if fiss > 0.6 else '△ Moderate' if fiss > 0.3 else '✗ Unstable'})")
print(f"  ECI  NeurIPS                    : {eci_neu:.4f}")
print(f"  ECI  ASSISTments                : {eci_ast:.4f}")
print(f"  ECI  Mean                       : {eci_mean:.4f}  "
      f"({'✓ Consistent' if eci_mean > 0.6 else '△ Moderate' if eci_mean > 0.3 else '✗ Inconsistent'})")

print(f"\n  Top feature (NeurIPS)    : {feature_names[rank_neu[0]]}")
print(f"  Top feature (ASSISTments): {feature_names[rank_ast[0]]}")

print()
print(">>> Step 3 COMPLETE — proceed to Step 4 (RO3 Task Correlation) <<<")
