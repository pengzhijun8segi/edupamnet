"""
deep_diagnose.py
深度诊断：直接检查去重后特征值与标签的关系
"""
import numpy as np
from data_loader_aligned import load_aligned_data, UNIFIED_FEATURES

X_n, y_n, X_a, y_a, dim = load_aligned_data(
    data_path='./data/', sample_size=30000, random_state=42)

print("=== NeurIPS 去重后检查 ===")
print(f"总行数: {len(X_n)}")
uniq = np.unique(X_n.round(4), axis=0)
print(f"唯一特征行数: {len(uniq)}")

# 检查特征值范围
print("\n各特征统计：")
for i, f in enumerate(UNIFIED_FEATURES):
    vals = X_n[:, i]
    print(f"  {f}: min={vals.min():.4f} max={vals.max():.4f} "
          f"unique={len(np.unique(vals.round(4)))} std={vals.std():.4f}")

# 关键：检查 student_ability 和 historical_accuracy 是否完全相同
corr = np.corrcoef(X_n[:, 0], X_n[:, 2])[0, 1]
print(f"\nstudent_ability vs historical_accuracy 相关: {corr:.6f}")
print("(如果=1.0，这两列完全相同，信息冗余)")

# 检查是否可以用单个特征完美分类
print("\n=== 单特征分类测试（NeurIPS）===")
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
for i, f in enumerate(UNIFIED_FEATURES):
    dt = DecisionTreeClassifier(max_depth=1, random_state=42)
    scores = cross_val_score(dt, X_n[:, i:i+1], y_n.astype(int),
                              cv=5, scoring='f1')
    print(f"  {f}: F1={scores.mean():.4f} (单特征深度1决策树)")

print("\n=== 用全部5特征深度1决策树 ===")
dt_all = DecisionTreeClassifier(max_depth=1, random_state=42)
scores_all = cross_val_score(dt_all, X_n, y_n.astype(int), cv=5, scoring='f1')
print(f"  全特征 depth=1: F1={scores_all.mean():.4f}")

dt_all2 = DecisionTreeClassifier(max_depth=3, random_state=42)
scores_all2 = cross_val_score(dt_all2, X_n, y_n.astype(int), cv=5, scoring='f1')
print(f"  全特征 depth=3: F1={scores_all2.mean():.4f}")

print("\n=== ASSISTments 单特征测试 ===")
for i, f in enumerate(UNIFIED_FEATURES):
    dt = DecisionTreeClassifier(max_depth=1, random_state=42)
    scores = cross_val_score(dt, X_a[:, i:i+1], y_a.astype(int),
                              cv=5, scoring='f1')
    print(f"  {f}: F1={scores.mean():.4f}")
