"""
diagnose_features.py
诊断对齐后的特征是否仍含泄露字段
"""
import numpy as np
from data_loader_aligned import load_aligned_data, UNIFIED_FEATURES

print("=== 对齐特征列表 ===")
for i, f in enumerate(UNIFIED_FEATURES):
    print(f"  [{i}] {f}")

print("\n=== 加载少量数据检查相关性 ===")
X_neu, y_neu, X_ast, y_ast, dim = load_aligned_data(
    data_path='./data/', sample_size=1000, random_state=42)

print("\nNeurIPS 各特征与标签 y 的相关系数：")
for i, fname in enumerate(UNIFIED_FEATURES):
    corr = np.corrcoef(X_neu[:, i], y_neu)[0, 1]
    flag = " *** 高度相关，疑似泄露！" if abs(corr) > 0.8 else ""
    print(f"  [{i}] {fname:30s}: corr={corr:.4f}{flag}")

print("\nASSISTments 各特征与标签 y 的相关系数：")
for i, fname in enumerate(UNIFIED_FEATURES):
    corr = np.corrcoef(X_ast[:, i], y_ast)[0, 1]
    flag = " *** 高度相关，疑似泄露！" if abs(corr) > 0.8 else ""
    print(f"  [{i}] {fname:30s}: corr={corr:.4f}{flag}")
