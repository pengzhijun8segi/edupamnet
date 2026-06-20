"""
diagnose_duplicates.py
检查采样后数据是否存在训练集/测试集重叠（行重复）
"""
import numpy as np
from sklearn.model_selection import train_test_split
from data_loader_aligned import load_aligned_data

X_neu, y_neu, X_ast, y_ast, dim = load_aligned_data(
    data_path='./data/', sample_size=30000, random_state=42)

print("=== NeurIPS 数据检查 ===")
print(f"总行数: {len(X_neu)}")
# 检查重复行
df_neu = np.unique(X_neu, axis=0)
print(f"唯一行数: {len(df_neu)}")
print(f"重复行数: {len(X_neu) - len(df_neu)}")
print(f"重复率: {(len(X_neu) - len(df_neu)) / len(X_neu):.2%}")

# 划分训练/测试
X_tr, X_te, y_tr, y_te = train_test_split(
    X_neu, y_neu, test_size=0.2, stratify=y_neu.astype(int), random_state=42)

# 检查训练集和测试集是否有重叠行
tr_set = set(map(tuple, X_tr.round(6)))
te_set = set(map(tuple, X_te.round(6)))
overlap = tr_set & te_set
print(f"\n训练集行数: {len(X_tr)}")
print(f"测试集行数: {len(X_te)}")
print(f"训练/测试重叠行数: {len(overlap)}")
print(f"重叠率: {len(overlap)/len(X_te):.2%}")

print("\n=== ASSISTments 数据检查 ===")
print(f"总行数: {len(X_ast)}")
df_ast = np.unique(X_ast, axis=0)
print(f"唯一行数: {len(df_ast)}")
print(f"重复行数: {len(X_ast) - len(df_ast)}")
print(f"重复率: {(len(X_ast) - len(df_ast)) / len(X_ast):.2%}")

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_ast, y_ast, test_size=0.2, stratify=y_ast.astype(int), random_state=42)
tr_set2 = set(map(tuple, X_tr2.round(6)))
te_set2 = set(map(tuple, X_te2.round(6)))
overlap2 = tr_set2 & te_set2
print(f"\n训练集行数: {len(X_tr2)}")
print(f"测试集行数: {len(X_te2)}")
print(f"训练/测试重叠行数: {len(overlap2)}")
print(f"重叠率: {len(overlap2)/len(X_te2):.2%}")

print("\n=== 特征值分布检查（前5行）===")
print("NeurIPS X[:5]:")
print(X_neu[:5].round(4))
print("y[:5]:", y_neu[:5])
