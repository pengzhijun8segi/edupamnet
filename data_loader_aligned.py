"""
data_loader_aligned.py  (v2)
跨平台语义特征对齐加载器

修复：
1. 去除 NeurIPS 中因数据截断导致无效的 confidence/time_of_day/day_of_week
2. 统一使用 5 个在两平台均有效的核心特征
3. 去重后再采样，避免 RF/XGBoost 记忆重复行

统一特征（5维）:
  student_ability     — 学生历史答题能力
  question_difficulty — 题目历史难度
  historical_accuracy — 学生累计正确率
  streak_correct      — 连续答对次数
  streak_incorrect    — 连续答错次数
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# NeurIPS 列名 → 通用名
NEURIPS_MAP = {
    'StudentAbility':     'student_ability',
    'QuestionDifficulty': 'question_difficulty',
    'HistoricalAccuracy': 'historical_accuracy',
    'StreakCorrect':      'streak_correct',
    'StreakIncorrect':    'streak_incorrect',
}

# ASSISTments 列名 → 通用名
ASSISTMENTS_MAP = {
    'user_ability':       'student_ability',
    'problem_difficulty': 'question_difficulty',
    'historical_accuracy':'historical_accuracy',
    'streak_correct':     'streak_correct',
    'streak_incorrect':   'streak_incorrect',
}

# 统一特征顺序
UNIFIED_FEATURES = [
    'student_ability',
    'question_difficulty',
    'historical_accuracy',
    'streak_correct',
    'streak_incorrect',
]
INPUT_DIM = len(UNIFIED_FEATURES)  # = 5


def align_features(X, feature_names, col_map):
    """按语义映射抽取统一特征矩阵"""
    name_to_idx    = {name: i for i, name in enumerate(feature_names)}
    unified_to_orig = {v: k for k, v in col_map.items()}

    result = np.zeros((len(X), INPUT_DIM), dtype=np.float32)
    for j, uni_name in enumerate(UNIFIED_FEATURES):
        orig_name = unified_to_orig.get(uni_name)
        if orig_name and orig_name in name_to_idx:
            result[:, j] = X[:, name_to_idx[orig_name]]
    return result


def load_aligned_data(data_path='./data/', sample_size=30000,
                      random_state=42):
    """
    加载、对齐、去重、采样两平台数据。

    返回:
        X_neu, y_neu  — NeurIPS   (sample_size, 5)
        X_ast, y_ast  — ASSISTments (sample_size, 5)
        INPUT_DIM     — 5
    """
    from data_loader02 import CrossPlatformDataLoader
    loader = CrossPlatformDataLoader(data_path=data_path)

    # ── 加载原始数据 ─────────────────────────────────────────
    print("[DATA] Loading NeurIPS...")
    X_neu_raw, y_neu_raw, _, feat_neu = loader.load_neurips_dataset(
        task='correctness')
    print(f"       NeurIPS full: {X_neu_raw.shape}")

    print("[DATA] Loading ASSISTments...")
    X_ast_raw, y_ast_raw, _, feat_ast = loader.load_assistments_dataset(
        task='correctness')
    print(f"       ASSISTments full: {X_ast_raw.shape}")

    # ── 语义对齐 ─────────────────────────────────────────────
    X_neu_al = align_features(X_neu_raw, feat_neu, NEURIPS_MAP)
    X_ast_al = align_features(X_ast_raw, feat_ast, ASSISTMENTS_MAP)
    print(f"[ALIGN] NeurIPS aligned   : {X_neu_al.shape}")
    print(f"[ALIGN] ASSISTments aligned: {X_ast_al.shape}")
    print(f"[ALIGN] Unified features ({INPUT_DIM}): {UNIFIED_FEATURES}")

    # ── 去重（按特征+标签联合去重）──────────────────────────
    def dedup(X, y):
        combined = np.hstack([X, y.reshape(-1, 1)])
        _, idx   = np.unique(combined.round(6), axis=0, return_index=True)
        return X[idx], y[idx]

    X_neu_dd, y_neu_dd = dedup(X_neu_al, y_neu_raw.astype(np.float32))
    X_ast_dd, y_ast_dd = dedup(X_ast_al, y_ast_raw.astype(np.float32))
    print(f"[DEDUP] NeurIPS    : {X_neu_al.shape[0]:,} → {X_neu_dd.shape[0]:,} unique rows")
    print(f"[DEDUP] ASSISTments: {X_ast_al.shape[0]:,} → {X_ast_dd.shape[0]:,} unique rows")

    # ── 分层采样 ─────────────────────────────────────────────
    def stratified_sample(X, y, n):
        if len(X) <= n:
            print(f"    Total {len(X):,} <= {n:,}, using full deduplicated set")
            return X, y
        _, Xs, _, ys = train_test_split(
            X, y, test_size=n / len(X),
            stratify=y.astype(int), random_state=random_state)
        return Xs, ys

    X_neu_s, y_neu_s = stratified_sample(X_neu_dd, y_neu_dd, sample_size)
    X_ast_s, y_ast_s = stratified_sample(X_ast_dd, y_ast_dd, sample_size)
    print(f"[SAMPLE] NeurIPS sampled    : {X_neu_s.shape}  balance={y_neu_s.mean():.3f}")
    print(f"[SAMPLE] ASSISTments sampled: {X_ast_s.shape}  balance={y_ast_s.mean():.3f}")

    # ── 标准化 ───────────────────────────────────────────────
    scaler_n = StandardScaler()
    scaler_a = StandardScaler()
    X_neu_sc = scaler_n.fit_transform(X_neu_s).astype(np.float32)
    X_ast_sc = scaler_a.fit_transform(X_ast_s).astype(np.float32)

    print(f"[OK] Final — NeurIPS: {X_neu_sc.shape}  "
          f"ASSISTments: {X_ast_sc.shape}  input_dim={INPUT_DIM}")

    return X_neu_sc, y_neu_s, X_ast_sc, y_ast_s, INPUT_DIM


if __name__ == '__main__':
    X_n, y_n, X_a, y_a, dim = load_aligned_data()
    print(f"\n[VERIFY] input_dim={dim}")
    assert X_n.shape[1] == X_a.shape[1] == dim
    print("[PASS] 维度对齐验证通过")

    # 快速检查重复率
    import numpy as np
    uniq_n = len(np.unique(X_n.round(6), axis=0))
    uniq_a = len(np.unique(X_a.round(6), axis=0))
    print(f"[CHECK] NeurIPS    唯一行: {uniq_n}/{len(X_n)} ({uniq_n/len(X_n):.1%})")
    print(f"[CHECK] ASSISTments 唯一行: {uniq_a}/{len(X_a)} ({uniq_a/len(X_a):.1%})")
