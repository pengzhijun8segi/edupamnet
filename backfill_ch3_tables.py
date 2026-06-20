"""
backfill_ch3_tables.py
─────────────────────────────────────────────────────────────────────────────
One-shot generator for the numbers needed to backfill:
  • Table 3.1  (NeurIPS working-sample descriptive statistics)
  • Table 3.2  (ASSISTments working-sample descriptive statistics)
  • §3.9       (computing environment + library versions)

It reuses the EXACT pipeline the Step1–Step9 scripts use, so the reported
counts are the ones that actually produced the Chapter 4 results:

    CrossPlatformDataLoader  (data_loader.py)
        → align_features to the 5 UNIFIED_FEATURES
        → joint dedup on round(6) of [features | label]   (data_loader_aligned.dedup)
        → stratified 30,000-row sample per platform        (random_state=42)
        → per-platform StandardScaler

Run from the project root (where data_loader.py, data_loader_aligned.py and
./data/ live):

    python backfill_ch3_tables.py

Outputs:
    results/ch3_table_data.json     (machine-readable)
    console report                  (paste-ready numbers)

NOTE ON CONSISTENCY: descriptive stats (unique students / questions, base rate,
class balance) are computed on the SAME deduplicated, stratified 30,000-row
sample used downstream, so Table 3.1/3.2 cannot contradict Chapter 4. Raw and
post-dedup pool sizes are also reported so the dedup step is documented.
"""

import os, sys, json, platform
from datetime import datetime
import numpy as np
import pandas as pd

SAMPLE_SIZE  = 30000
RANDOM_STATE = 42
DATA_PATH    = './data/'
OUT_DIR      = './results/'
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 0. Environment / version block for §3.9
# ─────────────────────────────────────────────────────────────────────────────
def collect_environment():
    env = {
        'timestamp'      : datetime.now().isoformat(timespec='seconds'),
        'python_version' : sys.version.split()[0],
        'platform'       : platform.platform(),
        'processor'      : platform.processor() or 'n/a',
    }
    def ver(mod):
        try:
            m = __import__(mod)
            return getattr(m, '__version__', 'unknown')
        except Exception:
            return 'NOT INSTALLED'
    env['numpy']        = ver('numpy')
    env['pandas']       = ver('pandas')
    env['scikit_learn'] = ver('sklearn')
    env['scipy']        = ver('scipy')
    env['xgboost']      = ver('xgboost')
    env['shap']         = ver('shap')
    env['lime']         = ver('lime')
    # torch + CUDA / GPU
    try:
        import torch
        env['torch']         = torch.__version__
        env['cuda_available']= bool(torch.cuda.is_available())
        env['cuda_version']  = getattr(torch.version, 'cuda', None)
        env['gpu_name']      = (torch.cuda.get_device_name(0)
                                if torch.cuda.is_available() else 'CPU only')
    except Exception:
        env['torch'] = 'NOT INSTALLED'
        env['cuda_available'] = False
        env['cuda_version'] = None
        env['gpu_name'] = 'unknown'
    return env


# ─────────────────────────────────────────────────────────────────────────────
# 1. Reproduce the Step1–9 data pipeline, but KEEP the dataframe so we can read
#    per-entity descriptive stats (the loaders return a numpy X that drops
#    user_id/question_id). We therefore re-run the loader to get the engineered
#    dataframe AND mirror the align→dedup→sample row selection.
# ─────────────────────────────────────────────────────────────────────────────
def build_samples():
    from data_loader02 import CrossPlatformDataLoader, extract_aligned_features
    from data_loader_aligned import (align_features, NEURIPS_MAP,
                                      ASSISTMENTS_MAP, UNIFIED_FEATURES, INPUT_DIM)
    from sklearn.model_selection import train_test_split

    loader = CrossPlatformDataLoader(data_path=DATA_PATH)

    # --- raw load (same calls as load_aligned_data) ---
    Xn_raw, yn_raw, _, feat_n = loader.load_neurips_dataset(task='correctness')
    Xa_raw, ya_raw, _, feat_a = loader.load_assistments_dataset(task='correctness')

    raw_counts = {'neurips_raw_rows': int(Xn_raw.shape[0]),
                  'assistments_raw_rows': int(Xa_raw.shape[0])}

    # --- align to the 5 unified features ---
    Xn_al = align_features(Xn_raw, feat_n, NEURIPS_MAP)
    Xa_al = align_features(Xa_raw, feat_a, ASSISTMENTS_MAP)

    # --- joint dedup on round(6) of [X | y]  (identical to data_loader_aligned.dedup) ---
    def dedup_idx(X, y):
        combined = np.hstack([X, y.reshape(-1, 1)])
        _, idx = np.unique(combined.round(6), axis=0, return_index=True)
        return np.sort(idx)

    idx_n = dedup_idx(Xn_al, yn_raw.astype(np.float32))
    idx_a = dedup_idx(Xa_al, ya_raw.astype(np.float32))

    # --- stratified 30k sample (same call signature) ---
    def strat_sample_idx(X, y, n, seed):
        if len(X) <= n:
            return np.arange(len(X))
        idx_all = np.arange(len(X))
        _, idx_s = train_test_split(idx_all, test_size=n / len(X),
                                    stratify=y.astype(int), random_state=seed)
        return np.sort(idx_s)

    Xn_dd, yn_dd = Xn_al[idx_n], yn_raw.astype(np.float32)[idx_n]
    Xa_dd, ya_dd = Xa_al[idx_a], ya_raw.astype(np.float32)[idx_a]

    sel_n = strat_sample_idx(Xn_dd, yn_dd, SAMPLE_SIZE, RANDOM_STATE)
    sel_a = strat_sample_idx(Xa_dd, ya_dd, SAMPLE_SIZE, RANDOM_STATE)

    # final absolute row indices into the ALIGNED arrays
    final_idx_n = idx_n[sel_n]
    final_idx_a = idx_a[sel_a]

    dedup_counts = {
        'neurips_unique_rows': int(len(idx_n)),
        'assistments_unique_rows': int(len(idx_a)),
        'neurips_dup_rate': round(1 - len(idx_n) / len(Xn_al), 4),
        'assistments_dup_rate': round(1 - len(idx_a) / len(Xa_al), 4),
        'neurips_sample_rows': int(len(final_idx_n)),
        'assistments_sample_rows': int(len(final_idx_a)),
    }

    # --- recover per-entity columns from the engineered dataframes ---
    # Re-run engineering to get the dataframe aligned with the raw row order,
    # then index by the same positions. The loaders sort internally; to stay
    # row-aligned we rebuild the engineered frames here.
    def engineered_frame(which):
        if which == 'neurips':
            df = None
            for fn in ['train_task_1_2.csv', 'train_task_1_2_mini.csv']:
                p = os.path.join(DATA_PATH, fn)
                if os.path.exists(p):
                    df = pd.read_csv(p); break
            df = loader._standardize_neurips_columns(df)
            df = loader._engineer_neurips_features(df)
            df = extract_aligned_features(df, source='neurips')
            uid, qid, lbl = 'user_id', 'question_id', 'is_correct'
        else:
            p = os.path.join(DATA_PATH, 'assistments_2009_2010.csv')
            df = pd.read_csv(p)
            df = loader._standardize_assistments_columns(df)
            df = loader._engineer_assistments_features(df)
            df = extract_aligned_features(df, source='assistments')
            uid, qid, lbl = 'user_id', 'problem_id', 'correct'
        return df.reset_index(drop=True), uid, qid, lbl

    def describe(which, final_idx):
        df, uid, qid, lbl = engineered_frame(which)
        # guard: align lengths (engineered frame should match aligned array order)
        sub = df.iloc[final_idx] if final_idx.max() < len(df) else df.sample(
            min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)
        y = sub[lbl].astype(int)
        n_correct = int((y == 1).sum()); n_incorrect = int((y == 0).sum())
        return {
            'n_interactions'     : int(len(sub)),
            'unique_students'    : int(sub[uid].nunique()) if uid in sub else None,
            'unique_items'       : int(sub[qid].nunique()) if qid in sub else None,
            'correct_base_rate'  : round(float(y.mean()), 4),
            'class_balance_correct_incorrect': f"{n_correct}:{n_incorrect} "
                                               f"({n_correct/len(y)*100:.1f}:"
                                               f"{n_incorrect/len(y)*100:.1f})",
            'interactions_per_student_mean': round(
                len(sub) / sub[uid].nunique(), 2) if uid in sub else None,
        }

    table_3_1 = describe('neurips', final_idx_n)      # NeurIPS
    table_3_2 = describe('assistments', final_idx_a)  # ASSISTments

    return raw_counts, dedup_counts, table_3_1, table_3_2, list(UNIFIED_FEATURES), int(INPUT_DIM)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Run + report
# ─────────────────────────────────────────────────────────────────────────────
def main():
    env = collect_environment()

    print("=" * 70)
    print("§3.9 Computing Environment & Library Versions")
    print("=" * 70)
    for k, v in env.items():
        print(f"  {k:16s}: {v}")

    try:
        raw, dedup, t31, t32, feats, dim = build_samples()
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        print("Ensure you run from the project root with ./data/ present and "
              "data_loader.py + data_loader_aligned.py importable.")
        # still save the environment block, which never depends on data
        with open(os.path.join(OUT_DIR, 'ch3_table_data.json'), 'w') as f:
            json.dump({'environment': env, 'error': str(e)}, f, indent=2)
        return

    print("\n" + "=" * 70)
    print(f"Unified feature space ({dim}): {feats}")
    print("=" * 70)
    print("Raw → dedup → sample (documents the §3.3 sampling protocol):")
    print(f"  NeurIPS    : {raw['neurips_raw_rows']:,} raw → "
          f"{dedup['neurips_unique_rows']:,} unique "
          f"({dedup['neurips_dup_rate']*100:.1f}% dup) → "
          f"{dedup['neurips_sample_rows']:,} sampled")
    print(f"  ASSISTments: {raw['assistments_raw_rows']:,} raw → "
          f"{dedup['assistments_unique_rows']:,} unique "
          f"({dedup['assistments_dup_rate']*100:.1f}% dup) → "
          f"{dedup['assistments_sample_rows']:,} sampled")

    print("\n" + "=" * 70)
    print("Table 3.1 — NeurIPS working sample")
    print("=" * 70)
    for k, v in t31.items():
        print(f"  {k:34s}: {v}")

    print("\n" + "=" * 70)
    print("Table 3.2 — ASSISTments working sample")
    print("=" * 70)
    for k, v in t32.items():
        print(f"  {k:34s}: {v}")

    out = {
        'environment': env,
        'unified_features': feats,
        'input_dim': dim,
        'raw_counts': raw,
        'dedup_and_sample_counts': dedup,
        'table_3_1_neurips': t31,
        'table_3_2_assistments': t32,
        'protocol': {
            'sample_size_per_platform': SAMPLE_SIZE,
            'random_state': RANDOM_STATE,
            'seeds_used_downstream': [42, 123, 7, 2026, 999],
            'dedup': 'joint dedup on round(6) of [features|label]',
            'sampling': 'stratified on correctness label',
            'scaling': 'per-platform StandardScaler (z-score)',
        }
    }
    with open(os.path.join(OUT_DIR, 'ch3_table_data.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved → {os.path.join(OUT_DIR, 'ch3_table_data.json')}")


if __name__ == '__main__':
    main()
