"""
backfill_ch3_tables_v2.py
─────────────────────────────────────────────────────────────────────────────
Robust one-shot generator for Table 3.1 / 3.2 (and re-prints the §3.9 env block).

Why v2: the previous version called loader internals (_standardize_neurips_columns)
that don't exist in your current data_loader.py. This version touches NO private
methods. It gets every number two robust ways:

  (1) Dataset-level descriptive stats  → direct pandas read of the raw CSVs
      (unique students, unique questions/problems, raw correct rate, #interactions)

  (2) Working-sample stats             → the SAME public load_aligned_data()
      your Step1–9 scripts use. We capture its own "[DEDUP] … → … unique rows"
      and "[SAMPLE] … balance=…" printouts, and use the returned y arrays for
      the 30k-sample base rate / class balance. No re-implementation of feature
      engineering, so it cannot drift from the real pipeline.

Run from project root (./data/ present, data_loader_aligned.py importable):

    python backfill_ch3_tables_v2.py

Outputs: results/ch3_table_data.json  +  paste-ready console report.
"""

import os, sys, re, json, io, platform, contextlib
from datetime import datetime
import numpy as np
import pandas as pd

SAMPLE_SIZE  = 30000
RANDOM_STATE = 42
DATA_PATH    = './data/'
OUT_DIR      = './results/'
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 0. §3.9 environment block
# ─────────────────────────────────────────────────────────────────────────────
def collect_environment():
    env = {'timestamp': datetime.now().isoformat(timespec='seconds'),
           'python_version': sys.version.split()[0],
           'platform': platform.platform(),
           'processor': platform.processor() or 'n/a'}
    def ver(mod):
        try:
            return getattr(__import__(mod), '__version__', 'unknown')
        except Exception:
            return 'NOT INSTALLED'
    for m in ['numpy', 'pandas', 'sklearn', 'scipy', 'xgboost', 'shap', 'lime']:
        env[m if m != 'sklearn' else 'scikit_learn'] = ver(m)
    try:
        import torch
        env.update(torch=torch.__version__,
                   cuda_available=bool(torch.cuda.is_available()),
                   cuda_version=getattr(torch.version, 'cuda', None),
                   gpu_name=(torch.cuda.get_device_name(0)
                             if torch.cuda.is_available() else 'CPU only'))
    except Exception:
        env.update(torch='NOT INSTALLED', cuda_available=False,
                   cuda_version=None, gpu_name='unknown')
    return env


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset-level stats from RAW csv (no loader internals)
# ─────────────────────────────────────────────────────────────────────────────
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def raw_stats_neurips():
    df = None
    for fn in ['train_task_1_2.csv', 'train_task_1_2_mini.csv']:
        p = os.path.join(DATA_PATH, fn)
        if os.path.exists(p):
            df = pd.read_csv(p); used = fn; break
    if df is None:
        return {'error': 'NeurIPS train file not found'}
    uid = find_col(df, ['UserId', 'user_id'])
    qid = find_col(df, ['QuestionId', 'question_id'])
    lbl = find_col(df, ['IsCorrect', 'is_correct'])
    return {
        'source_file': used,
        'raw_interactions': int(len(df)),
        'unique_students': int(df[uid].nunique()) if uid else None,
        'unique_questions': int(df[qid].nunique()) if qid else None,
        'raw_correct_rate': round(float(df[lbl].mean()), 4) if lbl else None,
        'interactions_per_student_mean': round(len(df)/df[uid].nunique(), 2) if uid else None,
    }

def raw_stats_assistments():
    p = os.path.join(DATA_PATH, 'assistments_2009_2010.csv')
    if not os.path.exists(p):
        return {'error': 'assistments_2009_2010.csv not found'}
    df = pd.read_csv(p)
    uid = find_col(df, ['user_id', 'UserId'])
    pid = find_col(df, ['problem_id', 'ProblemId'])
    lbl = find_col(df, ['correct', 'IsCorrect', 'is_correct'])
    return {
        'source_file': 'assistments_2009_2010.csv',
        'raw_interactions': int(len(df)),
        'unique_students': int(df[uid].nunique()) if uid else None,
        'unique_problems': int(df[pid].nunique()) if pid else None,
        'raw_correct_rate': round(float(df[lbl].mean()), 4) if lbl else None,
        'interactions_per_student_mean': round(len(df)/df[uid].nunique(), 2) if uid else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Working-sample stats via the REAL public pipeline (stdout capture + y arrays)
# ─────────────────────────────────────────────────────────────────────────────
def working_sample_stats():
    from data_loader_aligned import load_aligned_data, UNIFIED_FEATURES, INPUT_DIM

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        Xn, yn, Xa, ya, dim = load_aligned_data(
            data_path=DATA_PATH, sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE)
    log = buf.getvalue()

    def num(x):  # "658,454" -> 658454
        return int(x.replace(',', ''))

    def parse_dedup(platform_name):
        m = re.search(rf"\[DEDUP\]\s*{platform_name}\s*:\s*([\d,]+)\s*[→\-]+\s*([\d,]+)\s*unique", log)
        return (num(m.group(1)), num(m.group(2))) if m else (None, None)

    def balance(y):
        y = np.asarray(y).astype(int)
        nc, ni = int((y == 1).sum()), int((y == 0).sum())
        return {
            'n_interactions': int(len(y)),
            'correct_base_rate': round(float(y.mean()), 4),
            'class_balance_correct_incorrect':
                f"{nc}:{ni} ({nc/len(y)*100:.1f}:{ni/len(y)*100:.1f})",
        }

    n_raw, n_uniq = parse_dedup('NeurIPS')
    a_raw, a_uniq = parse_dedup('ASSISTments')

    return {
        'unified_features': list(UNIFIED_FEATURES),
        'input_dim': int(dim),
        'pipeline_log': log,
        'neurips': {'aligned_pool': n_raw, 'unique_after_dedup': n_uniq, **balance(yn)},
        'assistments': {'aligned_pool': a_raw, 'unique_after_dedup': a_uniq, **balance(ya)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Run + report
# ─────────────────────────────────────────────────────────────────────────────
def main():
    env = collect_environment()
    print("=" * 70); print("§3.9 Computing Environment & Library Versions"); print("=" * 70)
    for k, v in env.items():
        print(f"  {k:16s}: {v}")

    out = {'environment': env}

    print("\n" + "=" * 70); print("Raw dataset-level stats (from CSV, for Table 3.1/3.2 overview)"); print("=" * 70)
    try:
        out['neurips_raw'] = raw_stats_neurips()
        out['assistments_raw'] = raw_stats_assistments()
        for name, d in [('NeurIPS', out['neurips_raw']), ('ASSISTments', out['assistments_raw'])]:
            print(f"\n[{name}]")
            for k, v in d.items():
                print(f"  {k:32s}: {v}")
    except Exception as e:
        print(f"[WARN] raw stats failed: {e}")
        out['raw_error'] = str(e)

    print("\n" + "=" * 70); print("Working-sample stats (via real load_aligned_data pipeline)"); print("=" * 70)
    try:
        ws = working_sample_stats()
        out['working_sample'] = {k: v for k, v in ws.items() if k != 'pipeline_log'}
        print(f"  unified_features ({ws['input_dim']}): {ws['unified_features']}")
        for name in ['neurips', 'assistments']:
            d = ws[name]
            print(f"\n[{name}]  pool {d['aligned_pool']} → unique {d['unique_after_dedup']} "
                  f"→ sample {d['n_interactions']}")
            print(f"  correct_base_rate : {d['correct_base_rate']}")
            print(f"  class_balance     : {d['class_balance_correct_incorrect']}")
    except Exception as e:
        print(f"[WARN] working-sample stats failed: {e}")
        print("  (Ensure data_loader_aligned.py is importable from project root.)")
        out['working_sample_error'] = str(e)

    with open(os.path.join(OUT_DIR, 'ch3_table_data.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved → {os.path.join(OUT_DIR, 'ch3_table_data.json')}")
    print("\nNote: unique students/questions are reported at DATASET level (raw CSV) — "
          "the standard Table 3.1/3.2 'dataset description' quantity. The 30k working "
          "sample contributes #interactions + class balance. If you also need unique-"
          "student counts WITHIN the 30k sample, add `return_ids=True` to load_aligned_data "
          "(one-line patch) and I'll wire it in.")


if __name__ == '__main__':
    main()
