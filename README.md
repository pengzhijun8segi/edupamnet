# edupamnet
# EduPAMNet

**Platform-Aware Multi-Task Learning for Cross-Platform Student Performance Prediction**

This repository contains the implementation accompanying the doctoral thesis *EduPAMNet: Platform-Aware Multi-Task Learning for Cross-Platform Student Performance Prediction*. It provides the data-processing pipeline, the model, and the full set of experiment and evaluation scripts used to produce the results reported in the thesis.

> **Note on scope of the contribution.** The framework's central positive finding is the **near-complete transferability of its shared representation across platforms** (Pattern Transfer Score ≈ 0.99). EduPAMNet's raw within-platform predictive accuracy is competitive but does **not** exceed strong gradient-boosting baselines, and a controlled audit finds that the platform-specific decoder components add no significant predictive or fairness benefit. The repository and its results are reported in that honest, self-critical spirit.

---

## Overview

EduPAMNet couples a **shared encoder** with **platform-specific decoders** over a five-dimensional, semantically aligned feature space, and is trained with a focal-loss classification objective. It is evaluated across two datasets collected a decade apart — the **NeurIPS 2020 Education Challenge** and **ASSISTments 2009–2010** — along four research objectives: cross-platform predictive performance (RO1), interpretability consistency (RO2), task-relationship stability (RO3), and fairness/robustness (RO4). Three purpose-built metrics are introduced: the Feature Importance Stability Score (FISS), the Explanation Consistency Index (ECI), and the Pattern Transfer Score (PTS).

---

## Repository structure

```
.
├── data_loader02.py                 # Raw loading + feature engineering (both platforms)
├── data_loader_aligned.py           # 5-D semantic alignment, dedup, stratified sampling, scaling
├── optimized_pamnet_implementation.py  # Full EduPAMNet architecture (development implementation)
├── edupamnet_final.py               # Clean reference implementation (final protocol only)
│
├── step2_no_adversarial_2.py        # RO1 — final focal-loss-only training + baselines
├── step3_ro2_interpretability.py    # RO2 — SHAP/LIME; computes FISS and ECI
├── step4_ro3_task_correlation.py    # RO3 — task-relationship stability (NeurIPS-specific)
├── step5_pts_computation.py         # PTS — representation transfer via linear probes
├── step6_ro4_formal2.py             # RO4 — robustness (noise, FGSM) and fairness by ability strata
├── step7_eci_redundancy_test.py     # ECI redundancy ablation (5-D vs 4-D)
├── step8_aware_vs_agnostic.py       # Architectural-complexity audit (aware vs agnostic)
├── step9_supplementary_analyses.py  # Supplementary inferential tests (paired/permutation)
├── step1_ro4_verify3.py             # Environment / pipeline smoke-test (not a results script)
│
├── requirements.txt
├── LICENSE
└── README.md
```

A separate `config.py` from earlier development is **deprecated** and is not used by the final pipeline; all authoritative hyperparameters are those in `step2_no_adversarial_2.py` and `optimized_pamnet_implementation.py`.

---

## Requirements

Developed and tested with **Python 3.9.12** and the following core packages:

```
torch==1.12.1          # CUDA 11.6 build
numpy==1.21.5
pandas==1.4.2
scikit-learn==1.0.2
xgboost==3.10.0
shap==0.41.0
lime==0.2.0.1
```

Install with:

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for training but not required.

---

## Datasets

The datasets are **not redistributed** here. Obtain them from their original sources and place them under `./data/`:

- **NeurIPS 2020 Education Challenge** —https://eedi.com/projects/neurips-education-challenge
              data dictory（arXiv）：https://arxiv.org/abs/2007.12061
- **ASSISTments 2009–2010** — https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data

The pipeline aligns both platforms to five shared features — `student_ability`, `question_difficulty`, `historical_accuracy`, `streak_correct`, `streak_incorrect` — then deduplicates, draws a label-stratified sample of 30,000 interactions per platform, and standardises per platform. See the thesis (Appendix C) for full preprocessing detail.

---

## Reproducing the experiments

All scripts use the five fixed random seeds `[42, 123, 7, 2026, 999]`.

```bash
# 0. (optional) verify the environment and pipeline
python step1_ro4_verify3.py

# 1. RO1 — train EduPAMNet and the baselines (produces the saved models the later steps use)
python step2_no_adversarial_2.py

# 2. RO2 / RO3 / PTS / RO4 and the supplementary analyses
python step3_ro2_interpretability.py
python step4_ro3_task_correlation.py
python step5_pts_computation.py
python step6_ro4_formal2.py
python step7_eci_redundancy_test.py
python step8_aware_vs_agnostic.py
python step9_supplementary_analyses.py
```

Step 2 must be run first, as several later steps load the models it saves. Results are written to `./results/`.

**Reproducibility note.** The reported numbers were produced by the full development implementation (`optimized_pamnet_implementation.py`), in which the components removed during development were present but dormant. Because instantiating fewer modules changes the random-initialisation order, running the trimmed `edupamnet_final.py` will not reproduce the reported numbers bit-for-bit; it is provided as a clean, faithful statement of the final architecture and training objective.

---

## Custom metrics

- **FISS** — Spearman rank correlation between the two platforms' SHAP feature-importance vectors (cross-platform importance stability).
- **ECI** — Spearman rank correlation between SHAP and LIME importance vectors within a platform (explanation-method agreement).
- **PTS** — ratio of a transfer probe's F1 to a native probe's F1 on the target platform (transferability of the shared representation).

Formal definitions are given in the thesis (Appendix D).

---

## Citation

If you use this code, please cite the thesis:

```bibtex

@phdthesis{peng2026edupamnet,
  title   = {EduPAMNet: Platform-Aware Multi-Task Learning for Cross-Platform Student Performance Prediction},
  author  = {Peng},
  school  = {SEGi University},
  address = {Kuala Lumpur, Malaysia},
  year    = {2026},
  type    = {thesis}
}
```

---

## License

Released under the terms of the license in [`LICENSE`](LICENSE).
