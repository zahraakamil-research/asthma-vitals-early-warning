# Early warning of severe asthma deterioration from vital signs

Code for the paper:

> Z. K. Sadoon and H. H. Jabbar, "Early Warning of Severe Asthma Deterioration from Vital Signs via Decision-Level Fusion of GINA 2026 Rules and a Conformer Model", submitted to the *International Journal of Intelligent Engineering and Systems*, 2026.

The approach has two parallel branches whose decisions are fused by reporting the more severe one:

1. **Instantaneous GINA 2026 rule.** It is applied to every reading, at all ages. The result is urgent if SpO2 < 92 % or RR > 30 /min (RR > 40 /min at age ≤ 5), and warning if SpO2 is 92–93 %.
2. **Conformer early warning.** It is used for adults with 12 h of hourly HR, RR and SpO2 plus age. It estimates the risk that a sustained GINA 2026 severe episode starts within the next 2 h. The warning threshold is 0.30 and the urgent threshold is 0.47; both were selected on validation data.

## Data are not included

This repository contains **code only**. Both databases are available from PhysioNet under their own terms:

- **MIMIC-IV v3.1** (development and internal test): https://physionet.org/content/mimiciv/3.1/. Access requires a credentialed PhysioNet account, completion of the required human-subjects training and a signed data use agreement. The study queried it through Google BigQuery.
- **eICU-CRD Demo v2.0.1** (external test): https://physionet.org/content/eicu-crd-demo/2.0.1/.

The following are derived from patient records and are therefore **not** published here:

- patient identifiers and split lists;
- window files;
- model weights;
- per-window probabilities.

Credentialed PhysioNet users may request them from the corresponding author.

## Repository layout

```
config.py                         paths, thresholds, seeds and training settings
src/gina_rule.py                  instantaneous rule, episode criterion, alarm burden
src/data_mimic.py                 BigQuery cohort and hourly vitals, 12 h windows, labels, patient split
src/data_eicu.py                  eICU-CRD Demo cohort and windows with the same build setting
src/models.py                     Conformer, 1D ResNet, PatchTST (all take x: B x 12 x 3 and age)
src/train.py                      two-stage training, checkpoint ensemble, CatBoost
src/evaluate.py                   thresholds, hour/episode metrics, bootstrap CIs, Wilson CI, calibration
src/explain.py                    Integrated Gradients
scripts/00_verify_against_original.py   checks that the data preparation rebuilds the study dataset
scripts/01_build_mimic_dataset.py       Sections 3.2-3.4
scripts/02_train_models.py              Sections 3.5-3.6
scripts/03_evaluate_internal.py         Tables 3-7, Section 4.6
scripts/04_external_eicu.py             Section 4.9, Table 9, eICU part of Table 8
scripts/05_rule_latency_cohort.py       Table 2, Fig. 2 counts, Table 8, Section 4.10
scripts/06_explain_and_figures.py       Section 4.7, Figs. 3-5
```

## How to run (Google Colab)

```python
!git clone https://github.com/zahraakamil-research/asthma-vitals-early-warning
%cd asthma-vitals-early-warning
!pip -q install catboost
from google.colab import drive; drive.mount('/content/drive')
```

Then:

1. Set `GCP_PROJECT` in `config.py` to the Google Cloud project linked to your PhysioNet account.
2. Run the scripts in order:

```python
from google.colab import auth; auth.authenticate_user()   # BigQuery sign-in must run inside the notebook
%run scripts/01_build_mimic_dataset.py
%run scripts/02_train_models.py
%run scripts/03_evaluate_internal.py
%run scripts/04_external_eicu.py
%run scripts/05_rule_latency_cohort.py
%run scripts/06_explain_and_figures.py
```

All derived files are written to `config.WORK` on your own Google Drive.

## Main settings

| Item | Value |
|---|---|
| Cohort | adult ICU stays with ICD-9 493 or ICD-10 J45 (MIMIC-IV v3.1) |
| Signals | HR (220045), RR (220210), SpO2 (220277); plausible ranges 20–250, 1–80, 50–100 |
| Windows | hourly means, forward-fill ≤ 2 h, 12 h input, 2 h horizon |
| Episode label | GINA 2026 severe criterion (SpO2 < 92 % or RR > 30 /min) in both horizon hours |
| Exclusions | windows already meeting the criterion in the last input hour; one-hour (transient) windows removed from training only |
| Standardisation | fixed constants in `config.NORM_STATS` (mean and SD per channel and for age), identical to those of the trained model |
| Split | by patient (`subject_id`), 70 / 15 / 15, seed 42; drawn once over the 5,537 patients with at least one 6 h window at an earlier development stage and kept fixed (`split_patients`) |
| Training | weighted BCE (pos_weight 47.8), Adam 1e-3, batch 512, ≤ 100 epochs, early stopping (patience 10) on validation AUC; then focal loss (α 0.8, γ 2), Adam 1e-4, 10 epochs; ensemble of fine-tuning epochs 3–5 |
| Thresholds | warning = max F1 with recall ≥ 0.80; urgent = lowest threshold with accuracy ≥ 0.95 (validation only) |
| Episode evaluation | 6 h look-back; an alarm is a run of consecutive alarm hours |
| Uncertainty | patient-level bootstrap with 2,000 resamples (stay-level in eICU); Wilson interval for eICU episode recall |

## Environment

The study used Python 3.13, PyTorch 2.14, torchaudio 2.11, scikit-learn 1.6, NumPy 2.4, pandas 2.2 and CatBoost, on Google Colab with an NVIDIA L4 GPU. GPU training is not bit-for-bit deterministic, so retrained models may differ slightly in the last decimal places.

## Reproducibility note

The study was carried out interactively in Colab notebooks. This repository reorganises the cells that produced the reported results into scripts, without changing the methods.

`scripts/00_verify_against_original.py` rebuilds the dataset in memory and compares it with the original study files. Checked on 15 Nov 2026, the rebuilt dataset was identical to the one used in the paper in:

- windows (522,638) and patients (5,334);
- episode, transient and in-episode labels;
- input values (same total);
- patient split;
- age.

The standardisation constants were computed once at an early development stage. They are published in `config.py` exactly as used by the model.

## License

MIT License. See `LICENSE`.

## Contact

Zahraa Kamil Sadoon, Department of Computer Science, University of Sumer, Iraq. Contact through the corresponding-author address given in the paper.
