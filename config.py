"""Shared settings for all scripts (values as used in the paper)."""

# Paths (Google Colab + Google Drive layout used in the study; change to your own)
DRIVE = '/content/drive/MyDrive'
WORK = f'{DRIVE}/asthma-vitals-early-warning'  # all derived files (patient-level: never share)
MIMIC_DIR = f'{WORK}/mimic'
OUT_DIR = f'{WORK}/outputs'                    # checkpoints, probabilities, figures
EICU_DIR = f'{WORK}/eicu'
EICU_RAW = '/content/eicu_demo'                # eICU-CRD Demo csv.gz tables

# Google Cloud project linked to your credentialed PhysioNet account (BigQuery access)
GCP_PROJECT = 'your-gcp-project-id'
MIMIC_HOSP = 'physionet-data.mimiciv_3_1_hosp'
MIMIC_ICU = 'physionet-data.mimiciv_3_1_icu'

# Signals: MIMIC-IV chartevents item identifiers and plausible ranges
VITALS = ['heart_rate', 'resp_rate', 'spo2']
ITEMIDS = {'heart_rate': 220045, 'resp_rate': 220210, 'spo2': 220277}
PLAUSIBLE = {'heart_rate': (20, 250), 'resp_rate': (1, 80), 'spo2': (50, 100)}

# Windows and labels
INPUT_H = 12          # input window (hours)
HORIZON_H = 2         # prediction horizon (hours); episode = criterion met in both hours
FFILL_LIMIT = 2       # forward-fill of short gaps (hours)
MIN_AGE = 18          # adults only for the Conformer branch

# Standardisation constants used by the trained (frozen) model: per-channel mean and SD of HR, RR, SpO2 and of age.
# They were computed once, at an early development stage, on the training windows of the first 12 h build
# (aggregate values only, no patient-level data) and kept fixed for every model in the paper.
NORM_STATS = dict(mu=[86.29158020019531, 19.915254592895508, 96.0790786743164],
                  sd=[16.73603630065918, 5.598541736602783, 2.844135284423828],
                  a_mu=59.624290466308594, a_sd=16.39702796936035)

# Training
SEED = 42             # split, ResNet, PatchTST, CatBoost, bootstrap
CONFORMER_SEED = 0    # seed used for the Conformer run reported in the paper
POS_WEIGHT = 47.8     # negatives / positives in the training set
BATCH = 512
MAX_EPOCHS, PATIENCE, LR = 100, 10, 1e-3
FOCAL_ALPHA, FOCAL_GAMMA, FT_LR, FT_EPOCHS = 0.8, 2.0, 1e-4, 10
ENSEMBLE_EPOCHS = (3, 4, 5)

# Decision rules and evaluation
THRESHOLDS = {'Warning': 0.30, 'Urgent': 0.47}   # selected on validation (see evaluate.pick_thresholds)
LOOKBACK_H = 6
N_BOOT = 2000
