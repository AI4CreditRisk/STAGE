import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, roc_curve)
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import random
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data")

SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

import time
import atexit
from contextlib import contextmanager

# ==============================
# Global runtime statistics
# ==============================
_GLOBAL_START_TIME = time.perf_counter()

def _format_seconds(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s:.2f}s"
    elif m > 0:
        return f"{m}m {s:.2f}s"
    else:
        return f"{s:.2f}s"

def print_total_runtime():
    total_time = time.perf_counter() - _GLOBAL_START_TIME
    print("=" * 60)
    print(f"Total Runtime: {_format_seconds(total_time)}")
    print("=" * 60)

atexit.register(print_total_runtime)

# ==============================
# Code block timer
# ==============================
@contextmanager
def timer(name="Code Block", use_cuda_sync=True):
    try:
        import torch
        if use_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

    start = time.perf_counter()
    print(f"[Timer Start] {name}")

    try:
        yield
    finally:
        try:
            import torch
            if use_cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

        elapsed = time.perf_counter() - start
        print(f"[Timer End] {name}: {_format_seconds(elapsed)}")

def timeit(func):
    def wrapper(*args, **kwargs):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

        start = time.perf_counter()
        print(f"[Function Start] {func.__name__}")

        result = func(*args, **kwargs)

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

        elapsed = time.perf_counter() - start
        print(f"[Function End] {func.__name__}: {_format_seconds(elapsed)}")

        return result

    return wrapper

# ==============================
# Epoch timer
# ==============================
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

class EpochTimer:
    def __init__(self):
        self.times = []
        self._t0 = None

    def start(self):
        _sync()
        self._t0 = time.perf_counter()

    def end(self):
        _sync()
        dt = time.perf_counter() - self._t0
        self.times.append(dt)
        return dt

    def avg(self):
        return sum(self.times) / len(self.times) if self.times else 0.0

    def summary(self):
        print("=" * 60)
        print(f"Average Training Time per Epoch: {self.avg():.4f}s")
        print("=" * 60)

# ==== Dataset selection ====
DATASET = 'internet'
# 'lendingclub' | 'gmsc' | 'internet' | 'creditcard'

DATASET_CONFIG = {
    'lendingclub': {'path': os.path.join(data_dir, "LoanStatus_2017Q1_ml_01.csv")},
    'gmsc':        {'path': os.path.join(data_dir, "cs_training_fe_01.csv")},
    'internet':    {'path': os.path.join(data_dir, "train_internetdata3.csv")},
    'creditcard':  {'path': os.path.join(data_dir, "default of credit card clientsUCI.csv")},
}

# ============================================================
# 1. INPUT : Data loading and cleaning
# ============================================================
def load_and_clean(dataset_name):
    cfg = DATASET_CONFIG[dataset_name]
    path = cfg['path']
    print(f"\n{'='*60}\n[INPUT] Dataset: {dataset_name} | File: {path}\n{'='*60}")

    if dataset_name == 'creditcard':
        try:
            df = pd.read_excel(path, skiprows=1)
        except Exception:
            df = pd.read_csv(path, skiprows=1)
    else:
        df = pd.read_csv(path, low_memory=False)
    print(f"Raw data shape: {df.shape}")

    if dataset_name == 'lendingclub':
        target_col = 'loan_status'
        default_set = {'Charged Off', 'Default', 'Late (31-120 days)',
                       'In Grace Period', 'Late (16-30 days)'}
        good_set = {'Current', 'Fully Paid'}
        df = df[df[target_col].isin(default_set.union(good_set))].copy()
        df[target_col] = df[target_col].apply(lambda x: 1 if x in default_set else 0)

    elif dataset_name == 'gmsc':
        candidates = ['SeriousDlqin2yrs', 'seriousdlqin2yrs', 'target', 'label', 'default']
        target_col = next((c for c in candidates if c in df.columns), None)
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df = df[df[target_col].notna()].copy()
        df[target_col] = df[target_col].astype(int)
        df = df[df[target_col].isin([0, 1])].copy()

    elif dataset_name == 'internet':
        target_col = 'is_default'
        df = df.dropna(subset=[target_col]).copy()
        df[target_col] = df[target_col].astype(int)

    elif dataset_name == 'creditcard':
        target_col = 'default payment next month'
        if target_col not in df.columns:
            cand = [c for c in df.columns if 'default' in c.lower()]
            target_col = cand[0]
        if 'ID' in df.columns:
            df = df.drop(columns=['ID'])

    print(f"Target column identified: {target_col}")
    print(f"Label distribution:\n{df[target_col].value_counts()}")

    if dataset_name in ('lendingclub', 'gmsc'):
        leakage_cols = [
            'id','member_id','url','desc','emp_title','title',
            'funded_amnt','funded_amnt_inv','out_prncp','out_prncp_inv',
            'total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee',
            'recoveries','collection_recovery_fee','last_pymnt_d','last_pymnt_amnt',
            'next_pymnt_d','last_credit_pull_d','pymnt_plan','initial_list_status','policy_code',
            'hardship_flag','hardship_type','hardship_reason','hardship_status',
            'deferral_term','hardship_amount','hardship_start_date','hardship_end_date',
            'payment_plan_start_date','hardship_length','hardship_dpd','hardship_loan_status',
            'orig_projected_additional_accrued_interest','hardship_payoff_balance_amount',
            'hardship_last_payment_amount','debt_settlement_flag','debt_settlement_flag_date',
            'settlement_status','settlement_date','settlement_amount','settlement_percentage',
            'settlement_term','disbursement_method','revol_bal_joint','sec_app_earliest_cr_line',
            'sec_app_inq_last_6mths','sec_app_mort_acc','sec_app_open_acc','sec_app_revol_util',
            'sec_app_open_act_il','sec_app_num_rev_accts','sec_app_chargeoff_within_12_mths',
            'sec_app_collections_12_mths_ex_med','sec_app_mths_since_last_major_derog',
        ]
    elif dataset_name == 'internet':
        leakage_cols = ['loan_id','user_id','title','policy_code',
                        'early_return','early_return_amount','early_return_amount_3mon']
    else:
        leakage_cols = []

    leakage_exist = [c for c in leakage_cols if c in df.columns]
    if leakage_exist:
        df = df.drop(columns=leakage_exist)
    print(f"Dropped {len(leakage_exist)} leakage columns, remaining {df.shape[1]} columns")

    for col in ['int_rate', 'revol_util']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('%','',regex=False), errors='coerce')

    if 'term' in df.columns:
        df['term'] = pd.to_numeric(df['term'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    if 'interest' in df.columns:
        df['interest'] = pd.to_numeric(df['interest'].astype(str).str.replace('%','',regex=False), errors='coerce')
    if 'recircle_util' in df.columns:
        df['recircle_util'] = pd.to_numeric(df['recircle_util'].astype(str).str.replace('%','',regex=False), errors='coerce')

    emp_map = {'< 1 year':0,'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5,
               '6 years':6,'7 years':7,'8 years':8,'9 years':9,'10+ years':10}
    if 'emp_length' in df.columns:
        df['emp_length'] = df['emp_length'].map(emp_map)
    if 'work_year' in df.columns:
        df['work_year'] = df['work_year'].map(emp_map)

    if dataset_name in ('lendingclub', 'gmsc'):
        if 'issue_d' in df.columns and 'earliest_cr_line' in df.columns:
            issue_d = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
            earliest_cr = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
            df['credit_history_months'] = ((issue_d - earliest_cr).dt.days / 30.44).fillna(0).clip(lower=0)
        else:
            df['credit_history_months'] = 0

        inq_12 = df.get('inq_last_12m', pd.Series(0, index=df.index)).fillna(0)
        inq_6  = df.get('inq_last_6mths', pd.Series(0, index=df.index)).fillna(0)
        df['inq_acceleration'] = np.where(inq_12>0, inq_6/(inq_12+1e-8), 0)

        open_24 = df.get('acc_open_past_24mths', pd.Series(0, index=df.index)).fillna(0)
        open_6  = df.get('open_acc_6m', pd.Series(0, index=df.index)).fillna(0)
        df['open_acceleration'] = np.where(open_24>0, open_6/(open_24+1e-8), 0)

        for c in ['issue_d','earliest_cr_line']:
            if c in df.columns:
                df.drop(columns=c, inplace=True)

    if dataset_name == 'internet':
        drop_date_cols = ['earlies_credit_mon','issue_month','year_of_loan','issue_date',
                          'loan_id','user_id','title','policy_code',
                          'early_return','early_return_amount','early_return_amount_3mon']
        drop_date_exist = [c for c in drop_date_cols if c in df.columns]
        df = df.drop(columns=drop_date_exist)

    cat_features = df.select_dtypes(include=["object","bool","category"]).columns
    for feat in cat_features:
        if feat != target_col:
            df[feat] = df[feat].astype('category').cat.codes

    target = df[target_col].values.astype(np.int64)
    features = df.drop(columns=[target_col]).copy()

    for c in features.columns:
        features[c] = pd.to_numeric(features[c], errors='coerce')
    features = features.fillna(0)

    scaler = StandardScaler()
    features[features.columns] = scaler.fit_transform(features[features.columns])
    print(f'Number of features after standardization: {features.shape[1]}')

    return features.values, target

# ============================================================
# 2. Multi-stage feature selection
# ============================================================
def multi_stage_feature_selection(X, y, var_thresh=0.01, corr_thresh=0.95, top_k_ratio=0.6):
    vt = VarianceThreshold(threshold=var_thresh)
    X_v = vt.fit_transform(X)

    corr = np.corrcoef(X_v, rowvar=False)
    upper = np.triu(np.abs(corr), k=1)
    drop_cols = set()
    for i in range(upper.shape[0]):
        for j in range(i+1, upper.shape[1]):
            if upper[i, j] > corr_thresh:
                drop_cols.add(j)
    keep_mask = np.array([i not in drop_cols for i in range(X_v.shape[1])])
    X_c = X_v[:, keep_mask]

    k = max(5, int(X_c.shape[1] * top_k_ratio))
    mi = mutual_info_classif(X_c, y, random_state=SEED)
    top_idx = np.argsort(mi)[-k:]
    X_final = X_c[:, top_idx]
    print(f"[Stage-1] Feature selection: {X.shape[1]} -> {X_final.shape[1]}")
    return X_final

# ============================================================
# 3. Feature attention enhancement
# ============================================================
class FeatureAttention(nn.Module):
    def __init__(self, n_features, hidden_dim=64):
        super().__init__()
        self.W = nn.Linear(n_features, hidden_dim)
        self.v = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        e = torch.tanh(self.W(x))
        scores = self.v(e)
        alpha = F.softmax(scores, dim=1)
        return x * alpha

# ============================================================
# 4. Adaptive neural network
# ============================================================
class AdaptiveNN(nn.Module):
    def __init__(self, n_features, hidden=(256, 64), dropout=0.3):
        super().__init__()
        self.attn = FeatureAttention(n_features, hidden_dim=64)
        layers, in_dim = [], n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, 1)

    def forward(self, x):
        x_att = self.attn(x)
        return self.out(self.mlp(x_att)).squeeze(-1)

# ============================================================
# 5. Dynamic gate
# ============================================================
class DynamicGate(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.gate = nn.Linear(n_features, 1, bias=False)

    def forward(self, x):
        return torch.sigmoid(self.gate(x)).squeeze(-1)

# ============================================================
# 6. SMOTE with feature perturbation + record nearest neighbors
# ============================================================
def perturbation_smote(X, y, sigma=0.01, random_state=SEED):
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)

    n_original = len(y)
    n_synth = len(y_res) - n_original
    minority_cls = int(np.argmin(np.bincount(y)))

    rng = np.random.RandomState(random_state)
    synth_nn_targets = None

    if n_synth > 0:
        noise = rng.normal(0, sigma, size=X_res[n_original:].shape)
        X_res[n_original:] += noise

        minority_X = X[y == minority_cls]
        nn = NearestNeighbors(n_neighbors=1).fit(minority_X)
        _, idx = nn.kneighbors(X_res[n_original:])
        synth_nn_targets = minority_X[idx.flatten()]

    return X_res, y_res, n_original, synth_nn_targets

# ============================================================
# 7. Cost-sensitive BCE
# ============================================================
class CostSensitiveBCE(nn.Module):
    def __init__(self, y_train):
        super().__init__()
        counts = np.bincount(y_train)
        p = counts / counts.sum()
        w = 1.0 / np.sqrt(p + 1e-12)
        self.register_buffer('w', torch.tensor(w, dtype=torch.float32))

    def forward(self, logits, y):
        p = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
        weights = self.w[y.long()]
        loss = -(y * torch.log(p) + (1 - y) * torch.log(1 - p))
        return (weights * loss).mean()

# ============================================================
# 8. HBA-LGBM training
# ============================================================
def train_hba_lgbm_fold(X_train, y_train, X_test, y_test, gamma=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_tr, y_tr, n_orig, synth_nn = perturbation_smote(X_train, y_train)

    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 25,
        'max_depth': 8,
        'min_data_in_leaf': 200,
        'feature_fraction': 1.0,
        'bagging_fraction': 0.5,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': SEED,
    }

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    lgb_model = lgb.train(lgb_params, dtrain, num_boost_round=20)
    p_lgb_tr   = lgb_model.predict(X_tr)
    p_lgb_test = lgb_model.predict(X_test)

    nn_model = AdaptiveNN(X_tr.shape[1], hidden=(1024, 64), dropout=0.3).to(device)
    gate     = DynamicGate(X_tr.shape[1]).to(device)
    criterion = CostSensitiveBCE(y_tr).to(device)
    optimizer = torch.optim.Adam(
        list(nn_model.parameters()) + list(gate.parameters()),
        lr=3e-3, weight_decay=1e-5)

    X_tr_t   = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t   = torch.tensor(y_tr, dtype=torch.float32)
    plgb_tr_t = torch.tensor(p_lgb_tr, dtype=torch.float32)
    idx_t    = torch.arange(len(y_tr))

    train_ds = TensorDataset(X_tr_t, y_tr_t, plgb_tr_t, idx_t)
    loader = DataLoader(train_ds, batch_size=1024, shuffle=True)

    if synth_nn is not None:
        synth_nn_full = torch.tensor(synth_nn, dtype=torch.float32).to(device)
    else:
        synth_nn_full = None

    epoch_timer = EpochTimer()

    for epoch in range(20):
        epoch_timer.start()

        nn_model.train(); gate.train()
        for xb, yb, plgb, idx in loader:
            xb, yb, plgb = xb.to(device), yb.to(device), plgb.to(device)
            optimizer.zero_grad()

            logits_nn = nn_model(xb)
            p_nn = torch.sigmoid(logits_nn)

            lam = gate(xb)

            p_fuse = lam * plgb + (1 - lam) * p_nn
            p_fuse = p_fuse.clamp(1e-7, 1 - 1e-7)
            fuse_logits = torch.log(p_fuse / (1 - p_fuse))

            loss = criterion(fuse_logits, yb)

            if synth_nn_full is not None:
                mask_synth = (idx >= n_orig)
                if mask_synth.any():
                    x_new = xb[mask_synth]
                    x_nn  = synth_nn_full[idx[mask_synth] - n_orig]
                    reg   = ((x_new - x_nn) ** 2).sum(dim=1).mean()
                    loss  = loss + gamma * reg

            loss.backward()
            optimizer.step()

        epoch_time = epoch_timer.end()
        print(f"Epoch {epoch+1:03d} | EpochTime {epoch_time:.2f}s")

    epoch_timer.summary()

    nn_model.eval(); gate.eval()
    with torch.no_grad():
        xt = torch.tensor(X_test, dtype=torch.float32).to(device)
        p_nn_test = torch.sigmoid(nn_model(xt)).cpu().numpy()
        lam_test  = gate(xt).cpu().numpy()

    p_test = lam_test * p_lgb_test + (1 - lam_test) * p_nn_test
    print(f"    λ distribution: mean={lam_test.mean():.4f}, std={lam_test.std():.4f}, "
          f"min={lam_test.min():.4f}, max={lam_test.max():.4f}")
    return p_test

# ============================================================
# 9. Evaluation metrics
# ============================================================
def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    gini = 2 * auc - 1
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    type1 = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    type2 = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = np.max(tpr - fpr)

    return auc, acc, prec, rec, f1, gini, type1, type2, ks, cm

# ============================================================
# 10. Main: single stratified split
# ============================================================
def main():
    with timer("Load & Clean"):
        X, y = load_and_clean(DATASET)

    with timer("Feature Selection"):
        X_sel = multi_stage_feature_selection(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y,
        test_size=0.2,
        stratify=y,
        random_state=SEED
    )
    print(f"\n[Split] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Train label distribution: {np.bincount(y_train)}")
    print(f"  Test  label distribution: {np.bincount(y_test)}")

    with timer("HBA-LGBM Training"):
        y_prob = train_hba_lgbm_fold(X_train, y_train, X_test, y_test)

    auc, acc, prec, rec, f1, gini, type1, type2, ks, cm = evaluate(y_test, y_prob, threshold=0.5)

    print("\n" + "=" * 60)
    print(f"[Result on {DATASET}] (seed={SEED})")
    print("=" * 60)
    print(f"  AUC        : {auc:.4f}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1         : {f1:.4f}")
    print(f"  Gini       : {gini:.4f}")
    print(f"  KS         : {ks:.4f}")
    print(f"  Type I Err : {type1:.4f}")
    print(f"  Type II Err: {type2:.4f} ")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]:<6d} FP={cm[0,1]:<6d}")
    print(f"    FN={cm[1,0]:<6d} TP={cm[1,1]:<6d}")
    print("=" * 60)

if __name__ == "__main__":
    main()